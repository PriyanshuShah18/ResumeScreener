import json
import os
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import httpx
from app.config import get_settings
from app.schemas import JobDescriptionData, ResumeData, normalize_skill

logger = logging.getLogger(__name__)

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

SKILL_GRAPH_CACHE_FILE = Path(__file__).resolve().parent.parent / "skill_graph_cache.json"
SKILL_GRAPH_RELATIONS = ("equivalent", "parent", "adjacent")


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


def _dedupe_terms(values: list[Any], limit: int = 12) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_skill(str(value))
        if not normalized or normalized in seen or len(normalized) > 80:
            continue
        seen.add(normalized)
        terms.append(normalized)
        if len(terms) >= limit:
            break
    return terms


def normalize_skill_graph_entry(skill: str, value: Any) -> dict[str, Any]:
    relations: dict[str, list[str]] = {name: [] for name in SKILL_GRAPH_RELATIONS}
    if isinstance(value, dict):
        for relation in SKILL_GRAPH_RELATIONS:
            raw_values = value.get(relation) or value.get(f"{relation}s") or []
            if not isinstance(raw_values, list):
                raw_values = [raw_values]
            relations[relation] = _dedupe_terms(raw_values, limit=8)
    elif isinstance(value, list):
        relations["parent"] = _dedupe_terms(value, limit=8)

    base_skill = normalize_skill(skill)
    terms = _dedupe_terms(
        [base_skill]
        + relations["equivalent"]
        + relations["parent"]
        + relations["adjacent"],
        limit=16,
    )
    return {
        "terms": terms or [base_skill],
        "relations": relations,
    }


def skill_graph_terms(value: Any) -> list[str]:
    if isinstance(value, dict):
        raw_terms = value.get("terms")
        if isinstance(raw_terms, list):
            return _dedupe_terms(raw_terms, limit=16)
        relations = value.get("relations")
        if isinstance(relations, dict):
            combined: list[Any] = []
            for relation in SKILL_GRAPH_RELATIONS:
                relation_values = relations.get(relation, [])
                if isinstance(relation_values, list):
                    combined.extend(relation_values)
            return _dedupe_terms(combined, limit=16)
    if isinstance(value, list):
        return _dedupe_terms(value, limit=16)
    if isinstance(value, str):
        return _dedupe_terms([value], limit=1)
    return []


class _RateLimiter:
    """Sliding-window rate limiter."""
    def __init__(self, max_calls: int, window_seconds: float):
        self.max_calls = max_calls
        self.window = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        now = time.monotonic()
        with self._lock:
            while self._timestamps and now - self._timestamps[0] >= self.window:
                self._timestamps.popleft()
            if len(self._timestamps) < self.max_calls:
                self._timestamps.append(now)
                return True
            return False


class LLMUnderstandingService:
    """High-resiliency LLM service with Wait-and-Retry and multi-layer fallback."""

    def __init__(self):
        self.settings = get_settings()
        self.enabled = False
        self._cache_lock = threading.Lock()
        self.groq_client = None
        self.gemini_model = None
        self.openrouter_key = None
        self.ollama_base = str(self.settings.ollama_base_url).rstrip("/")
        self._groq_limiter = _RateLimiter(max_calls=25, window_seconds=60.0)
        self._gemini_limiter = _RateLimiter(max_calls=4, window_seconds=60.0)

        if not _get_bool_env("LLM_UNDERSTANDING_ENABLED", True):
            logger.info("LLM understanding service disabled by LLM_UNDERSTANDING_ENABLED")
            return

        # ── 1. Groq (Fastest) ──
        if self.settings.groq_api_key and Groq:
            try:
                self.groq_client = Groq(
                    api_key=self.settings.groq_api_key,
                    max_retries=1  # Fast-fail so we can move to 8B/Gemini instantly
                )
                self.enabled = True
                logger.info("✅ Groq initialized")
            except Exception: pass

        # ── 2. Gemini Direct ──
        if self.settings.gemini_api_key and genai:
            try:
                genai.configure(api_key=self.settings.gemini_api_key)
                self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
                self.enabled = True
                logger.info("✅ Gemini initialized")
            except Exception: pass

        # ── 3. OpenRouter ──
        self.openrouter_key = self.settings.openrouter_api_key
        if self.openrouter_key and len(str(self.openrouter_key)) > 10:
            self.enabled = True
            logger.info("✅ OpenRouter initialized")

        # ── 4. Local Ollama ──
        if self.ollama_base:
            self.enabled = True
            logger.info("✅ Local Ollama Fallback active at %s", self.ollama_base)

    def _generate_json_with_retry(self, provider_func, provider_name: str, retries: int = 2) -> Any:
        """Retry transient failures, but switch models immediately on rate limits."""
        for attempt in range(retries + 1):
            try:
                return provider_func()
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    logger.warning("%s rate limited; trying the next configured model.", provider_name)
                    raise exc
                if attempt < retries:
                    wait_time = (attempt + 1) * 5
                    logger.warning("⚠️ %s failed. Retrying in %ds... (Attempt %d/%d)", 
                                   provider_name, wait_time, attempt + 1, retries)
                    time.sleep(wait_time)
                    continue
                raise exc
        return None

    def extract_json_from_text(self, text: str, schema: Any) -> Any:
        """Single, authoritative extraction method used by _extract_structured_output."""
        prompt = (
            "Extract information from the following text and return it strictly in JSON format "
            f"according to this schema: {json.dumps(schema) if isinstance(schema, dict) else schema}\n\n"
            f"Text: {text}"
        )
        res, _ = self._generate_json(prompt, None)
        return res

    def _ollama_available_models(self) -> list[str]:
        try:
            with httpx.Client() as client:
                resp = client.get(f"{self.ollama_base}/api/tags", timeout=3.0)
                if resp.status_code != 200:
                    return []
                models = resp.json().get("models", [])
                return [
                    str(item.get("name") or item.get("model") or "").strip()
                    for item in models
                    if str(item.get("name") or item.get("model") or "").strip()
                ]
        except Exception:
            return []

    def _ollama_model_candidates(self) -> list[str]:
        preferred = [
            self.settings.model_id,
            "qwen2.5:32b",
            "qwen2.5:14b",
            "qwen2.5:7b",
            "llama3.2:latest",
            "llama3.1:8b",
            "phi3:14b",
            "qwen2.5:3b",
        ]
        installed = self._ollama_available_models()
        installed_set = set(installed)

        ordered: list[str] = []
        for model in preferred:
            if model and (not installed_set or model in installed_set):
                ordered.append(model)
        for model in installed:
            ordered.append(model)
        return list(dict.fromkeys(ordered))

    def _generate_json(self, prompt: str, default: Any) -> tuple[Any, str]:
        if not self.enabled:
            return default, "Disabled"

        # ── Layer 1: Groq (Tiered: 70B then Qwen then 8B) ──
        if self.groq_client and self._groq_limiter.acquire():
            for model in ["llama-3.3-70b-versatile", "qwen-2.5-32b", "llama-3.1-8b-instant"]:
                try:
                    def _groq_call(m=model, p=prompt):
                        return self.groq_client.chat.completions.create(
                            model=m,
                            messages=[{"role": "user", "content": p}],
                            response_format={"type": "json_object"},
                            timeout=25.0
                        )
                    resp = self._generate_json_with_retry(_groq_call, f"Groq:{model}")
                    if resp:
                        return json.loads(resp.choices[0].message.content), f"Groq:{model}"
                except Exception:
                    continue

        # ── Layer 2: Gemini Direct ──
        if self.gemini_model and self._gemini_limiter.acquire():
            logger.info("🔄 Falling back to Layer 2: Gemini Direct")
            try:
                # Add a 'High Context' hint for fallback models
                fallback_prompt = prompt + "\n\nCRITICAL: You are providing a high-precision extraction. Ensure all entities are captured accurately."
                resp = self.gemini_model.generate_content(
                    fallback_prompt,
                    generation_config=genai.GenerationConfig(response_mime_type="application/json"),
                )
                return json.loads(resp.text), "Gemini:2.5-flash"
            except Exception: pass

        # ── Layer 3: OpenRouter (Multi-model) ──
        if self.openrouter_key and len(str(self.openrouter_key)) > 10:
            logger.info("🔄 Falling back to Layer 3: OpenRouter")
            models = [
                "google/gemini-2.0-flash-001",
                "qwen/qwen-2.5-72b-instruct",
                "meta-llama/llama-3.1-8b-instruct:free",
                "qwen/qwen-2.5-32b-instruct",
                "mistralai/mistral-7b-instruct:free",
                "google/gemini-flash-1.5-8b"
            ]
            for model in models:
                try:
                    with httpx.Client() as client:
                        resp = client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {self.openrouter_key}"},
                            json={
                                "model": model,
                                "messages": [{"role": "user", "content": prompt + "\nAct as a high-precision extraction expert."}],
                                "response_format": {"type": "json_object"}
                            },
                            timeout=25.0
                        )
                        if resp.status_code == 200:
                            return json.loads(resp.json()["choices"][0]["message"]["content"]), f"OpenRouter:{model}"
                except Exception: continue

        # ── Layer 4: Local Ollama (Unbeatable Reliability) ──
        logger.info("🔄 Falling back to Layer 4: Local Ollama")
        models = self._ollama_model_candidates()
        
        for model in models:
            try:
                logger.debug("Trying local model: %s", model)
                with httpx.Client() as client:
                    resp = client.post(
                        f"{self.ollama_base}/api/chat",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": prompt + "\nOutput valid JSON only."}],
                            "stream": False, "format": "json"
                        },
                        timeout=45.0 # Higher timeout for 32B models
                    )
                    if resp.status_code == 200:
                        return json.loads(resp.json()["message"]["content"]), f"Ollama:{model}"
            except Exception: continue

        return default, "None"

    def generate_skill_graph(self, skills: list[str]) -> dict[str, list[str]]:
        if not skills: return {}
        if not self.enabled:
            return {s: [s] for s in skills}
        cache = self._load_cache()
        normalized_skills = [normalize_skill(s) for s in skills if normalize_skill(s)]
        uncached = [s for s in normalized_skills if s not in cache]
        if not uncached:
            return {s: skill_graph_terms(cache[s]) or [s] for s in normalized_skills if s in cache}
        prompt = (
            "Build a conservative skill relation graph. Return STRICTLY a JSON object where each key is an EXACT string "
            "from the provided 'Skills' list. Each value must be an object with optional arrays: "
            "'equivalent' for spelling/synonym equivalents, 'parent' for broader categories, and 'adjacent' for closely related tools. "
            "Only include relationships you are confident are professionally relevant.\n"
            f"Skills: {json.dumps(uncached)}"
        )
        result, _ = self._generate_json(prompt, {})
        if isinstance(result, dict):
            valid_updates = {
                normalize_skill(k): normalize_skill_graph_entry(normalize_skill(k), v)
                for k, v in result.items()
                if normalize_skill(k) in uncached and isinstance(v, (dict, list))
            }
            cache.update(valid_updates)
            self._save_cache(cache)
        return {s: skill_graph_terms(cache.get(s)) or [s] for s in normalized_skills}

    def _load_cache(self) -> dict[str, Any]:
        if SKILL_GRAPH_CACHE_FILE.exists():
            try: return json.loads(SKILL_GRAPH_CACHE_FILE.read_text())
            except: pass
        return {}

    def _save_cache(self, cache: dict):
        with self._cache_lock:
            data = json.dumps(cache, indent=2)
            tmp = SKILL_GRAPH_CACHE_FILE.with_name(
                f"{SKILL_GRAPH_CACHE_FILE.stem}.{os.getpid()}.{threading.get_ident()}.tmp"
            )

            # Windows can briefly lock files under concurrent writes/readers.
            # Use atomic replace and retry a few times before warning.
            for attempt in range(5):
                try:
                    tmp.write_text(data, encoding="utf-8")
                    os.replace(str(tmp), str(SKILL_GRAPH_CACHE_FILE))
                    return
                except Exception as exc:
                    if attempt >= 4:
                        logger.warning("Failed to save skill graph cache (Windows lock?): %s", exc)
                    else:
                        time.sleep(0.05 * (attempt + 1))
                finally:
                    try:
                        if tmp.exists():
                            tmp.unlink()
                    except Exception:
                        pass

llm_service = LLMUnderstandingService()

