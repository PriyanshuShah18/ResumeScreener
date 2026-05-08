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
from app.schemas import JobDescriptionData, ResumeData

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

        # ── 1. Groq (Fastest) ──
        self.groq_client = None
        if self.settings.groq_api_key and Groq:
            try:
                self.groq_client = Groq(
                    api_key=self.settings.groq_api_key,
                    max_retries=1  # Fast-fail so we can move to 8B/Gemini instantly
                )
                self.enabled = True
                logger.info("✅ Groq initialized")
            except Exception: pass

        self._groq_limiter = _RateLimiter(max_calls=25, window_seconds=60.0)

        # ── 2. Gemini Direct ──
        self.gemini_model = None
        self._gemini_limiter = _RateLimiter(max_calls=4, window_seconds=60.0)
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
        self.ollama_base = str(self.settings.ollama_base_url).rstrip("/")
        if self.ollama_base:
            self.enabled = True
            logger.info("✅ Local Ollama Fallback active at %s", self.ollama_base)

    def _generate_json_with_retry(self, provider_func, provider_name: str, retries: int = 2) -> Any:
        """Helper to retry a provider if it hits a rate limit."""
        for attempt in range(retries + 1):
            try:
                return provider_func()
            except Exception as exc:
                if "429" in str(exc) and attempt < retries:
                    wait_time = (attempt + 1) * 5
                    logger.warning("⚠️ %s rate limited. Retrying in %ds... (Attempt %d/%d)", 
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

    def _generate_json(self, prompt: str, default: Any) -> tuple[Any, str]:
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
        # Order: User preference, then Largest/Best models
        models = [
            self.settings.model_id, # Respect user-configured model (default qwen2.5vl:3b)
            "qwen2.5:32b",
            "qwen2.5:14b",
            "llama3.1:8b",
            "qwen2.5:7b",
            "phi3:14b",
            "qwen2.5:3b"
        ]
        # Remove duplicates while preserving order
        models = list(dict.fromkeys(models))
        
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
        cache = self._load_cache()
        uncached = [s for s in skills if s not in cache]
        if not uncached: return {s: cache[s] for s in skills if s in cache}
        prompt = (
            "Group these skills into broader categories. Return STRICTLY a JSON object where each key is an EXACT string "
            "from the provided 'Skills' list, and the value is a list of its parent categories.\n"
            f"Skills: {json.dumps(uncached)}"
        )
        result, _ = self._generate_json(prompt, {})
        if isinstance(result, dict):
            # Strict validation to ensure we don't pollute the cache with malformed LLM keys
            valid_updates = {k: v for k, v in result.items() if k in uncached and isinstance(v, list)}
            cache.update(valid_updates)
            self._save_cache(cache)
        return {s: cache.get(s, [s]) for s in skills}

    def _load_cache(self) -> dict[str, Any]:
        if SKILL_GRAPH_CACHE_FILE.exists():
            try: return json.loads(SKILL_GRAPH_CACHE_FILE.read_text())
            except: pass
        return {}

    def _save_cache(self, cache: dict):
        with self._cache_lock:
            try:
                data = json.dumps(cache, indent=2)
                tmp = SKILL_GRAPH_CACHE_FILE.with_suffix(".tmp")
                tmp.write_text(data)
                if os.path.exists(SKILL_GRAPH_CACHE_FILE):
                    os.remove(SKILL_GRAPH_CACHE_FILE)
                os.rename(str(tmp), str(SKILL_GRAPH_CACHE_FILE))
            except Exception as e:
                logger.warning("Failed to save skill graph cache (Windows lock?): %s", e)

llm_service = LLMUnderstandingService()

