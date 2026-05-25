from __future__ import annotations

import json
import logging
import re
import base64
import io
import time
from typing import Any

import requests
from PIL import Image
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.core.constants import (
    JD_NON_SKILL_PHRASES,
    JD_ROLE_ALIASES,
    EXTRACTION_DOMAIN_NOISE as DOMAIN_NOISE,
    EXTRACTION_SKILL_NOISE as SKILL_NOISE,
    KNOWN_EDUCATION,
    SOFT_SKILL_MARKERS,
    TOOL_CONTEXT_MARKERS,
)
from app.services.ocr import annotate_resume_sections
from app.schemas.schemas import JobDescriptionData, ResumeData
from app.utils.text_utils import normalize_skill, normalize_whitespace
from app.services.llm_understanding import llm_service
from app.services.heuristics import heuristic_job_description, heuristic_resume

logger = logging.getLogger(__name__)


def scrub_pii(text: str) -> str:
    """Replace PII (email, phone) with placeholders for external LLM calls."""
    text = re.sub(r'\b[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'(?:\+?\d[\d\s\-\(\)]{8,14}\d)', '[PHONE]', text)
    return text


class HRExtractionService:
    """Production-optimized extraction service.

    Routing strategy:
      - Text-rich documents (≥300 chars) → heuristic regex + Groq/Gemini LLM (fast, ~3-8s)
      - Image-only documents (scanned PDFs) → Ollama VLM (slower, ~30-90s)

    This avoids the Ollama single-request bottleneck for the common case.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.load_error = ""
        # Backward compatibility for existing tests and legacy call sites.
        self.model = None
        self.processor = None
        self._ollama_available = False
        # Check connectivity to Ollama on init
        try:
            resp = requests.get(f"{self.settings.ollama_base_url}/api/tags", timeout=2)
            if resp.ok:
                self._ollama_available = True
        except Exception:
            logger.warning("Ollama not reachable at %s — VLM extraction disabled", self.settings.ollama_base_url)

    @property
    def model_available(self) -> bool:
        return bool(self.model is not None and self.processor is not None) or llm_service.enabled

    def preload(self) -> None:
        pass  # Ollama loads on first request

    # ─── Public Extraction API ───────────────────────────────────────────

    def extract_job_description(self, text: str, max_retries: int = 1) -> JobDescriptionData:
        return self._extract_structured_output(
            input_text=text,
            schema_cls=JobDescriptionData,
            prompt=self.job_description_prompt(text),
            heuristic=heuristic_job_description,
            max_retries=max_retries,
        )

    def extract_resume(
        self,
        document_text: str,
        page_images: list[Image.Image] | None = None,
        max_retries: int = 1,
    ) -> ResumeData:
        annotated_text = annotate_resume_sections(document_text)
        return self._extract_structured_output(
            input_text=annotated_text,
            schema_cls=ResumeData,
            prompt=self.resume_prompt(annotated_text, has_images=bool(page_images)),
            heuristic=heuristic_resume,
            page_images=page_images or [],
            max_retries=max_retries,
        )

    # ─── Core Extraction Logic ───────────────────────────────────────────

    def _extract_structured_output(
        self,
        input_text: str,
        schema_cls,
        prompt: str,
        heuristic,
        page_images: list[Image.Image] | None = None,
        max_retries: int = 1,
    ):
        has_text = len(input_text.strip()) >= 5
        has_images = bool(page_images)

        if not has_text and not has_images:
            return schema_cls()

        if has_images and not has_text and (self.model is None or self.processor is None):
            raise RuntimeError("OCR text is unavailable and the vision model is not loaded")

        t0 = time.perf_counter()

        # ── FAST PATH: Text-rich documents → Heuristic + Groq/Gemini ──
        if has_text:
            # Step 1: Always run the fast heuristic (< 10ms)
            heuristic_data = heuristic(input_text)

            # Step 2: Legacy local model path (kept for backward compatibility/tests)
            if self.model is not None and self.processor is not None:
                try:
                    raw_response = self.generate_response([{"role": "user", "content": prompt}])
                    if raw_response and raw_response.strip():
                        parsed_json = self.parse_json(raw_response)
                        validated = schema_cls(**parsed_json)
                        if self.has_meaningful_extraction(validated):
                            result = self.merge_with_heuristic(schema_cls, validated, input_text, heuristic)
                            elapsed = time.perf_counter() - t0
                            logger.info("✅ Fast-path extraction (Local-Model) in %.1fs", elapsed)
                            self._log_extraction_result(result, "Fast-path:Local-Model")
                            return result
                except Exception as exc:
                    logger.warning("Legacy local model extraction failed, continuing with fallback chain: %s", exc)

            # Step 2: Try Groq/Gemini LLM for higher quality (2-5s)
            if llm_service.enabled:
                try:
                    # Optionally scrub PII from the text sent to external LLMs
                    llm_prompt = prompt
                    if self.settings.scrub_pii_for_llm:
                        llm_prompt = self.resume_prompt(
                            scrub_pii(input_text),
                            has_images=bool(page_images)
                        ) if schema_cls is ResumeData else self.job_description_prompt(scrub_pii(input_text))
                        logger.debug("PII scrubbed for external LLM call")
                    llm_json, provider = llm_service._generate_json(llm_prompt + "\nReturn valid JSON only.", None)
                    if llm_json:
                        # Robustness: If LLM returned a list, try to find a dict in it or fail gracefully
                        if isinstance(llm_json, list):
                            if len(llm_json) > 0 and isinstance(llm_json[0], dict):
                                llm_json = llm_json[0]
                                logger.debug("Extracted first dict from LLM list")
                            else:
                                raise ValueError("LLM returned a list instead of a mapping")

                        validated = schema_cls(**llm_json)
                        if self.has_meaningful_extraction(validated):
                            result = self.merge_with_heuristic(schema_cls, validated, input_text, heuristic)
                            elapsed = time.perf_counter() - t0
                            logger.info("✅ Fast-path extraction (%s) in %.1fs", provider, elapsed)
                            self._log_extraction_result(result, f"Fast-path:{provider}")
                            return result
                except Exception as exc:
                    logger.warning("Structured AI extraction failed, falling back to heuristic: %s", exc)

            elapsed = time.perf_counter() - t0
            logger.info("✅ Heuristic-only extraction in %.1fs", elapsed)
            result = schema_cls(**heuristic_data)
            self._log_extraction_result(result, "Heuristic")
            return result

        # ── SLOW PATH: Image-only documents → Ollama VLM ──
        if has_images and self.model is not None and self.processor is not None:
            messages = [{"role": "user", "content": prompt}]
            base64_image = self.image_to_base64(self.resize_image(page_images[0]))
            messages[0]["images"] = [base64_image]

            current_error = ""
            raw_response = ""

            for attempt in range(max_retries + 1):
                if attempt > 0 and current_error and raw_response.strip():
                    messages.append({"role": "assistant", "content": raw_response})
                    messages.append({
                        "role": "user",
                        "content": f"Your previous output failed validation:\n{current_error}\nReturn corrected JSON only.",
                    })

                try:
                    raw_response = self.generate_response(messages)
                except Exception as exc:
                    logger.error("Ollama VLM request failed: %s", exc)
                    break

                if not raw_response.strip():
                    current_error = "Model returned an empty response"
                    continue

                logger.info("Ollama VLM response (attempt %d): %.300s", attempt, raw_response)

                try:
                    parsed_json = self.parse_json(raw_response)
                    validated = schema_cls(**parsed_json)
                    if self.has_meaningful_extraction(validated):
                        elapsed = time.perf_counter() - t0
                        logger.info("✅ VLM extraction in %.1fs", elapsed)
                        self._log_extraction_result(validated, "VLM:Ollama")
                        return validated
                except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                    current_error = str(exc)
                    logger.warning("VLM extraction attempt %d failed: %s", attempt, exc)

        # ── FINAL FALLBACK ──
        if not has_text and has_images:
            raise RuntimeError("OCR text is unavailable and the vision model is not loaded")

        elapsed = time.perf_counter() - t0
        logger.info("⚠️ Fallback heuristic extraction in %.1fs", elapsed)
        result = schema_cls(**heuristic(input_text))
        self._log_extraction_result(result, "Fallback-Heuristic")
        return result

    def _log_extraction_result(self, result, path: str) -> None:
        """Log structured summary of extraction results for observability."""
        is_jd = isinstance(result, JobDescriptionData)
        skills_count = len(result.must_have_skills) if is_jd else len(getattr(result, 'skills', []))
        exp_count = len(getattr(result, 'experience_entries', []))
        
        logger.info(
            "Extraction result: path=%s fields_populated=%d skills=%d experience_entries=%d",
            path,
            sum(1 for v in result.model_dump().values() if v),
            skills_count,
            exp_count,
        )

    # ─── Ollama Communication ────────────────────────────────────────────

    def generate_ollama_response(self, messages: list[dict[str, Any]]) -> str:
        payload = {
            "model": self.settings.model_id,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 1500,
            },
        }
        
        last_exc = None
        backoff = 1.0
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.settings.ollama_base_url}/api/chat",
                    json=payload,
                    timeout=300,
                )
                response.raise_for_status()
                return response.json().get("message", {}).get("content", "")
            except Exception as exc:
                last_exc = exc
                logger.warning("Ollama attempt %d failed: %s. Retrying in %.1fs...", attempt + 1, exc, backoff)
                time.sleep(backoff)
                backoff *= 2
        
        raise last_exc or Exception("Ollama extraction failed after 3 attempts")

    def generate_response(self, messages: list[dict[str, Any]]) -> str:
        """Backward-compatible wrapper used by tests and legacy call paths."""
        return self.generate_ollama_response(messages)

    # ─── Image Utilities ─────────────────────────────────────────────────

    def image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=80)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def resize_image(self, image: Image.Image) -> Image.Image:
        max_size = 800
        if image.width <= max_size and image.height <= max_size:
            return image
        ratio = min(max_size / image.width, max_size / image.height)
        return image.resize(
            (int(image.width * ratio), int(image.height * ratio)),
            Image.Resampling.LANCZOS,
        )

    # ─── Validation & Merging ────────────────────────────────────────────

    def has_meaningful_extraction(self, extracted) -> bool:
        if isinstance(extracted, ResumeData):
            return any([extracted.name, extracted.email, extracted.skills, extracted.experience_entries])
        if isinstance(extracted, JobDescriptionData):
            return any([extracted.title, extracted.must_have_skills])
        return True

    def merge_with_heuristic(self, schema_cls, extracted, input_text: str, heuristic):
        if not input_text.strip():
            return extracted
        heuristic_model = schema_cls(**heuristic(input_text))
        merged = extracted.model_dump()
        for key, value in heuristic_model.model_dump().items():
            if self.is_empty_value(merged.get(key)) and not self.is_empty_value(value):
                merged[key] = value
        return schema_cls(**merged)

    def is_empty_value(self, value: Any) -> bool:
        return value in ("", None, 0, 0.0, [], {})

    # ─── JSON Parsing ────────────────────────────────────────────────────

    def parse_json(self, response: str) -> dict[str, Any]:
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", response.strip(), count=1)
        stripped = re.sub(r"\n?```\s*$", "", stripped.strip())

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = stripped[start : end + 1]
            cleaned = re.sub(r",(\s*[\}\]])", r"\1", cleaned)
            return json.loads(cleaned)
        return json.loads(stripped)

    # ─── Prompt Templates ────────────────────────────────────────────────

    def job_description_prompt(self, text: str) -> str:
        schema_json = JobDescriptionData.model_json_schema()
        for field in ("implicit_skills", "inferred_seniority", "domain_expectations"):
            schema_json.get("properties", {}).pop(field, None)

        return (
            "You are an expert technical recruiter. Read the job description and return ONLY valid JSON "
            f"matching this schema: {json.dumps(schema_json)}.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "0. Do not restrict extraction to predefined skill lists.\n"
            "1. If the job description is extremely brief (e.g. just a title like 'AI/ML Developer'), you should infer 3-5 "
            "relevant 'good-to-have' skills or 'domain_keywords' that are standard for this role to provide context. "
            "Only place skills in 'must_have_skills' if they are explicitly mentioned or are the absolute core of the title.\n"
            "2. Ensure the extraction is balanced. Do not invent requirements that would unfairly penalize a candidate if missing.\n"
            "3. Extract all explicit skills, domain keywords, and responsibilities.\n"
            "4. Categorize the role into one of these 'archetypes': 'management', 'research', 'analyst', 'finance', 'senior', 'data_ml', 'product', or 'standard'.\n\n"
            f"Job description text:\n{text}"
        )

    def resume_prompt(self, text: str, has_images: bool = False) -> str:
        schema_json = ResumeData.model_json_schema()

        source = "resume image" if has_images else "resume text"
        return (
            f"You are an HR resume extraction agent. Read the {source} and return only valid JSON matching "
            f"this schema: {json.dumps(schema_json)}. "
            "\n\nSTRICT INSTRUCTIONS:\n"
            "1. Use section markers like [CONTACT_INFO], [SKILLS], [EXPERIENCE] etc. as structural hints.\n"
            "2. Extract ALL relevant skills, tools, and certifications.\n"
            "3. Do not limit extraction to predefined categories or static skill lists.\n"
            "4. Extract only the 3 MOST RECENT job experiences and 3 MOST RECENT projects.\n"
            "5. For experience highlights: PRESERVE quantified impact numbers (%, ms, users, $, scale) "
            "and ownership verbs (led, designed, built, shipped, owned). These are critical for scoring. "
            "Truncate generic filler but keep measurable outcomes. Max 120 chars per bullet.\n"
            "\nReturn ONLY a strictly valid JSON object."
            f"\n\nResume text:\n{text}"
        )

    def strip_bullet_prefix(self, value: str) -> str:
        return re.sub(r"^[\-\*\u2022\s]+", "", value)
