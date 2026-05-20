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

from app.config import Settings, get_settings
from app.constants import (
    JD_NON_SKILL_PHRASES,
    JD_ROLE_ALIASES,
    EXTRACTION_DOMAIN_NOISE as DOMAIN_NOISE,
    EXTRACTION_SKILL_NOISE as SKILL_NOISE,
    KNOWN_EDUCATION,
    SOFT_SKILL_MARKERS,
    TOOL_CONTEXT_MARKERS,
)
from app.ocr import annotate_resume_sections
from app.schemas import JobDescriptionData, ResumeData, normalize_skill, normalize_whitespace
from app.llm_understanding import llm_service

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
            heuristic=self.heuristic_job_description,
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
            heuristic=self.heuristic_resume,
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

    # ─── Heuristic Extractors (instant, regex-based) ─────────────────────

    def heuristic_job_description(self, text: str) -> dict[str, Any]:
        lines = [normalize_whitespace(line) for line in text.splitlines() if normalize_whitespace(line)]
        lowered_text = text.lower()
        title = ""
        title_match = re.search(r"(?im)^(?:job title|role|position)\s*:\s*(.+)$", text)
        if title_match:
            title = normalize_whitespace(title_match.group(1))
        elif lines:
            title = lines[0]

        years = 0.0
        year_matches = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)", lowered_text)
        if year_matches:
            years = max(float(match) for match in year_matches)

        sections = self.split_named_sections(lines)
        must_have = self.extract_skills("\n".join(sections.get("required", [])), limit=16)
        if not must_have:
            must_have = self.extract_skills(text, limit=12)
        must_have = self.sanitize_jd_skill_terms(must_have, limit=16)
        good_to_have = self.sanitize_jd_skill_terms(
            self.extract_skills("\n".join(sections.get("preferred", [])), limit=14),
            limit=14,
        )
        domain_keywords: list[str] = []

        # Sparse title-only JDs (e.g. "AI/ML Developer") need minimal domain context
        # so ranking doesn't become too brittle under heuristic extraction.
        if len(lines) <= 2 and not any(sections.values()):
            inferred = self.infer_short_jd_context(title)
            good_to_have = self.sanitize_jd_skill_terms(good_to_have + inferred["good_to_have"], limit=14)
            domain_keywords = self.sanitize_jd_skill_terms(inferred["domain_keywords"], limit=12)

        return {
            "title": title,
            "must_have_skills": must_have,
            "good_to_have_skills": good_to_have,
            "min_years_experience": years,
            "required_education": [token for token in KNOWN_EDUCATION if token in text.lower()],
            "required_certifications": [],
            "domain_keywords": domain_keywords,
            "responsibilities": sections.get("responsibilities", [])[:6],
        }

    def heuristic_resume(self, text: str) -> dict[str, Any]:
        """Robust regex-based resume extraction that searches the FULL text."""
        sections = self.extract_sections(annotate_resume_sections(text))
        header_text = "\n".join(sections.get("CONTACT_INFO", [])[:10])

        # ── Search FULL TEXT for contact info (not just header section) ──
        email = re.search(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", header_text)
        if not email:
            email = re.search(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", text)

        phone = re.search(r"(?:(?:\+\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d[\d\s\-]{7,}\d)", header_text)
        if not phone:
            phone = re.search(r"(?:(?:\+\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d[\d\s\-]{7,}\d)", text)

        link_matches = re.findall(r"(?i)\b(?:https?://|www\.)\S+\b", text)
        linkedin = next((link for link in link_matches if "linkedin" in link.lower()), "")
        portfolio = next((link for link in link_matches if "github" in link.lower()), "")
        if not portfolio:
            portfolio = next((link for link in link_matches if "linkedin" not in link.lower()), "")

        # ── Extract name from first non-link, non-email line ──
        name = ""
        all_lines = text.strip().splitlines()
        candidate_lines = sections.get("CONTACT_INFO", [])[:10] or all_lines[:10]
        for line in candidate_lines:
            cleaned = line.strip()
            if not cleaned or len(cleaned) < 3:
                continue
            if "@" in cleaned or "http" in cleaned.lower() or "linkedin" in cleaned.lower():
                continue
            if re.match(r"^[\d\+\(\)\-\s]+$", cleaned):  # Skip phone-only lines
                continue
            tokens = cleaned.split()
            if 1 <= len(tokens) <= 5 and all(t[0].isupper() or t[0] == '.' for t in tokens if t):
                name = cleaned
                break
        if not name:
            for line in all_lines[:5]:
                cleaned = line.strip()
                if cleaned and 2 <= len(cleaned.split()) <= 4 and "@" not in cleaned:
                    name = cleaned
                    break

        # ── Location extraction ──
        location = ""
        loc_match = re.search(
            r"(?im)^\s*(?:location|address|city|based in)\s*[:\-]\s*(.+)$",
            text[:2000],
        )
        if loc_match:
            location = normalize_whitespace(loc_match.group(1))[:80]
        else:
            # Try common patterns like "City, State" or "City, Country"
            loc_match2 = re.search(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)?,\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)?(?:,\s*[A-Z][a-z]+)?)", text[:2000])
            if loc_match2:
                location = loc_match2.group(1).strip()

        # ── Skills: search dedicated section first, then full text ──
        skills_text = "\n".join(sections.get("SKILLS", []))
        skills = self.extract_skills(skills_text, limit=25) if skills_text.strip() else []
        if len(skills) < 5:
            # Broaden: scan the full text for technical terms
            full_skills = self.extract_skills(text, limit=30)
            seen = set(s.lower() for s in skills)
            for s in full_skills:
                if s.lower() not in seen:
                    skills.append(s)
                    seen.add(s.lower())
                if len(skills) >= 25:
                    break

        tools_text = "\n".join(sections.get("TOOLS", []))
        tools = self.extract_skills(tools_text, limit=20) if tools_text.strip() else []
        if not tools:
            tools = self.extract_tools_from_lines(all_lines, limit=20)

        # ── Summary ──
        summary_lines = sections.get("SUMMARY", []) or sections.get("OBJECTIVE", [])
        summary = " ".join(summary_lines[:3]).strip()
        if not summary:
            # Use first paragraph-like block as summary
            for line in all_lines[1:10]:
                cleaned = line.strip()
                if len(cleaned) > 50 and "@" not in cleaned and "http" not in cleaned:
                    summary = cleaned[:300]
                    break

        # ── Experience ──
        exp_lines = sections.get("EXPERIENCE", []) or sections.get("WORK_EXPERIENCE", [])
        experience_entries = self.parse_experience_entries(exp_lines) if exp_lines else []
        if not experience_entries:
            # Try parsing from full text if sections failed
            experience_entries = self.parse_experience_entries(all_lines)

        # ── Experience years ──
        years = 0.0
        year_matches = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)", text.lower())
        if year_matches:
            years = max(float(m) for m in year_matches)

        # ── Education ──
        edu_lines = sections.get("EDUCATION", [])
        education_entries = self.parse_education_entries(edu_lines) if edu_lines else []

        # ── Projects ──
        proj_lines = sections.get("PROJECTS", [])
        projects = self.extract_project_lines(proj_lines, limit=5) if proj_lines else []

        # ── Certifications ──
        cert_lines = sections.get("CERTIFICATIONS", [])
        certifications = self.extract_phrases("\n".join(cert_lines), limit=10) if cert_lines else []

        return {
            "name": name,
            "email": email.group(0) if email else "",
            "phone": phone.group(0) if phone else "",
            "location": location,
            "linkedin": linkedin,
            "portfolio": portfolio,
            "summary": summary,
            "skills": skills,
            "tools": tools,
            "total_years_experience": years,
            "experience_entries": experience_entries,
            "education_entries": education_entries,
            "certifications": certifications,
            "projects": projects,
            "metadata": {
                "heuristic_confidence": self.heuristic_confidence(
                    email=bool(email),
                    phone=bool(phone),
                    skills=skills,
                    tools=tools,
                    experience_entries=experience_entries,
                    education_entries=education_entries,
                    projects=projects,
                )
            },
        }

    # ─── Section & Skill Parsing Utilities ───────────────────────────────

    def split_named_sections(self, lines: list[str]) -> dict[str, list[str]]:
        sections: dict[str, list[str]] = {"required": [], "preferred": [], "responsibilities": [], "education": []}
        current = ""
        for line in lines:
            low = line.lower()
            if "required" in low and "skill" in low:
                current = "required"
                continue
            if any(t in low for t in ("preferred", "nice to have", "good to have", "bonus")):
                current = "preferred"
                continue
            if "responsibil" in low:
                current = "responsibilities"
                continue
            if "education" in low or "qualification" in low:
                current = "education"
                continue
            if current:
                sections[current].append(line)
        return sections

    def extract_sections(self, text: str) -> dict[str, list[str]]:
        sections: dict[str, list[str]] = {}
        current = "CONTACT_INFO"
        for line in text.splitlines():
            if not line.strip():
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1]
                sections.setdefault(current, [])
                continue
            sections.setdefault(current, []).append(line.strip())
        return sections

    def extract_skills(self, text: str, limit: int = 16) -> list[str]:
        return self.extract_phrases(text, limit=limit, max_tokens=5)

    def sanitize_jd_skill_terms(self, skills: list[str], limit: int = 16) -> list[str]:
        sanitized: list[str] = []
        seen: set[str] = set()

        for skill in skills:
            cleaned = normalize_skill(skill)
            if not cleaned:
                continue

            for alias, replacement in JD_ROLE_ALIASES.items():
                cleaned = re.sub(rf"\b{re.escape(alias)}\b", replacement, cleaned)
            cleaned = normalize_whitespace(cleaned)

            if cleaned in JD_NON_SKILL_PHRASES:
                continue
            if cleaned in SKILL_NOISE:
                continue
            if cleaned in seen:
                continue

            seen.add(cleaned)
            sanitized.append(cleaned)
            if len(sanitized) >= limit:
                break

        return sanitized

    def infer_short_jd_context(self, title: str) -> dict[str, list[str]]:
        normalized_title = normalize_skill(title)
        if re.search(r"\b(ai|ml|machine learning|artificial intelligence)\b", normalized_title):
            return {
                "good_to_have": ["python", "deep learning", "natural language processing"],
                "domain_keywords": ["artificial intelligence", "machine learning"],
            }
        if re.search(r"\b(full stack|fullstack)\b", normalized_title):
            return {
                "good_to_have": ["javascript", "react", "node.js"],
                "domain_keywords": ["web development", "backend api", "frontend"],
            }
        return {"good_to_have": [], "domain_keywords": []}

    def extract_tools_from_lines(self, lines: list[str], limit: int = 16) -> list[str]:
        chunks: list[str] = []
        for idx, line in enumerate(lines):
            lowered = line.lower()
            if any(marker in lowered for marker in TOOL_CONTEXT_MARKERS):
                if ":" in line:
                    chunks.append(line.split(":", 1)[1])
                if idx + 1 < len(lines):
                    chunks.append(lines[idx + 1])
        return self.extract_skills("\n".join(chunks), limit=limit)

    def extract_project_lines(self, lines: list[str], limit: int = 5) -> list[str]:
        projects: list[str] = []
        seen: set[str] = set()
        for line in lines:
            cleaned = normalize_whitespace(self.strip_bullet_prefix(line))
            if not cleaned:
                continue
            normalized = cleaned.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            projects.append(cleaned)
            if len(projects) >= limit:
                break
        return projects

    def heuristic_confidence(
        self,
        *,
        email: bool,
        phone: bool,
        skills: list[str],
        tools: list[str],
        experience_entries: list[Any],
        education_entries: list[Any],
        projects: list[str],
    ) -> dict[str, int]:
        contact = 100 if email and phone else 60 if email or phone else 0
        skill_confidence = min((len(skills) + len(tools)) * 12, 100)
        experience_confidence = min(len(experience_entries) * 35, 100)
        education_confidence = min(len(education_entries) * 50, 100)
        project_confidence = min(len(projects) * 25, 100)
        overall = round(
            0.20 * contact
            + 0.30 * skill_confidence
            + 0.25 * experience_confidence
            + 0.15 * education_confidence
            + 0.10 * project_confidence
        )
        return {
            "overall": overall,
            "contact": round(contact),
            "skills": round(skill_confidence),
            "experience": round(experience_confidence),
            "education": round(education_confidence),
            "projects": round(project_confidence),
        }

    def extract_phrases(self, text: str, limit: int = 16, max_tokens: int = 6) -> list[str]:
        items: list[str] = []
        seen: set[str] = set()
        for chunk in re.split(r"[\n,;/|]+", text):
            cleaned = normalize_skill(chunk)
            if not cleaned or cleaned in seen or cleaned in SKILL_NOISE:
                continue
            tokens = cleaned.split()
            if len(tokens) > max_tokens:
                continue
            seen.add(cleaned)
            items.append(cleaned)
            if len(items) >= limit:
                break
        return items

    def parse_experience_entries(self, lines: list[str]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        for line in lines:
            cleaned = normalize_whitespace(line).strip()
            if not cleaned:
                continue

            date_match = re.search(
                r"(?i)\b(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)?\.?\s*\d{4}|\d{4})\s*(?:-|to|\u2013)\s*(?:present|current|now|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)?\.?\s*\d{4}|\d{4})",
                cleaned,
            )
            has_delimiter = "|" in cleaned or " at " in cleaned.lower()

            if date_match or has_delimiter:
                if current:
                    entries.append(current)
                remaining = cleaned.replace(date_match.group(0) if date_match else "", "").strip(" -|,")
                title, company = remaining, ""
                if "|" in remaining:
                    parts = [p.strip() for p in remaining.split("|") if p.strip()]
                    title = parts[0] if parts else remaining
                    company = parts[1] if len(parts) > 1 else ""
                elif " at " in remaining.lower():
                    parts = re.split(r"(?i)\bat\b", remaining, maxsplit=1)
                    title = parts[0].strip()
                    company = parts[1].strip() if len(parts) > 1 else ""

                start_date, end_date = "", ""
                if date_match:
                    dp = re.split(r"(?i)\s*(?:-|to|\u2013)\s*", date_match.group(0), maxsplit=1)
                    if len(dp) == 2:
                        start_date, end_date = dp

                current = {
                    "company": company,
                    "title": title,
                    "start_date": start_date,
                    "end_date": end_date,
                    "highlights": [],
                    "skills_used": self.extract_skills(remaining),
                }
                continue

            if current is None:
                current = {"company": "", "title": cleaned, "start_date": "", "end_date": "", "highlights": [], "skills_used": []}
                continue

            current.setdefault("highlights", []).append(cleaned)
            current.setdefault("skills_used", []).extend(self.extract_skills(cleaned))

        if current:
            entries.append(current)
        return entries[:10]

    def parse_education_entries(self, lines: list[str]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for line in lines:
            cleaned = normalize_whitespace(line).strip()
            if not cleaned:
                continue
            grad_match = re.search(r"\b(?:19|20)\d{2}\b", cleaned)
            parts = [p.strip() for p in re.split(r"[|,]", cleaned) if p.strip()]
            entries.append({
                "degree": parts[0] if parts else cleaned,
                "institution": parts[1] if len(parts) > 1 else "",
                "field_of_study": "",
                "graduation_date": grad_match.group(0) if grad_match else "",
                "score": "",
            })
        return entries[:5]

    def strip_bullet_prefix(self, value: str) -> str:
        return re.sub(r"^[\-\*\u2022\s]+", "", value)
