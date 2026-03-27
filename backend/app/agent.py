from __future__ import annotations

import json
import logging
import re
from typing import Any

from PIL import Image
from pydantic import ValidationError

from app.config import Settings
from app.ocr import annotate_resume_sections
from app.schemas import JobDescriptionData, ResumeData, normalize_skill, normalize_whitespace

try:
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
except ImportError:  # pragma: no cover
    AutoProcessor = None
    Qwen2VLForConditionalGeneration = None

try:
    from qwen_vl_utils import process_vision_info
except ImportError:  # pragma: no cover
    process_vision_info = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

logger = logging.getLogger(__name__)

KNOWN_SKILLS = {
    "react native",
    "ios",
    "ios development",
    "android",
    "android development",
    "swift",
    "kotlin",
    "xcode",
    "expo",
    "redux",
    "firebase",
    "python",
    "java",
    "javascript",
    "typescript",
    "sql",
    "postgresql",
    "mysql",
    "mongodb",
    "fastapi",
    "django",
    "flask",
    "react",
    "node.js",
    "node",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "terraform",
    "git",
    "linux",
    "pandas",
    "numpy",
    "pytorch",
    "tensorflow",
    "scikit-learn",
    "machine learning",
    "deep learning",
    "nlp",
    "rest api",
    "graphql",
    "communication",
    "leadership",
    "agile",
    "scrum",
}

KNOWN_EDUCATION = {
    "b.tech",
    "bachelor",
    "bachelors",
    "master",
    "m.tech",
    "mba",
    "bsc",
    "msc",
    "phd",
    "computer science",
    "engineering",
}


class HRExtractionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.processor = None
        self.load_error = ""
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    @property
    def model_available(self) -> bool:
        return self.model is not None and self.processor is not None

    def preload(self) -> None:
        try:
            self._load_model()
        except Exception as exc:  # pragma: no cover
            self.load_error = str(exc)
            logger.warning("Model preload failed, heuristic extraction will be used: %s", exc)

    def _load_model(self) -> None:
        if self.model_available:
            return
        if not AutoProcessor or not Qwen2VLForConditionalGeneration or not torch:
            raise RuntimeError("transformers/torch dependencies are unavailable")

        logger.info("Loading HR extraction model %s on %s", self.settings.model_id, self.device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.settings.model_id, torch_dtype=dtype)
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.settings.model_id)

    def extract_job_description(self, text: str, max_retries: int = 2) -> JobDescriptionData:
        return self._extract_structured_output(
            input_text=text,
            schema_cls=JobDescriptionData,
            prompt=self.job_description_prompt(text),
            heuristic=self.heuristic_job_description,
            max_retries=max_retries,
        )

    def extract_resume(self, document_text: str, page_images: list[Image.Image] | None = None, max_retries: int = 2) -> ResumeData:
        annotated_text = annotate_resume_sections(document_text)
        return self._extract_structured_output(
            input_text=annotated_text,
            schema_cls=ResumeData,
            prompt=self.resume_prompt(annotated_text, has_images=bool(page_images)),
            heuristic=self.heuristic_resume,
            page_images=page_images or [],
            max_retries=max_retries,
        )

    def _extract_structured_output(
        self,
        input_text: str,
        schema_cls,
        prompt: str,
        heuristic,
        page_images: list[Image.Image] | None = None,
        max_retries: int = 2,
    ):
        page_images = page_images or []
        if not input_text.strip() and not page_images:
            return schema_cls()

        if not input_text.strip() and page_images and not self.model_available:
            raise RuntimeError(
                "OCR text is unavailable and the vision model is not loaded. Install/configure Tesseract or enable the model."
            )

        if not self.model_available:
            if not self.load_error:
                self.load_error = "Model not loaded"
            return schema_cls(**heuristic(input_text))

        current_error = ""
        raw_response = ""
        messages = [self.build_message(prompt, page_images)]

        for attempt in range(max_retries + 1):
            if attempt > 0 and current_error:
                messages.append({"role": "assistant", "content": raw_response})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Your previous output failed validation:\n{current_error}\nReturn corrected JSON only.",
                    }
                )

            raw_response = self.generate_response(messages)
            parsed_json = self.parse_json(raw_response)

            try:
                validated = schema_cls(**parsed_json)
                if not self.has_meaningful_extraction(validated):
                    raise ValueError("Extraction produced an empty or low-signal structured result")
                return self.merge_with_heuristic(schema_cls, validated, input_text, heuristic)
            except (ValidationError, ValueError) as exc:
                current_error = str(exc)
                logger.warning("Structured extraction validation failed: %s", exc)

        if not input_text.strip() and page_images:
            raise RuntimeError("Vision-based extraction failed and no OCR text was available for fallback.")

        logger.warning("Falling back to heuristic extraction after model retries were exhausted")
        return schema_cls(**heuristic(input_text))

    def has_meaningful_extraction(self, extracted) -> bool:
        if isinstance(extracted, ResumeData):
            return any(
                [
                    bool(extracted.name),
                    bool(extracted.email),
                    bool(extracted.phone),
                    bool(extracted.linkedin),
                    bool(extracted.portfolio),
                    bool(extracted.summary),
                    bool(extracted.skills),
                    bool(extracted.tools),
                    bool(extracted.experience_entries),
                    bool(extracted.education_entries),
                    bool(extracted.certifications),
                    bool(extracted.projects),
                    bool(extracted.metadata),
                ]
            )

        if isinstance(extracted, JobDescriptionData):
            return any(
                [
                    bool(extracted.title),
                    bool(extracted.must_have_skills),
                    bool(extracted.good_to_have_skills),
                    bool(extracted.required_education),
                    bool(extracted.required_certifications),
                    bool(extracted.domain_keywords),
                    bool(extracted.responsibilities),
                    extracted.min_years_experience > 0,
                ]
            )

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

    def build_message(self, prompt: str, page_images: list[Image.Image]) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if page_images:
            content.append({"type": "image", "image": self.resize_image(page_images[0])})
        content.append({"type": "text", "text": prompt})
        return {"role": "user", "content": content}

    def generate_response(self, messages: list[dict[str, Any]]) -> str:
        if not self.processor or not self.model:
            raise RuntimeError("Model is not available")

        rendered_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        has_images = any(
            isinstance(message.get("content"), list) and any(part.get("type") == "image" for part in message["content"])
            for message in messages
        )

        if has_images:
            if not process_vision_info:
                raise RuntimeError("qwen-vl-utils is required for image-assisted extraction")
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[rendered_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            inputs = self.processor(text=[rendered_prompt], padding=True, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=1200)
        trimmed_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def parse_json(self, response: str) -> dict[str, Any]:
        try:
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(response[start : end + 1])
            return json.loads(response)
        except json.JSONDecodeError:
            return {}

    def resize_image(self, image: Image.Image) -> Image.Image:
        max_size = 1000
        if image.width <= max_size and image.height <= max_size:
            return image

        ratio = min(max_size / image.width, max_size / image.height)
        return image.resize((int(image.width * ratio), int(image.height * ratio)), Image.Resampling.LANCZOS)

    def job_description_prompt(self, text: str) -> str:
        schema_json = JobDescriptionData.model_json_schema()
        return (
            "You are an HR screening extraction agent. Read the job description and return only valid JSON "
            f"matching this schema: {json.dumps(schema_json)}. "
            "Infer must-have skills, nice-to-have skills, minimum experience, education, certifications, "
            "domain keywords, and concise responsibilities. Use empty strings, 0, or empty arrays when information is absent.\n\n"
            f"Job description:\n{text}"
        )

    def resume_prompt(self, text: str, has_images: bool = False) -> str:
        schema_json = ResumeData.model_json_schema()
        source_instruction = (
            "You are given a resume image and OCR text. OCR may be noisy or incomplete. Use the visible resume image as the primary source and use OCR text as supporting evidence. "
            if has_images
            else "You are given resume text extracted from a document. "
        )
        return (
            "You are an HR resume extraction agent. Read the resume text and return only valid JSON matching "
            f"this schema: {json.dumps(schema_json)}. "
            f"{source_instruction}"
            "Use section markers like [CONTACT_INFO], [SKILLS], [EXPERIENCE], [EDUCATION], and [PROJECTS] when present. "
            "Prefer explicit evidence from the resume content. If a detail is clearly visible in the image but imperfect in OCR, still extract it. "
            "Use empty strings, 0, or empty arrays when information is absent.\n\n"
            f"Resume text:\n{text}"
        )

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
        required_text = "\n".join(sections.get("required", []))
        preferred_text = "\n".join(sections.get("preferred", []))
        education_text = "\n".join(sections.get("education", []))
        certification_text = "\n".join(sections.get("certifications", []))
        responsibility_lines = sections.get("responsibilities", [])

        must_have = self.extract_skills(required_text or text)
        good_to_have = self.extract_skills(preferred_text)
        domain_keywords = self.extract_domain_keywords(title, responsibility_lines, text)

        required_education = [token for token in KNOWN_EDUCATION if token in education_text.lower()]
        required_certifications = self.extract_phrases(certification_text or "", 6)

        if not responsibility_lines:
            responsibility_lines = self.extract_bullets(text, limit=6)

        return {
            "title": title,
            "must_have_skills": must_have,
            "good_to_have_skills": good_to_have,
            "min_years_experience": years,
            "required_education": required_education,
            "required_certifications": required_certifications,
            "domain_keywords": domain_keywords,
            "responsibilities": responsibility_lines,
        }

    def heuristic_resume(self, text: str) -> dict[str, Any]:
        annotated_text = annotate_resume_sections(text)
        sections = self.extract_sections(annotated_text)
        lines = [line for line in annotated_text.splitlines() if line and not line.startswith("[")]

        header_lines = sections.get("CONTACT_INFO", []) or lines[:5]
        header_text = "\n".join(header_lines)

        name = ""
        for line in header_lines:
            if "@" in line.lower() or "linkedin" in line.lower() or "github" in line.lower():
                continue
            if 2 <= len(line.split()) <= 5:
                name = line
                break

        email_match = re.search(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", header_text)
        phone_match = re.search(r"(?:(?:\+\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d[\d\s\-]{7,}\d)", header_text)
        link_matches = re.findall(r"(?i)\b(?:https?://|www\.)\S+\b", header_text + "\n" + "\n".join(sections.get("LINKS", [])))

        location = ""
        for line in header_lines:
            if email_match and email_match.group(0) in line:
                continue
            if phone_match and phone_match.group(0) in line:
                continue
            if re.search(r"\b(?:india|usa|united states|remote|hybrid)\b", line.lower()) or "," in line:
                location = line
                break

        summary = " ".join(sections.get("SUMMARY", [])[:4]).strip()
        skills = self.extract_skills("\n".join(sections.get("SKILLS", [])) or text)
        tools = [skill for skill in skills if skill in {"aws", "azure", "gcp", "docker", "kubernetes", "terraform", "git", "linux"}]
        experience_entries = self.parse_experience_entries(sections.get("EXPERIENCE", []))
        education_entries = self.parse_education_entries(sections.get("EDUCATION", []))
        certifications = self.extract_phrases("\n".join(sections.get("CERTIFICATIONS", [])), 10)
        projects = self.extract_projects(sections.get("PROJECTS", []))

        metadata: dict[str, Any] = {}
        if link_matches:
            metadata["links"] = link_matches

        linkedin = next((link for link in link_matches if "linkedin" in link.lower()), "")
        portfolio = next((link for link in link_matches if "linkedin" not in link.lower()), "")

        return {
            "name": name,
            "email": email_match.group(0) if email_match else "",
            "phone": phone_match.group(0) if phone_match else "",
            "location": location,
            "linkedin": linkedin,
            "portfolio": portfolio,
            "summary": summary,
            "skills": skills,
            "tools": tools,
            "experience_entries": experience_entries,
            "education_entries": education_entries,
            "certifications": certifications,
            "projects": projects,
            "metadata": metadata,
        }

    def split_named_sections(self, lines: list[str]) -> dict[str, list[str]]:
        sections = {"required": [], "preferred": [], "responsibilities": [], "education": [], "certifications": []}
        current = ""
        for line in lines:
            lowered = line.lower().rstrip(":")
            if "required" in lowered and "skill" in lowered:
                current = "required"
                continue
            if any(token in lowered for token in ("preferred", "nice to have", "good to have", "bonus")):
                current = "preferred"
                continue
            if "responsibilit" in lowered:
                current = "responsibilities"
                continue
            if "education" in lowered or "qualification" in lowered:
                current = "education"
                continue
            if "certification" in lowered or "license" in lowered:
                current = "certifications"
                continue
            if current:
                sections[current].append(line)
        return sections

    def extract_sections(self, text: str) -> dict[str, list[str]]:
        sections: dict[str, list[str]] = {}
        current = "CONTACT_INFO"
        sections[current] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1]
                sections.setdefault(current, [])
                continue
            sections.setdefault(current, []).append(line.strip())
        return sections

    def extract_skills(self, text: str) -> list[str]:
        lowered = text.lower()
        found = [skill for skill in KNOWN_SKILLS if skill in lowered]
        if found:
            return sorted({normalize_skill(skill) for skill in found})
        return self.extract_phrases(text, limit=12)

    def extract_domain_keywords(self, title: str, responsibilities: list[str], text: str) -> list[str]:
        combined = f"{title}\n" + "\n".join(responsibilities) + "\n" + text
        candidates = re.findall(r"[a-z][a-z0-9\+\#]{3,}", combined.lower())
        result: list[str] = []
        seen: set[str] = set()
        for token in candidates:
            normalized = normalize_skill(token)
            if not normalized or normalized in seen or normalized in KNOWN_SKILLS or normalized in {"years", "experience"}:
                continue
            seen.add(normalized)
            result.append(normalized)
            if len(result) >= 10:
                break
        return result

    def extract_bullets(self, text: str, limit: int) -> list[str]:
        lines = [normalize_whitespace(line.lstrip("-*•")) for line in text.splitlines()]
        bullets = [line for line in lines if line and len(line.split()) > 4]
        return bullets[:limit]

    def extract_phrases(self, text: str, limit: int) -> list[str]:
        items: list[str] = []
        seen: set[str] = set()
        for chunk in re.split(r"[\n,;/|]+", text):
            cleaned = normalize_skill(chunk)
            if not cleaned or len(cleaned.split()) > 6 or cleaned in seen:
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
            cleaned = normalize_whitespace(line.lstrip("-*•"))
            if not cleaned:
                continue

            date_match = re.search(
                r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)?\.?\s*\d{4}\s*(?:-|to|–)\s*(?:present|current|now|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)?\.?\s*\d{4})",
                cleaned,
            )
            delimiter_entry = "|" in cleaned or " at " in cleaned.lower()
            if date_match or delimiter_entry:
                if current:
                    entries.append(current)
                current = self.build_experience_entry(cleaned, date_match.group(0) if date_match else "")
                continue

            if current is None:
                current = self.build_experience_entry(cleaned, "")
                continue

            current.setdefault("highlights", []).append(cleaned)
            current.setdefault("skills_used", []).extend(self.extract_skills(cleaned))

        if current:
            entries.append(current)

        return entries[:10]

    def build_experience_entry(self, line: str, date_range: str) -> dict[str, Any]:
        remaining = line.replace(date_range, "").strip(" -|,")
        title = ""
        company = ""

        if "|" in remaining:
            parts = [part.strip() for part in remaining.split("|") if part.strip()]
            if len(parts) >= 2:
                title, company = parts[0], parts[1]
            elif parts:
                title = parts[0]
        elif " at " in remaining.lower():
            parts = re.split(r"(?i)\bat\b", remaining, maxsplit=1)
            title = parts[0].strip(" ,-")
            company = parts[1].strip(" ,-") if len(parts) > 1 else ""
        elif "," in remaining:
            parts = [part.strip() for part in remaining.split(",", 1)]
            title = parts[0]
            company = parts[1] if len(parts) > 1 else ""
        else:
            title = remaining

        start_date = ""
        end_date = ""
        if date_range:
            date_parts = re.split(r"(?i)\s*(?:-|to|–)\s*", date_range, maxsplit=1)
            if len(date_parts) == 2:
                start_date, end_date = date_parts

        return {
            "company": company,
            "title": title,
            "start_date": start_date,
            "end_date": end_date,
            "highlights": [],
            "skills_used": self.extract_skills(remaining),
        }

    def parse_education_entries(self, lines: list[str]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for line in lines:
            cleaned = normalize_whitespace(line.lstrip("-*•"))
            if not cleaned:
                continue
            graduation_match = re.search(r"\b(?:19|20)\d{2}\b", cleaned)
            parts = [part.strip() for part in re.split(r"[|,]", cleaned) if part.strip()]
            degree = parts[0] if parts else cleaned
            institution = parts[1] if len(parts) > 1 else ""
            entries.append(
                {
                    "degree": degree,
                    "institution": institution,
                    "field_of_study": "",
                    "graduation_date": graduation_match.group(0) if graduation_match else "",
                    "score": "",
                }
            )
        return entries[:5]

    def extract_projects(self, lines: list[str]) -> list[str]:
        projects: list[str] = []
        for line in lines:
            cleaned = normalize_whitespace(line.lstrip("-*•"))
            if cleaned:
                projects.append(cleaned)
        return projects[:10]
