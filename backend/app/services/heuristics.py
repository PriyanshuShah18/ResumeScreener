import re
from typing import Any

from app.core.constants import (
    JD_NON_SKILL_PHRASES,
    JD_ROLE_ALIASES,
    KNOWN_EDUCATION,
    SCORING_SKILL_NOISE as SKILL_NOISE,
    TOOL_CONTEXT_MARKERS,
)
from app.utils.text_utils import normalize_skill, normalize_whitespace
from app.services.ocr import annotate_resume_sections

def split_named_sections(lines: list[str]) -> dict[str, list[str]]:
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

def extract_sections(text: str) -> dict[str, list[str]]:
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

def extract_phrases(text: str, limit: int = 16, max_tokens: int = 6) -> list[str]:
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

def extract_skills(text: str, limit: int = 16) -> list[str]:
    return extract_phrases(text, limit=limit, max_tokens=5)

def sanitize_jd_skill_terms(skills: list[str], limit: int = 16) -> list[str]:
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

def infer_short_jd_context(title: str) -> dict[str, list[str]]:
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

def extract_tools_from_lines(lines: list[str], limit: int = 16) -> list[str]:
    chunks: list[str] = []
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if any(marker in lowered for marker in TOOL_CONTEXT_MARKERS):
            if ":" in line:
                chunks.append(line.split(":", 1)[1])
            if idx + 1 < len(lines):
                chunks.append(lines[idx + 1])
    return extract_skills("\n".join(chunks), limit=limit)

def strip_bullet_prefix(value: str) -> str:
    return re.sub(r"^[\-\*\u2022\s]+", "", value)

def extract_project_lines(lines: list[str], limit: int = 5) -> list[str]:
    projects: list[str] = []
    seen: set[str] = set()
    for line in lines:
        cleaned = normalize_whitespace(strip_bullet_prefix(line))
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

def parse_experience_entries(lines: list[str]) -> list[dict[str, Any]]:
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
                "skills_used": extract_skills(remaining),
            }
            continue

        if current is None:
            current = {"company": "", "title": cleaned, "start_date": "", "end_date": "", "highlights": [], "skills_used": []}
            continue

        current.setdefault("highlights", []).append(cleaned)
        current.setdefault("skills_used", []).extend(extract_skills(cleaned))

    if current:
        entries.append(current)
    return entries[:10]

def parse_education_entries(lines: list[str]) -> list[dict[str, Any]]:
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


def heuristic_job_description(text: str) -> dict[str, Any]:
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

    sections = split_named_sections(lines)
    must_have = extract_skills("\n".join(sections.get("required", [])), limit=16)
    if not must_have:
        must_have = extract_skills(text, limit=12)
    must_have = sanitize_jd_skill_terms(must_have, limit=16)
    good_to_have = sanitize_jd_skill_terms(
        extract_skills("\n".join(sections.get("preferred", [])), limit=14),
        limit=14,
    )
    domain_keywords: list[str] = []

    # Sparse title-only JDs (e.g. "AI/ML Developer") need minimal domain context
    # so ranking doesn't become too brittle under heuristic extraction.
    if len(lines) <= 2 and not any(sections.values()):
        inferred = infer_short_jd_context(title)
        good_to_have = sanitize_jd_skill_terms(good_to_have + inferred["good_to_have"], limit=14)
        domain_keywords = sanitize_jd_skill_terms(inferred["domain_keywords"], limit=12)

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

def heuristic_resume(text: str) -> dict[str, Any]:
    """Robust regex-based resume extraction that searches the FULL text."""
    sections = extract_sections(annotate_resume_sections(text))
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
    skills = extract_skills(skills_text, limit=25) if skills_text.strip() else []
    if len(skills) < 5:
        # Broaden: scan the full text for technical terms
        full_skills = extract_skills(text, limit=30)
        seen = set(s.lower() for s in skills)
        for s in full_skills:
            if s.lower() not in seen:
                skills.append(s)
                seen.add(s.lower())
            if len(skills) >= 25:
                break

    tools_text = "\n".join(sections.get("TOOLS", []))
    tools = extract_skills(tools_text, limit=20) if tools_text.strip() else []
    if not tools:
        tools = extract_tools_from_lines(all_lines, limit=20)

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
    experience_entries = parse_experience_entries(exp_lines) if exp_lines else []
    if not experience_entries:
        # Try parsing from full text if sections failed
        experience_entries = parse_experience_entries(all_lines)

    # ── Experience years ──
    years = 0.0
    year_matches = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)", text.lower())
    if year_matches:
        years = max(float(m) for m in year_matches)

    # ── Education ──
    edu_lines = sections.get("EDUCATION", [])
    education_entries = parse_education_entries(edu_lines) if edu_lines else []

    # ── Projects ──
    proj_lines = sections.get("PROJECTS", [])
    projects = extract_project_lines(proj_lines, limit=5) if proj_lines else []

    # ── Certifications ──
    cert_lines = sections.get("CERTIFICATIONS", [])
    certifications = extract_phrases("\n".join(cert_lines), limit=10) if cert_lines else []

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
            "heuristic_confidence": heuristic_confidence(
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
