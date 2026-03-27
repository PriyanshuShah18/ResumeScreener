from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from dateutil import parser
from pydantic import BaseModel, Field, field_validator, model_validator

EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")
PHONE_RE = re.compile(r"(?:(?:\+\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d[\d\s\-]{7,}\d)")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(value: Any) -> str:
    if value is None:
        return ""
    return WHITESPACE_RE.sub(" ", str(value)).strip()


def normalize_skill(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\+\#\./ ]", " ", normalize_whitespace(value).lower())
    return WHITESPACE_RE.sub(" ", cleaned).strip(" .,:;")


def split_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[\n,;/|]+", str(value))

    seen: set[str] = set()
    result: list[str] = []
    for item in raw_items:
        normalized = normalize_skill(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def parse_date_value(value: Any) -> str:
    text = normalize_whitespace(value)
    if not text:
        return ""
    if text.lower() in {"present", "current", "now", "ongoing"}:
        return "Present"
    try:
        parsed = parser.parse(text, fuzzy=True, default=datetime(2000, 1, 1))
        if re.fullmatch(r"\d{4}", text):
            return parsed.strftime("%Y")
        return parsed.strftime("%Y-%m")
    except (ValueError, OverflowError, TypeError):
        return text


def month_index(value: str) -> int | None:
    if not value:
        return None
    if value == "Present":
        today = date.today()
        return today.year * 12 + today.month
    try:
        if re.fullmatch(r"\d{4}", value):
            return int(value) * 12 + 1
        if re.fullmatch(r"\d{4}-\d{2}", value):
            year, month = value.split("-")
            return int(year) * 12 + int(month)
    except ValueError:
        return None
    return None


class ExperienceEntry(BaseModel):
    company: str = Field(default="", description="Employer or organization name")
    title: str = Field(default="", description="Candidate role title")
    start_date: str = Field(default="", description="Role start date")
    end_date: str = Field(default="", description="Role end date or Present")
    duration_months: int = Field(default=0, ge=0, description="Role duration in months")
    highlights: list[str] = Field(default_factory=list, description="Bullets or summary lines for the role")
    skills_used: list[str] = Field(default_factory=list, description="Skills or tools mentioned for the role")

    @field_validator("company", "title", mode="before")
    @classmethod
    def clean_text(cls, value: Any) -> str:
        return normalize_whitespace(value)

    @field_validator("highlights", mode="before")
    @classmethod
    def clean_highlights(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [normalize_whitespace(item) for item in value if normalize_whitespace(item)]
        return [normalize_whitespace(item) for item in str(value).splitlines() if normalize_whitespace(item)]

    @field_validator("skills_used", mode="before")
    @classmethod
    def clean_skills_used(cls, value: Any) -> list[str]:
        return split_tokens(value)

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def clean_dates(cls, value: Any) -> str:
        return parse_date_value(value)

    @model_validator(mode="after")
    def compute_duration(self) -> "ExperienceEntry":
        if self.duration_months:
            return self

        start_value = month_index(self.start_date)
        end_value = month_index(self.end_date)
        if start_value is None or end_value is None or end_value < start_value:
            return self

        self.duration_months = max(end_value - start_value, 0)
        return self


class EducationEntry(BaseModel):
    institution: str = Field(default="", description="School, college, or university")
    degree: str = Field(default="", description="Degree title")
    field_of_study: str = Field(default="", description="Major or specialization")
    graduation_date: str = Field(default="", description="Graduation month or year")
    score: str = Field(default="", description="GPA, percentage, or grade")

    @field_validator("institution", "degree", "field_of_study", "score", mode="before")
    @classmethod
    def clean_text(cls, value: Any) -> str:
        return normalize_whitespace(value)

    @field_validator("graduation_date", mode="before")
    @classmethod
    def clean_date(cls, value: Any) -> str:
        return parse_date_value(value)


class JobDescriptionData(BaseModel):
    title: str = Field(default="", description="Job title or role name")
    must_have_skills: list[str] = Field(default_factory=list, description="Critical skills and technologies")
    good_to_have_skills: list[str] = Field(default_factory=list, description="Optional or preferred skills")
    min_years_experience: float = Field(default=0.0, ge=0, description="Minimum years of experience required")
    required_education: list[str] = Field(default_factory=list, description="Required degrees or education signals")
    required_certifications: list[str] = Field(default_factory=list, description="Required or preferred certifications")
    domain_keywords: list[str] = Field(default_factory=list, description="Domain or business context keywords")
    responsibilities: list[str] = Field(default_factory=list, description="Key responsibilities from the JD")

    @field_validator("title", mode="before")
    @classmethod
    def clean_title(cls, value: Any) -> str:
        return normalize_whitespace(value)

    @field_validator(
        "must_have_skills",
        "good_to_have_skills",
        "required_education",
        "required_certifications",
        "domain_keywords",
        mode="before",
    )
    @classmethod
    def clean_skill_lists(cls, value: Any) -> list[str]:
        return split_tokens(value)

    @field_validator("responsibilities", mode="before")
    @classmethod
    def clean_responsibilities(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [normalize_whitespace(item) for item in value if normalize_whitespace(item)]
        return [normalize_whitespace(item) for item in str(value).splitlines() if normalize_whitespace(item)]

    @field_validator("min_years_experience", mode="before")
    @classmethod
    def clean_years(cls, value: Any) -> float:
        if value is None or value == "":
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        match = re.search(r"(\d+(?:\.\d+)?)", str(value))
        return float(match.group(1)) if match else 0.0


class ResumeData(BaseModel):
    name: str = Field(default="", description="Candidate full name")
    email: str = Field(default="", description="Primary email")
    phone: str = Field(default="", description="Primary phone number")
    location: str = Field(default="", description="Candidate location")
    linkedin: str = Field(default="", description="LinkedIn profile URL")
    portfolio: str = Field(default="", description="Portfolio, GitHub, or personal site URL")
    summary: str = Field(default="", description="Professional summary")
    skills: list[str] = Field(default_factory=list, description="Core candidate skills")
    tools: list[str] = Field(default_factory=list, description="Tools and platforms")
    experience_entries: list[ExperienceEntry] = Field(default_factory=list, description="Experience history")
    education_entries: list[EducationEntry] = Field(default_factory=list, description="Education history")
    certifications: list[str] = Field(default_factory=list, description="Professional certifications")
    projects: list[str] = Field(default_factory=list, description="Project highlights")
    total_years_experience: float = Field(default=0.0, ge=0, description="Total inferred years of experience")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra structured metadata captured during parsing")

    @field_validator("name", "location", "summary", "linkedin", "portfolio", mode="before")
    @classmethod
    def clean_text(cls, value: Any) -> str:
        return normalize_whitespace(value)

    @field_validator("email", mode="before")
    @classmethod
    def clean_email(cls, value: Any) -> str:
        text = normalize_whitespace(value).lower()
        match = EMAIL_RE.search(text)
        return match.group(0) if match else ""

    @field_validator("phone", mode="before")
    @classmethod
    def clean_phone(cls, value: Any) -> str:
        text = normalize_whitespace(value)
        match = PHONE_RE.search(text)
        if not match:
            return ""
        digits = re.sub(r"[^\d+]", "", match.group(0))
        if digits.startswith("++"):
            digits = digits[1:]
        return digits

    @field_validator("skills", "tools", "certifications", mode="before")
    @classmethod
    def clean_skill_lists(cls, value: Any) -> list[str]:
        return split_tokens(value)

    @field_validator("projects", mode="before")
    @classmethod
    def clean_projects(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [normalize_whitespace(item) for item in value if normalize_whitespace(item)]
        return [normalize_whitespace(item) for item in str(value).splitlines() if normalize_whitespace(item)]

    @field_validator("total_years_experience", mode="before")
    @classmethod
    def clean_total_years(cls, value: Any) -> float:
        if value is None or value == "":
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        match = re.search(r"(\d+(?:\.\d+)?)", str(value))
        return float(match.group(1)) if match else 0.0

    @model_validator(mode="after")
    def compute_years_of_experience(self) -> "ResumeData":
        if self.total_years_experience:
            return self

        total_months = sum(entry.duration_months for entry in self.experience_entries)
        if total_months:
            self.total_years_experience = round(total_months / 12, 1)
        return self


class CandidateScore(BaseModel):
    total_score: int = Field(default=0, ge=0, le=100)
    skills_score: int = Field(default=0, ge=0, le=40)
    experience_score: int = Field(default=0, ge=0, le=30)
    education_score: int = Field(default=0, ge=0, le=15)
    keyword_score: int = Field(default=0, ge=0, le=10)
    completeness_score: int = Field(default=0, ge=0, le=5)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendation: str = Field(default="low fit")


class ScreeningResult(BaseModel):
    source_file: str = Field(description="Original resume filename")
    resume_data: ResumeData
    score: CandidateScore
    recruiter_feedback: str = Field(default="", description="Short recruiter-style summary")
    warnings: list[str] = Field(default_factory=list)


class ScreenResumesResponse(BaseModel):
    job_summary: JobDescriptionData
    results: list[ScreeningResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    processed_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
