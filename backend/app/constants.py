"""
Centralized constants for the HR-Screening system.

All shared lookup tables, noise filters, markers, and static mappings live here
so that every module imports from a single source of truth.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Stopwords — filtered out during keyword extraction in scoring
# ---------------------------------------------------------------------------
STOPWORDS: set[str] = {
    "and",
    "with",
    "for",
    "the",
    "that",
    "from",
    "into",
    "your",
    "their",
    "have",
    "will",
    "this",
    "role",
    "team",
    "work",
}

# ---------------------------------------------------------------------------
# Skill noise — generic terms removed when normalizing skill lists for scoring
# ---------------------------------------------------------------------------
SCORING_SKILL_NOISE: set[str] = {
    "experience",
    "years",
    "project",
    "projects",
    "system",
    "platform",
    "development",
    "engineering",
    "application",
}

# ---------------------------------------------------------------------------
# Extraction noise — generic terms filtered during heuristic JD/resume parsing
# ---------------------------------------------------------------------------
EXTRACTION_SKILL_NOISE: set[str] = {
    "required",
    "preferred",
    "must have",
    "good to have",
    "responsibilities",
    "responsibility",
    "qualification",
    "qualifications",
    "ability",
    "abilities",
    "knowledge",
    "experience",
    "years",
    "year",
    "strong",
    "excellent",
    "team",
    "role",
    "candidate",
}

EXTRACTION_DOMAIN_NOISE: set[str] = EXTRACTION_SKILL_NOISE | {
    "skills",
    "skill",
    "education",
    "certification",
    "certifications",
    "minimum",
    "required skills",
}

# ---------------------------------------------------------------------------
# Seniority mapping — used for career-progression scoring
# ---------------------------------------------------------------------------
SENIORITY_MAP: dict[str, int] = {
    "intern": 1,
    "trainee": 1,
    "junior": 2,
    "associate": 2,
    # Indian tech / FAANG leveling
    "sde-1": 2, "sde 1": 2, "sde1": 2,
    "l3": 2,
    "ic3": 3,
    "engineer": 3,
    "developer": 3,
    "analyst": 3,
    "mts": 3,
    "sde-2": 3, "sde 2": 3, "sde2": 3,
    "l4": 3,
    "senior": 4,
    "sse": 4,
    "sde-3": 4, "sde 3": 4, "sde3": 4,
    "mts-2": 4,
    "l5": 4,
    "ic4": 4,
    "staff": 5,
    "lead": 5,
    "principal": 5,
    "architect": 5,
    "tl": 5,
    "mts-3": 5,
    "l6": 5,
    "ic5": 5,
    "manager": 6,
    "head": 6,
    "director": 6,
    "em": 6,
    "l7": 6,
    "ic6": 6,
}
# ---------------------------------------------------------------------------
# Known education tokens — matched during heuristic JD education extraction
# ---------------------------------------------------------------------------
KNOWN_EDUCATION: set[str] = {
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

# ---------------------------------------------------------------------------
# Skill aliases — canonical forms used by the semantic matcher
# ---------------------------------------------------------------------------
SKILL_ALIASES: dict[str, str] = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "reactjs": "react",
    "node": "node.js",
    "nodejs": "node.js",
    "js": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
    "py": "python",
}

# ---------------------------------------------------------------------------
# Context markers — used to identify tool/soft-skill lines in heuristic parsing
# ---------------------------------------------------------------------------
TOOL_CONTEXT_MARKERS: tuple[str, ...] = (
    "tool",
    "tools",
    "technology",
    "technologies",
    "framework",
    "frameworks",
    "platform",
    "platforms",
    "tech stack",
    "stack",
)

SOFT_SKILL_MARKERS: tuple[str, ...] = (
    "communication",
    "leadership",
    "collaboration",
    "stakeholder",
    "problem solving",
    "ownership",
    "mentoring",
    "teamwork",
    "adaptability",
)

# ---------------------------------------------------------------------------
# Resume section patterns — markers for annotating OCR / parsed resume text
# ---------------------------------------------------------------------------
SECTION_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("[SUMMARY]", ("professional summary", "profile summary", "summary", "objective", "about me")),
    ("[SKILLS]", ("skills", "technical skills", "core competencies", "tech stack", "key skills")),
    ("[EXPERIENCE]", ("experience", "work experience", "employment history", "professional experience")),
    ("[EDUCATION]", ("education", "academic background", "qualifications")),
    ("[CERTIFICATIONS]", ("certifications", "licenses", "credentials")),
    ("[PROJECTS]", ("projects", "project experience", "key projects")),
    ("[LINKS]", ("links", "profiles", "portfolio", "online presence")),
]

CONTACT_HINTS: tuple[str, ...] = ("@", "linkedin", "github", "portfolio", "phone", "mobile", "+")

# ---------------------------------------------------------------------------
# Supported file extensions for resume uploads
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

# ---------------------------------------------------------------------------
# Scoring configuration — tunable parameters for the scoring engine
# ---------------------------------------------------------------------------
SCORING_CONFIG: dict[str, float] = {
    "skill_evidence_decay": 0.85,
    "skill_base_weight": 0.60,
    "skill_highlight_weight": 0.60,
    "skill_project_weight": 0.50,
    "skill_listed_weight": 0.75,
    "progression_non_decreasing_weight": 0.65,
    "progression_upward_gain_weight": 0.35,
    "breadth_min_skills": 4,
    "breadth_floor": 0.7,
}
