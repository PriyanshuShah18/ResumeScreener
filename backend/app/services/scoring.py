from __future__ import annotations

import math
import re
from typing import Any

from app.core.constants import (
    JD_NON_SKILL_PHRASES,
    JD_ROLE_ALIASES,
    SCORING_SKILL_NOISE as SKILL_NOISE,
    SCORING_CONFIG,
    SENIORITY_MAP,
    STOPWORDS,
)
from app.schemas.schemas import (
    CandidateScore,
    JobDescriptionData,
    ResumeData,
    month_index,
)
from app.utils.text_utils import normalize_skill
from app.utils.nlp_utils import extract_impact_signals, ownership_ratio
from app.services.semantic_matching import SemanticMatcher, canonicalize_skill, get_semantic_matcher


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def normalize_set(values: list[str]) -> set[str]:
    normalized = {canonicalize_skill(value) for value in values if canonicalize_skill(value)}
    return {value for value in normalized if value and value not in SKILL_NOISE}


def sanitize_jd_skill_terms(values: list[str], limit: int | None = None) -> list[str]:
    sanitized: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = normalize_skill(value)
        if not cleaned:
            continue
        for alias, replacement in JD_ROLE_ALIASES.items():
            cleaned = re.sub(rf"\b{re.escape(alias)}\b", replacement, cleaned)
        cleaned = normalize_skill(cleaned)
        if not cleaned or cleaned in JD_NON_SKILL_PHRASES or cleaned in SKILL_NOISE:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        sanitized.append(cleaned)
        if limit is not None and len(sanitized) >= limit:
            break
    return sanitized


def bonus_caps_for_required_match_ratio(
    required_match_ratio: float,
    *,
    base_additional_cap: int = 10,
    base_domain_cap: int = 8,
) -> tuple[int, int]:
    ratio = clamp(required_match_ratio, 0.0, 1.0)
    if ratio == 0.0:
        return min(base_additional_cap, 2), min(base_domain_cap, 1)
    if ratio < 0.34:
        return min(base_additional_cap, 4), min(base_domain_cap, 2)
    if ratio < 0.67:
        return min(base_additional_cap, 6), min(base_domain_cap, 3)
    return max(base_additional_cap, 0), max(base_domain_cap, 0)


# ─── Impact Signal Extraction (reporting only, not scored) ──────────────────


def extract_phrases(value: str, limit: int = 15, max_tokens: int = 4) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for chunk in re.split(r"[\n,;/|]+", value):
        cleaned = canonicalize_skill(chunk)
        if not cleaned or cleaned in seen:
            continue
        token_count = len(cleaned.split())
        if token_count == 0 or token_count > max_tokens:
            continue
        if cleaned in SKILL_NOISE:
            continue
        seen.add(cleaned)
        candidates.append(cleaned)
        if len(candidates) >= limit:
            break
    return candidates


def is_skill_like_additional_term(term: str) -> bool:
    normalized = canonicalize_skill(term)
    if not normalized or normalized in SKILL_NOISE:
        return False

    tokens = normalized.split()
    if not tokens:
        return False

    # Hide project labels from "Additional Relevant Skills" cards.
    # Examples: "chat app", "expense tracker app", "shopping list app".
    project_label_tail_tokens = {
        "app",
        "apps",
        "application",
        "project",
        "projects",
        "website",
        "site",
        "portal",
        "dashboard",
    }
    if len(tokens) >= 2 and tokens[-1] in project_label_tail_tokens:
        return False

    return True


def filter_additional_relevant_skill_matches(
    skills: list[str],
    details: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    filtered_skills: list[str] = []
    allowed: set[str] = set()
    for skill in skills:
        normalized = canonicalize_skill(skill)
        if not is_skill_like_additional_term(normalized):
            continue
        if normalized in allowed:
            continue
        allowed.add(normalized)
        filtered_skills.append(normalized)

    filtered_details: list[dict[str, Any]] = []
    for detail in details:
        resume_skill = canonicalize_skill(detail.get("resume_skill", ""))
        if resume_skill in allowed:
            filtered_details.append({**detail, "resume_skill": resume_skill})

    return filtered_skills, filtered_details


def keyword_set(job: JobDescriptionData) -> set[str]:
    title_tokens = {
        token
        for token in re.findall(r"[a-z0-9\+\#]{3,}", job.title.lower())
        if token not in STOPWORDS
    }
    responsibility_tokens = {
        token
        for token in re.findall(r"[a-z0-9\+\#]{4,}", " ".join(job.responsibilities).lower())
        if token not in STOPWORDS
    }
    merged = list(title_tokens | responsibility_tokens | set(job.domain_keywords))
    return normalize_set(merged)


def candidate_skill_evidence(resume: ResumeData) -> dict[str, float]:
    evidence: dict[str, float] = {}

    def add_terms(terms: list[str], weight: float) -> None:
        for term in terms:
            canonical = canonicalize_skill(term)
            if not canonical or canonical in SKILL_NOISE:
                continue
            previous = evidence.get(canonical, 0.0)
            evidence[canonical] = min(1.0, previous + weight)

    add_terms(resume.skills, 0.6)
    add_terms(resume.tools, 0.6)
    add_terms(resume.certifications, 0.6)

    ordered_entries = sorted_experience_entries(resume)
    decay = SCORING_CONFIG["skill_evidence_decay"]
    for idx, entry in enumerate(reversed(ordered_entries)):
        decay_factor = decay ** idx
        add_terms(entry.skills_used, SCORING_CONFIG["skill_listed_weight"] * decay_factor)
        add_terms(extract_phrases("\n".join(entry.highlights), limit=12), SCORING_CONFIG["skill_highlight_weight"] * decay_factor)

    for idx, project in enumerate(resume.projects):
        decay_factor = decay ** idx
        add_terms(extract_phrases(project, limit=8), SCORING_CONFIG["skill_project_weight"] * decay_factor)

    return evidence


def cluster_skill_evidence(
    skill_evidence: dict[str, float],
    matcher: SemanticMatcher,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, float]]:
    clusters = matcher.cluster_skills(list(skill_evidence.keys()))
    clustered_evidence: dict[str, float] = {}
    cluster_details: list[dict[str, Any]] = []
    term_evidence: dict[str, float] = {}

    for cluster in clusters:
        canonical = cluster["canonical"]
        members = cluster["members"]
        max_weight = max((skill_evidence.get(member, 0.0) for member in members), default=0.0)
        clustered_evidence[canonical] = max_weight
        term_evidence[canonical] = max(term_evidence.get(canonical, 0.0), max_weight)
        for member in members:
            term_evidence[member] = max(term_evidence.get(member, 0.0), max_weight)
        cluster_details.append(
            {
                "canonical": canonical,
                "members": members,
                "max_evidence_weight": round(max_weight, 2),
            }
        )

    return clustered_evidence, cluster_details, term_evidence


def semantic_coverage(reference_terms: list[str], candidate_terms: list[str], matcher: SemanticMatcher) -> float:
    if not reference_terms:
        return 1.0
    if not candidate_terms:
        return 0.0

    credits: list[float] = []
    for term in reference_terms:
        _, similarity = matcher.best_match(term, candidate_terms)
        credits.append(matcher.similarity_credit(similarity))

    return sum(credits) / len(credits) if credits else 0.0


def infer_seniority_level(title: str) -> int:
    lowered = normalize_skill(title)
    if not lowered:
        return 0
    level = 0
    for token, mapped_level in SENIORITY_MAP.items():
        if token in lowered:
            level = max(level, mapped_level)
    return level


def sorted_experience_entries(resume: ResumeData) -> list[Any]:
    indexed_entries: list[tuple[int, int, Any]] = []
    for index, entry in enumerate(resume.experience_entries):
        date_index = month_index(entry.start_date)
        fallback = 999999 if date_index is None else date_index
        indexed_entries.append((fallback, index, entry))
    return [item[2] for item in sorted(indexed_entries, key=lambda value: (value[0], value[1]))]


def progression_score(resume: ResumeData) -> float:
    ordered_entries = sorted_experience_entries(resume)
    levels = [infer_seniority_level(entry.title) for entry in ordered_entries if infer_seniority_level(entry.title) > 0]
    if not levels:
        return 0.0
    if len(levels) == 1:
        # Scale by seniority: Intern(1)→0.52, Senior(4)→0.88, Staff(5)→1.0
        return clamp(0.4 + (levels[0] / 5.0) * 0.6, 0.0, 1.0)

    non_decreasing = sum(1 for i in range(1, len(levels)) if levels[i] >= levels[i - 1]) / (len(levels) - 1)
    upward_gain = max(levels[-1] - levels[0], 0) / 5.0
    return clamp(
        SCORING_CONFIG["progression_non_decreasing_weight"] * non_decreasing +
        SCORING_CONFIG["progression_upward_gain_weight"] * upward_gain,
        0.0, 1.0,
    )


def experience_domain_terms(resume: ResumeData) -> list[str]:
    terms = set(candidate_skill_evidence(resume).keys())
    for entry in resume.experience_entries:
        terms.update(extract_phrases(entry.title, limit=5))
        terms.update(extract_phrases("\n".join(entry.highlights), limit=10))
    terms.update(extract_phrases(resume.summary, limit=10))
    for project in resume.projects:
        terms.update(extract_phrases(project, limit=8))
    return sorted(terms)


def _append_unique(target: list[str], values: list[str]) -> None:
    seen = set(target)
    for value in values:
        normalized = normalize_skill(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        target.append(normalized)


def _education_entry_terms(entry: Any) -> list[str]:
    terms: list[str] = []
    degree = normalize_skill(entry.degree)
    field = normalize_skill(entry.field_of_study)
    institution = normalize_skill(entry.institution)

    for value in (degree, field, institution):
        _append_unique(terms, extract_phrases(value, limit=12, max_tokens=8))

    if degree and field:
        _append_unique(terms, [f"{degree} {field}"])

    combined = normalize_skill(f"{entry.degree} {entry.field_of_study} {entry.institution}")
    _append_unique(terms, extract_phrases(combined, limit=12, max_tokens=10))

    has_bachelor = any(token in degree for token in ("b.tech", "b.e", "bachelor", "bachelors"))
    has_master = any(token in degree for token in ("m.tech", "m.e", "master", "masters", "msc", "m.sc"))

    if "cse" in field:
        _append_unique(terms, ["computer science", "computer science engineering"])
    if "ai" in field.split() or "artificial intelligence" in field:
        _append_unique(terms, ["artificial intelligence"])

    if has_bachelor:
        bachelor_terms = ["bachelor", "b.tech", "bachelor of technology", "bachelor of engineering"]
        if field:
            bachelor_terms.extend([f"b.tech {field}", f"bachelor {field}"])
        if "computer science" in field:
            bachelor_terms.extend(["b.tech computer science", "bachelor computer science"])
        _append_unique(terms, bachelor_terms)

    if has_master:
        master_terms = ["master", "m.tech", "master of technology", "master of engineering"]
        if field:
            master_terms.extend([f"m.tech {field}", f"master {field}"])
        if "artificial intelligence" in field:
            master_terms.extend(["m.tech artificial intelligence", "master artificial intelligence"])
        _append_unique(terms, master_terms)

    return terms


def education_ratio(job: JobDescriptionData, resume: ResumeData, matcher: SemanticMatcher) -> float:
    criteria: list[float] = []

    if job.required_education:
        resume_education_terms = []
        for entry in resume.education_entries:
            resume_education_terms.extend(_education_entry_terms(entry))
        score = semantic_coverage(list(normalize_set(job.required_education)), resume_education_terms, matcher)
        criteria.append(score)

    if job.required_certifications:
        resume_certs = list(normalize_set(resume.certifications))
        required_certs = list(normalize_set(job.required_certifications))
        score = semantic_coverage(required_certs, resume_certs, matcher)
        criteria.append(score)

    if not criteria:
        return 1.0
    return sum(criteria) / len(criteria)


def score_budget(job: JobDescriptionData) -> dict[str, int]:
    archetype = job.archetype.lower()
    if "research" in archetype:
        return {"skills": 32, "experience": 28, "education": 25, "keyword": 10, "completeness": 5}
    if "management" in archetype:
        return {"skills": 20, "experience": 35, "education": 15, "keyword": 15, "completeness": 15}
    if "data_ml" in archetype:
        return {"skills": 35, "experience": 25, "education": 22, "keyword": 13, "completeness": 5}
    if "product" in archetype:
        return {"skills": 20, "experience": 30, "education": 15, "keyword": 20, "completeness": 15}
    if "senior" in archetype:
        return {"skills": 35, "experience": 35, "education": 10, "keyword": 10, "completeness": 10}
    if "analyst" in archetype:
        return {"skills": 30, "experience": 25, "education": 15, "keyword": 20, "completeness": 10}
    if "finance" in archetype:
        return {"skills": 25, "experience": 30, "education": 25, "keyword": 15, "completeness": 5}
    return {"skills": 40, "experience": 30, "education": 15, "keyword": 10, "completeness": 5}


def match_required_skill_clusters(
    required_clusters: list[dict[str, Any]],
    candidate_terms: list[str],
    matcher: SemanticMatcher,
) -> dict[str, Any]:
    matched: list[str] = []
    missing: list[str] = []
    details: list[dict[str, Any]] = []
    matched_candidates: set[str] = set()

    for cluster in required_clusters:
        cluster_label = cluster.get("canonical", "")
        cluster_members = cluster.get("members") or [cluster_label]

        best_jd_member = cluster_label
        best_candidate = ""
        best_similarity = 0.0

        for jd_member in cluster_members:
            candidate, similarity = matcher.best_match(jd_member, candidate_terms)
            if similarity > best_similarity:
                best_similarity = similarity
                best_candidate = candidate
                best_jd_member = jd_member

        is_match = bool(best_candidate) and best_similarity >= matcher.required_match_threshold
        if is_match:
            matched.append(cluster_label)
            matched_candidates.add(best_candidate)
        else:
            missing.append(cluster_label)

        details.append(
            {
                "jd_skill": cluster_label,
                "matched_jd_member": best_jd_member,
                "matched_skill": best_candidate,
                "similarity": round(best_similarity, 3),
                "is_match": is_match,
                "threshold": round(matcher.required_match_threshold, 3),
            }
        )

    return {
        "matched": matched,
        "missing": missing,
        "details": details,
        "matched_candidates": sorted(matched_candidates),
    }


def internship_or_trainee_dominant(resume: ResumeData) -> bool:
    entries = [entry for entry in resume.experience_entries if normalize_skill(entry.title)]
    if not entries:
        return False
    internship_like = 0
    for entry in entries:
        title = normalize_skill(entry.title)
        if re.search(r"\b(intern|internship|trainee|apprentice)\b", title):
            internship_like += 1
    return (internship_like / len(entries)) >= 0.5


def score_candidate(job: JobDescriptionData, resume: ResumeData) -> CandidateScore:
    matcher = get_semantic_matcher()
    budgets = score_budget(job)

    # Scoring-level sanitation guards against noisy parser/LLM JD output.
    sanitized_required = sanitize_jd_skill_terms(job.must_have_skills)
    sanitized_preferred = sanitize_jd_skill_terms(job.good_to_have_skills)

    required_clusters = matcher.cluster_skills(sorted(normalize_set(sanitized_required)))
    preferred_clusters = matcher.cluster_skills(sorted(normalize_set(sanitized_preferred)))
    required_skills = [cluster["canonical"] for cluster in required_clusters]
    preferred_skills = [cluster["canonical"] for cluster in preferred_clusters]

    raw_skill_evidence = candidate_skill_evidence(resume)
    clustered_skill_evidence, candidate_clusters, candidate_term_evidence = cluster_skill_evidence(raw_skill_evidence, matcher)
    candidate_skills = sorted(clustered_skill_evidence.keys())
    candidate_skill_terms = sorted(candidate_term_evidence.keys())

    required_match = match_required_skill_clusters(
        required_clusters,
        candidate_skill_terms,
        matcher,
    )
    preferred_match = matcher.match_skills(
        preferred_skills,
        candidate_skill_terms,
        threshold=matcher.semantic_threshold,
    )

    matched_required = required_match["matched"]
    missing_required = required_match["missing"]
    required_required_count = len(required_skills)
    matched_required_count = len(matched_required)
    required_match_ratio = (
        matched_required_count / required_required_count
        if required_required_count
        else 1.0
    )
    critical_missing = [
        detail["jd_skill"] for detail in required_match["details"] if detail["similarity"] < matcher.partial_threshold
    ]

    required_credit = 0.0
    preferred_credit = 0.0
    required_details: list[dict[str, Any]] = []
    preferred_details: list[dict[str, Any]] = []

    for detail in required_match["details"]:
        matched_skill = detail["matched_skill"]
        similarity = float(detail["similarity"])
        evidence_weight = candidate_term_evidence.get(matched_skill, 0.0)
        # Required skills are hard signals: no partial credit unless the
        # required-threshold match succeeded.
        credit = (matcher.similarity_credit(similarity) * evidence_weight) if detail["is_match"] else 0.0
        required_credit += credit
        required_details.append(
            {
                **detail,
                "evidence_weight": round(evidence_weight, 2),
                "credit": round(credit, 3),
            }
        )

    for detail in preferred_match["details"]:
        matched_skill = detail["matched_skill"]
        similarity = float(detail["similarity"])
        evidence_weight = candidate_term_evidence.get(matched_skill, 0.0)
        credit = matcher.similarity_credit(similarity) * evidence_weight
        preferred_credit += credit
        preferred_details.append(
            {
                **detail,
                "evidence_weight": round(evidence_weight, 2),
                "credit": round(credit, 3),
            }
        )

    required_coverage = math.sqrt(required_credit / len(required_skills)) if required_skills else 1.0
    preferred_coverage = math.sqrt(preferred_credit / len(preferred_skills)) if preferred_skills else 1.0

    # Breadth Scaling: A single match shouldn't give 100% credit for the entire domain
    breadth_min = SCORING_CONFIG["breadth_min_skills"]
    breadth_floor = SCORING_CONFIG["breadth_floor"]
    breadth_factor = clamp(len(required_skills) / breadth_min, breadth_floor, 1.0) if required_skills else 1.0
    required_coverage *= breadth_factor

    if required_skills and preferred_skills:
        skill_ratio = clamp((3 * required_coverage + preferred_coverage) / 4, 0.0, 1.0)
    elif required_skills:
        skill_ratio = clamp(required_coverage, 0.0, 1.0)
    elif preferred_skills:
        skill_ratio = clamp(preferred_coverage, 0.0, 1.0)
    else:
        skill_ratio = 1.0

    skills_score = round(budgets["skills"] * skill_ratio)
    # Harden against random profiles: if required-skill coverage is weak,
    # clamp the skills contribution so preferred/semantic noise cannot dominate.
    if required_skills and required_match_ratio == 0.0:
        skills_score = min(skills_score, round(budgets["skills"] * 0.15))
    elif required_skills and required_match_ratio < 0.34:
        skills_score = min(skills_score, round(budgets["skills"] * 0.30))

    keyword_reference = sorted(keyword_set(job))
    jd_context_terms = sorted(
        normalize_set(
            sanitized_required
            + sanitized_preferred
            + job.domain_keywords
            + keyword_reference
            + sanitize_jd_skill_terms(extract_phrases(job.title, limit=8), limit=8)
            + extract_phrases("\n".join(job.responsibilities), limit=20)
        )
    )
    excluded_candidates = set(required_match["matched_candidates"])
    excluded_candidates.update(
        detail["matched_skill"]
        for detail in preferred_match["details"]
        if detail["matched_skill"] and detail["similarity"] >= matcher.required_match_threshold
    )
    # To keep the UI concise, we only pull "Additional Relevant Skills" from explicitly 
    # listed skills/tools, avoiding long certification or project names bleeding into this section.
    explicit_candidate_skills = sorted(normalize_set(resume.skills + resume.tools))
    additional_relevant_skills, additional_relevant_details = matcher.find_additional_relevant_skills(
        jd_context_terms,
        explicit_candidate_skills,
        exclude=excluded_candidates,
        threshold=matcher.additional_relevance_threshold,
    )
    additional_relevant_skills, additional_relevant_details = filter_additional_relevant_skill_matches(
        additional_relevant_skills,
        additional_relevant_details,
    )
    additional_skills_bonus_raw = len(additional_relevant_skills) * 2

    if job.min_years_experience > 0:
        years_fit = clamp(resume.total_years_experience / job.min_years_experience, 0.0, 1.0)
    else:
        # "No minimum experience required" should not grant full experience credit.
        # For open-entry roles, scale years contribution against a soft baseline.
        years_fit = clamp(resume.total_years_experience / 2.0, 0.0, 1.0)

    target_seniority = infer_seniority_level(job.title + " " + " ".join(job.responsibilities))
    candidate_peak_seniority = max((infer_seniority_level(entry.title) for entry in resume.experience_entries), default=0)
    if target_seniority > 0:
        seniority_fit = clamp(candidate_peak_seniority / target_seniority, 0.0, 1.0)
    elif candidate_peak_seniority > 0:
        seniority_fit = clamp(candidate_peak_seniority / 4, 0.0, 1.0)
    else:
        seniority_fit = 0.0

    growth_fit = progression_score(resume)
    domain_reference = list(normalize_set(job.domain_keywords + sanitized_required + sanitized_preferred))
    domain_reference.extend(keyword_reference)
    resume_domain_terms = experience_domain_terms(resume)
    domain_alignment = semantic_coverage(domain_reference, resume_domain_terms, matcher)
    detected_domain_tags, domain_detection_details = matcher.detect_domain_tags(
        domain_reference,
        resume_domain_terms,
        threshold=matcher.additional_relevance_threshold,
    )
    domain_bonus_raw = len(detected_domain_tags)
    additional_skills_cap, domain_bonus_cap = bonus_caps_for_required_match_ratio(
        required_match_ratio,
        base_additional_cap=10,
        base_domain_cap=matcher.domain_bonus_max,
    )
    additional_skills_bonus_score = min(additional_skills_bonus_raw, additional_skills_cap)
    domain_bonus_score = min(domain_bonus_raw, domain_bonus_cap)

    if resume.total_years_experience == 0:
        experience_score = 0
    else:
        experience_ratio = clamp(0.40 * years_fit + 0.25 * seniority_fit + 0.15 * growth_fit + 0.20 * domain_alignment, 0.0, 1.0)
        if resume.total_years_experience < 1.0:
            experience_ratio = min(experience_ratio, 0.35)
        experience_score = round(budgets["experience"] * experience_ratio)
        if internship_or_trainee_dominant(resume) and required_match_ratio < 0.5:
            experience_score = min(experience_score, round(budgets["experience"] * 0.25))

    education_fit = education_ratio(job, resume, matcher)
    education_score = round(budgets["education"] * clamp(education_fit, 0.0, 1.0))

    keyword_candidates = extract_phrases(f"{resume.summary}\n" + "\n".join(resume.projects), limit=20)
    keyword_ratio = semantic_coverage(keyword_reference, keyword_candidates, matcher)
    keyword_score = round(budgets["keyword"] * math.sqrt(clamp(keyword_ratio, 0.0, 1.0)))

    completeness_fields = [
        bool(resume.name),
        bool(resume.email),
        bool(resume.summary),
        bool(resume.skills),
        bool(resume.experience_entries),
        bool(resume.education_entries),
        bool(resume.linkedin or resume.portfolio),
    ]
    completeness_ratio = sum(completeness_fields) / len(completeness_fields)
    completeness_score = round(budgets["completeness"] * completeness_ratio)

    # Compute base score from core dimensions (without bonuses)
    base_score = (
        skills_score
        + experience_score
        + education_score
        + keyword_score
        + completeness_score
    )

    # Apply penalty caps to base score BEFORE adding bonuses
    # This prevents bonus points from masking fundamental skill gaps
    if required_skills and not matched_required:
        base_score = min(base_score, 40)
    elif critical_missing:
        base_score = min(base_score, 65)

    total_score = min(base_score + additional_skills_bonus_score + domain_bonus_score, 100)

    confidence_score = round(
        clamp(
            (
                0.30 * skill_ratio
                + 0.25 * completeness_ratio
                + 0.15 * min(len(candidate_skills) / 12, 1.0)
                + 0.15 * (
                    sum(detail["evidence_weight"] for detail in required_details if detail["is_match"])
                    / max(len([detail for detail in required_details if detail["is_match"]]), 1)
                    if required_details
                    else 1.0
                )
                + 0.10 * min(len(additional_relevant_skills) / 5, 1.0)
                + 0.05 * domain_alignment
            )
            * 100,
            0,
            100,
        )
    )

    years_gap = max(job.min_years_experience - resume.total_years_experience, 0.0)
    risk_score = 0.0
    risk_score += min(len(critical_missing) * 20, 60)
    
    if required_skills and not matched_required:
        risk_score += 40

    risk_score += min(years_gap * 8, 25)
    
    if completeness_ratio < 0.5:
        risk_score += 15
    if domain_alignment < 0.5:
        risk_score += 10
        
    risk_score = round(clamp(risk_score, 0, 100))

    strengths: list[str] = []
    risks: list[str] = []

    if matched_required:
        strengths.append(f"Matched required skills: {', '.join(matched_required[:5])}")
    if additional_relevant_skills:
        strengths.append(f"Additional relevant skills: {', '.join(additional_relevant_skills[:5])}")
    if preferred_skills and preferred_coverage >= 0.5:
        strengths.append("Good alignment on preferred skills and tooling")
    if years_fit >= 1.0 and job.min_years_experience > 0:
        strengths.append(f"Meets experience threshold with {resume.total_years_experience:.1f} years")
    if domain_alignment >= 0.7:
        strengths.append("Experience shows strong alignment with the role domain")

    if critical_missing:
        risks.append(f"Critical skill gaps: {', '.join(critical_missing[:5])}")
    elif missing_required:
        risks.append(f"Missing required skills: {', '.join(missing_required[:5])}")
    if years_gap > 0:
        risks.append(
            f"Experience gap: requires {job.min_years_experience:.1f} years, profile shows {resume.total_years_experience:.1f}"
        )
    if completeness_ratio < 0.5:
        risks.append("Resume is missing key profile details")
    if domain_alignment < 0.4:
        risks.append("Domain evidence in experience appears limited")

    # Impact signal analysis (reporting only — does not affect numerical score)
    all_highlights = [h for e in resume.experience_entries for h in e.highlights]
    if all_highlights:
        impact = extract_impact_signals(all_highlights)
        ownership = ownership_ratio(all_highlights)
        if impact["quantification_rate"] >= 0.3:
            strengths.append(f"Strong quantified impact: {impact['quantified_bullets']}/{impact['total_bullets']} bullets have measurable outcomes")
        elif impact["total_bullets"] > 3 and impact["quantification_rate"] < 0.1:
            risks.append("Experience highlights lack quantified outcomes — probe for measurable impact in interview")
        if ownership >= 0.6:
            strengths.append("High ownership language — candidate demonstrates leadership in execution")
        elif ownership <= 0.2:
            risks.append("Contributor-heavy language — may not have driven outcomes independently")

    matched_required_evidence_weights = [
        detail["evidence_weight"] for detail in required_details if detail["is_match"]
    ]
    required_proof_ratio = (
        sum(matched_required_evidence_weights) / len(matched_required_evidence_weights)
        if matched_required_evidence_weights
        else (1.0 if not required_details else 0.0)
    )

    semantic_details = {
        "required_match_ratio": round(required_match_ratio, 3),
        "required_required_count": required_required_count,
        "matched_required_count": matched_required_count,
        "required": required_details,
        "preferred": preferred_details,
        "required_clusters": required_clusters,
        "preferred_clusters": preferred_clusters,
        "candidate_clusters": candidate_clusters,
        "additional_relevant": additional_relevant_details,
        "domain_detection": domain_detection_details,
        "thresholds": {
            "required_match_threshold": matcher.required_match_threshold,
            "partial_threshold": matcher.partial_threshold,
            "semantic_threshold": matcher.semantic_threshold,
            "additional_relevance_threshold": matcher.additional_relevance_threshold,
            "clustering_threshold": matcher.clustering_threshold,
        },
        "bonus_scores": {
            "additional_skills_bonus_score": additional_skills_bonus_score,
            "domain_bonus_score": domain_bonus_score,
            "additional_skills_bonus_raw": additional_skills_bonus_raw,
            "domain_bonus_raw": domain_bonus_raw,
        },
        "applied_bonus_caps": {
            "additional_skills_cap": additional_skills_cap,
            "domain_bonus_cap": domain_bonus_cap,
        },
        "evidence_weighted_confidence": {
            "required_proof_ratio": round(required_proof_ratio, 3),
            "score": round(required_proof_ratio * 100),
            "meaning": "Matched required skills backed by experience/project evidence score higher than list-only skills.",
        },
    }

    return CandidateScore(
        total_score=total_score,
        skills_score=skills_score,
        experience_score=experience_score,
        education_score=education_score,
        keyword_score=keyword_score,
        completeness_score=completeness_score,
        budgets=budgets,
        confidence_score=confidence_score,
        risk_score=risk_score,
        matched_skills=matched_required,
        missing_skills=missing_required,
        critical_missing_skills=critical_missing,
        additional_relevant_skills=additional_relevant_skills,
        additional_skills_bonus_score=additional_skills_bonus_score,
        detected_domain_tags=detected_domain_tags,
        semantic_match_details=semantic_details,
        strengths=strengths,
        risks=risks,
    )


def build_recruiter_feedback(
    job: JobDescriptionData,
    resume: ResumeData,
    score: CandidateScore,
    reasoning_summary: str = "",
) -> str:
    if reasoning_summary.strip():
        sanitized_summary = reasoning_summary.strip()
        if not score.critical_missing_skills:
            sanitized_summary = re.sub(
                r"\bcritical missing skills?\b",
                "missing required skills",
                sanitized_summary,
                flags=re.IGNORECASE,
            )
            sanitized_summary = re.sub(
                r"\bcritical skill gaps?\b",
                "skill gaps",
                sanitized_summary,
                flags=re.IGNORECASE,
            )
        return sanitized_summary

    headline = f"{resume.name or 'Candidate'} was scored for {job.title or 'the role'} with a total score of {score.total_score}/100."
    notes = score.strengths[:2] + score.risks[:2]
    if not notes:
        notes = ["Profile was processed successfully but surfaced limited structured evidence."]
    return f"{headline} " + " ".join(notes)
