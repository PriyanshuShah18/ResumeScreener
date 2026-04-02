from __future__ import annotations

import math
import re
from typing import Any

from app.constants import (
    SCORING_SKILL_NOISE as SKILL_NOISE,
    SENIORITY_MAP,
    SENIOR_ROLE_TOKENS,
    RESEARCH_ROLE_TOKENS,
    STOPWORDS,
)
from app.schemas import (
    CandidateScore,
    JobDescriptionData,
    ResumeData,
    month_index,
    normalize_skill,
)
from app.semantic_matching import SemanticMatcher, canonicalize_skill, get_semantic_matcher


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def normalize_set(values: list[str]) -> set[str]:
    normalized = {canonicalize_skill(value) for value in values if canonicalize_skill(value)}
    return {value for value in normalized if value and value not in SKILL_NOISE}


def extract_phrases(value: str, limit: int = 15) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for chunk in re.split(r"[\n,;/|]+", value):
        cleaned = canonicalize_skill(chunk)
        if not cleaned or cleaned in seen:
            continue
        token_count = len(cleaned.split())
        if token_count == 0 or token_count > 4:
            continue
        if cleaned in SKILL_NOISE:
            continue
        seen.add(cleaned)
        candidates.append(cleaned)
        if len(candidates) >= limit:
            break
    return candidates


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

    add_terms(resume.skills, 0.4)
    add_terms(resume.tools, 0.4)
    add_terms(resume.certifications, 0.4)

    ordered_entries = sorted_experience_entries(resume)
    for idx, entry in enumerate(reversed(ordered_entries)):
        decay_factor = 0.85 ** idx
        add_terms(entry.skills_used, 0.75 * decay_factor)
        add_terms(extract_phrases("\n".join(entry.highlights), limit=12), 0.60 * decay_factor)

    for idx, project in enumerate(resume.projects):
        decay_factor = 0.85 ** idx
        add_terms(extract_phrases(project, limit=8), 0.50 * decay_factor)

    return evidence


def cluster_skill_evidence(
    skill_evidence: dict[str, float],
    matcher: SemanticMatcher,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    clusters = matcher.cluster_skills(list(skill_evidence.keys()))
    clustered_evidence: dict[str, float] = {}
    cluster_details: list[dict[str, Any]] = []

    for cluster in clusters:
        canonical = cluster["canonical"]
        members = cluster["members"]
        max_weight = max((skill_evidence.get(member, 0.0) for member in members), default=0.0)
        clustered_evidence[canonical] = max_weight
        cluster_details.append(
            {
                "canonical": canonical,
                "members": members,
                "max_evidence_weight": round(max_weight, 2),
            }
        )

    return clustered_evidence, cluster_details


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
        return 0.6

    non_decreasing = sum(1 for i in range(1, len(levels)) if levels[i] >= levels[i - 1]) / (len(levels) - 1)
    upward_gain = max(levels[-1] - levels[0], 0) / 5.0
    return clamp(0.65 * non_decreasing + 0.35 * upward_gain, 0.0, 1.0)


def experience_domain_terms(resume: ResumeData) -> list[str]:
    terms = set(candidate_skill_evidence(resume).keys())
    for entry in resume.experience_entries:
        terms.update(extract_phrases(entry.title, limit=5))
        terms.update(extract_phrases("\n".join(entry.highlights), limit=10))
    terms.update(extract_phrases(resume.summary, limit=10))
    for project in resume.projects:
        terms.update(extract_phrases(project, limit=8))
    return sorted(terms)


def education_ratio(job: JobDescriptionData, resume: ResumeData, matcher: SemanticMatcher) -> float:
    criteria: list[float] = []

    if job.required_education:
        resume_education_terms = []
        for entry in resume.education_entries:
            resume_education_terms.extend(
                extract_phrases(f"{entry.degree} {entry.field_of_study} {entry.institution}", limit=12)
            )
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
    tokens = set(re.findall(r"[a-z]+", f"{job.title} {' '.join(job.responsibilities)}".lower()))
    if tokens & RESEARCH_ROLE_TOKENS:
        return {"skills": 32, "experience": 28, "education": 25, "keyword": 10, "completeness": 5}
    if tokens & SENIOR_ROLE_TOKENS:
        return {"skills": 35, "experience": 35, "education": 15, "keyword": 10, "completeness": 5}
    return {"skills": 40, "experience": 30, "education": 15, "keyword": 10, "completeness": 5}


def recommendation(total_score: int, risk_score: int) -> str:
    if risk_score >= 60:
        return "low fit"
    if total_score >= 85 and risk_score < 35:
        return "strong fit"
    if total_score >= 70 and risk_score < 55:
        return "good fit"
    if total_score >= 50 and risk_score < 60:
        return "review"
    return "low fit"


def score_candidate(job: JobDescriptionData, resume: ResumeData) -> CandidateScore:
    matcher = get_semantic_matcher()
    budgets = score_budget(job)

    required_clusters = matcher.cluster_skills(sorted(normalize_set(job.must_have_skills)))
    preferred_clusters = matcher.cluster_skills(sorted(normalize_set(job.good_to_have_skills)))
    required_skills = [cluster["canonical"] for cluster in required_clusters]
    preferred_skills = [cluster["canonical"] for cluster in preferred_clusters]

    raw_skill_evidence = candidate_skill_evidence(resume)
    clustered_skill_evidence, candidate_clusters = cluster_skill_evidence(raw_skill_evidence, matcher)
    candidate_skills = sorted(clustered_skill_evidence.keys())

    required_match = matcher.match_skills(
        required_skills,
        candidate_skills,
        threshold=matcher.required_match_threshold,
    )
    preferred_match = matcher.match_skills(
        preferred_skills,
        candidate_skills,
        threshold=matcher.semantic_threshold,
    )

    matched_required = required_match["matched"]
    missing_required = required_match["missing"]
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
        evidence_weight = clustered_skill_evidence.get(matched_skill, 0.0)
        credit = matcher.similarity_credit(similarity) * evidence_weight
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
        evidence_weight = clustered_skill_evidence.get(matched_skill, 0.0)
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

    if required_skills and preferred_skills:
        skill_ratio = clamp((3 * required_coverage + preferred_coverage) / 4, 0.0, 1.0)
    elif required_skills:
        skill_ratio = clamp(required_coverage, 0.0, 1.0)
    elif preferred_skills:
        skill_ratio = clamp(preferred_coverage, 0.0, 1.0)
    else:
        skill_ratio = 1.0

    skills_score = round(budgets["skills"] * skill_ratio)

    keyword_reference = sorted(keyword_set(job))
    jd_context_terms = sorted(
        normalize_set(
            job.must_have_skills
            + job.good_to_have_skills
            + job.domain_keywords
            + keyword_reference
            + extract_phrases(job.title, limit=8)
            + extract_phrases("\n".join(job.responsibilities), limit=20)
        )
    )
    excluded_candidates = set(required_match["matched_candidates"])
    excluded_candidates.update(
        detail["matched_skill"]
        for detail in preferred_match["details"]
        if detail["matched_skill"] and detail["similarity"] >= matcher.required_match_threshold
    )
    additional_relevant_skills, additional_relevant_details = matcher.find_additional_relevant_skills(
        jd_context_terms,
        candidate_skills,
        exclude=excluded_candidates,
        threshold=matcher.additional_relevance_threshold,
    )
    additional_skills_bonus_score = min(len(additional_relevant_skills) * 2, 10)

    if job.min_years_experience > 0:
        years_fit = clamp(resume.total_years_experience / job.min_years_experience, 0.0, 1.0)
    else:
        years_fit = 1.0

    target_seniority = infer_seniority_level(job.title + " " + " ".join(job.responsibilities))
    candidate_peak_seniority = max((infer_seniority_level(entry.title) for entry in resume.experience_entries), default=0)
    if target_seniority > 0:
        seniority_fit = clamp(candidate_peak_seniority / target_seniority, 0.0, 1.0)
    elif candidate_peak_seniority > 0:
        seniority_fit = clamp(candidate_peak_seniority / 4, 0.0, 1.0)
    else:
        seniority_fit = 0.0

    growth_fit = progression_score(resume)
    domain_reference = list(normalize_set(job.domain_keywords + job.must_have_skills + job.good_to_have_skills))
    domain_reference.extend(keyword_reference)
    resume_domain_terms = experience_domain_terms(resume)
    domain_alignment = semantic_coverage(domain_reference, resume_domain_terms, matcher)
    detected_domain_tags, domain_detection_details = matcher.detect_domain_tags(
        domain_reference,
        resume_domain_terms,
        threshold=matcher.additional_relevance_threshold,
    )
    domain_bonus_score = min(len(detected_domain_tags), matcher.domain_bonus_max)

    if resume.total_years_experience == 0:
        experience_score = 0
    else:
        experience_ratio = clamp(0.40 * years_fit + 0.25 * seniority_fit + 0.15 * growth_fit + 0.20 * domain_alignment, 0.0, 1.0)
        experience_score = round(budgets["experience"] * experience_ratio)

    education_fit = education_ratio(job, resume, matcher)
    education_score = round(budgets["education"] * clamp(education_fit, 0.0, 1.0))

    keyword_candidates = extract_phrases(f"{resume.summary}\n" + "\n".join(resume.projects), limit=20)
    keyword_ratio = semantic_coverage(keyword_reference, keyword_candidates, matcher)
    keyword_score = round(budgets["keyword"] * math.sqrt(clamp(keyword_ratio, 0.0, 1.0)))

    completeness_fields = [
        bool(resume.name),
        bool(resume.email),
        bool(resume.phone),
        bool(resume.summary),
        bool(resume.skills),
        bool(resume.experience_entries),
        bool(resume.education_entries),
        bool(resume.linkedin or resume.portfolio),
    ]
    completeness_ratio = sum(completeness_fields) / len(completeness_fields)
    completeness_score = round(budgets["completeness"] * completeness_ratio)

    total_score = (
        skills_score
        + experience_score
        + education_score
        + keyword_score
        + completeness_score
        + additional_skills_bonus_score
        + domain_bonus_score
    )
    
    if required_skills and not matched_required:
        total_score = min(total_score, 45)
    elif critical_missing:
        total_score = min(total_score, 60)
        
    total_score = min(total_score, 100)

    confidence_score = round(
        clamp(
            (
                0.30 * skill_ratio
                + 0.30 * completeness_ratio
                + 0.20 * min(len(candidate_skills) / 20, 1.0)
                + 0.10 * min(len(additional_relevant_skills) / 5, 1.0)
                + 0.10 * domain_alignment
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

    semantic_details = {
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
        },
    }

    return CandidateScore(
        total_score=total_score,
        skills_score=skills_score,
        experience_score=experience_score,
        education_score=education_score,
        keyword_score=keyword_score,
        completeness_score=completeness_score,
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
        recommendation=recommendation(total_score, risk_score),
    )


def build_recruiter_feedback(
    job: JobDescriptionData,
    resume: ResumeData,
    score: CandidateScore,
    reasoning_summary: str = "",
) -> str:
    if reasoning_summary.strip():
        return reasoning_summary.strip()

    headline = f"{resume.name or 'Candidate'} is a {score.recommendation} for {job.title or 'the role'}."
    notes = score.strengths[:2] + score.risks[:2]
    if not notes:
        notes = ["Profile was processed successfully but surfaced limited structured evidence."]
    return f"{headline} " + " ".join(notes)
