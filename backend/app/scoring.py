from __future__ import annotations

import re

from app.schemas import CandidateScore, JobDescriptionData, ResumeData, normalize_skill

STOPWORDS = {
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


def normalize_set(values: list[str]) -> set[str]:
    return {normalize_skill(value) for value in values if normalize_skill(value)}


def candidate_skill_set(resume: ResumeData) -> set[str]:
    skills = set(resume.skills) | set(resume.tools) | set(resume.certifications)
    for entry in resume.experience_entries:
        skills.update(entry.skills_used)
    return normalize_set(list(skills))


def experience_text(resume: ResumeData) -> str:
    parts = [resume.summary]
    for entry in resume.experience_entries:
        parts.append(entry.title)
        parts.append(entry.company)
        parts.extend(entry.highlights)
    parts.extend(resume.projects)
    return " ".join(parts).lower()


def text_overlap_ratio(reference: set[str], candidate_text: str) -> float:
    if not reference:
        return 1.0
    matches = sum(1 for token in reference if token and token in candidate_text)
    return matches / len(reference)


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
    return normalize_set(list(title_tokens | set(job.domain_keywords) | responsibility_tokens))


def education_ratio(job: JobDescriptionData, resume: ResumeData) -> float:
    criteria: list[bool] = []

    if job.required_education:
        resume_education = " ".join(
            f"{entry.degree} {entry.field_of_study} {entry.institution}" for entry in resume.education_entries
        ).lower()
        criteria.append(any(requirement in resume_education for requirement in job.required_education))

    if job.required_certifications:
        resume_certs = normalize_set(resume.certifications)
        required_certs = normalize_set(job.required_certifications)
        criteria.append(bool(required_certs & resume_certs))

    if not criteria:
        return 1.0
    return sum(1 for passed in criteria if passed) / len(criteria)


def recommendation(total_score: int) -> str:
    if total_score >= 85:
        return "strong fit"
    if total_score >= 70:
        return "good fit"
    if total_score >= 50:
        return "review"
    return "low fit"


def score_candidate(job: JobDescriptionData, resume: ResumeData) -> CandidateScore:
    resume_skills = candidate_skill_set(resume)
    required_skills = normalize_set(job.must_have_skills)
    preferred_skills = normalize_set(job.good_to_have_skills)

    matched_required = sorted(required_skills & resume_skills)
    missing_required = sorted(required_skills - resume_skills)

    required_score = round(30 * (len(matched_required) / len(required_skills))) if required_skills else 30
    preferred_score = round(10 * (len(preferred_skills & resume_skills) / len(preferred_skills))) if preferred_skills else 10

    candidate_experience_text = experience_text(resume)
    relevance_reference = required_skills | preferred_skills | keyword_set(job)
    relevance_ratio = text_overlap_ratio(relevance_reference, candidate_experience_text)
    experience_score = round(20 * relevance_ratio) if resume.experience_entries else 0

    if job.min_years_experience > 0:
        years_ratio = min(resume.total_years_experience / job.min_years_experience, 1.0)
        years_score = round(10 * years_ratio)
    else:
        years_score = 10

    education_score = round(15 * education_ratio(job, resume))
    keyword_score = round(10 * text_overlap_ratio(keyword_set(job), f"{resume.summary} {' '.join(resume.projects)}".lower()))

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
    completeness_score = round(5 * (sum(completeness_fields) / len(completeness_fields)))

    total_score = min(
        required_score + preferred_score + experience_score + years_score + education_score + keyword_score + completeness_score,
        100,
    )

    strengths: list[str] = []
    risks: list[str] = []

    if matched_required:
        strengths.append(f"Matched required skills: {', '.join(matched_required[:5])}")
    if preferred_skills & resume_skills:
        strengths.append(f"Matched preferred skills: {', '.join(sorted(preferred_skills & resume_skills)[:5])}")
    if resume.total_years_experience >= job.min_years_experience > 0:
        strengths.append(f"Meets experience threshold with {resume.total_years_experience:.1f} years")
    if education_score >= 10:
        strengths.append("Education and certification signals align with the JD")

    if missing_required:
        risks.append(f"Missing required skills: {', '.join(missing_required[:5])}")
    if job.min_years_experience > resume.total_years_experience:
        risks.append(
            f"Experience gap: requires {job.min_years_experience:.1f} years, profile shows {resume.total_years_experience:.1f}"
        )
    if not resume.education_entries and job.required_education:
        risks.append("Required education was not clearly found")
    if completeness_score < 3:
        risks.append("Resume is missing key profile details")

    return CandidateScore(
        total_score=total_score,
        skills_score=required_score + preferred_score,
        experience_score=experience_score + years_score,
        education_score=education_score,
        keyword_score=keyword_score,
        completeness_score=completeness_score,
        matched_skills=matched_required,
        missing_skills=missing_required,
        strengths=strengths,
        risks=risks,
        recommendation=recommendation(total_score),
    )


def build_recruiter_feedback(job: JobDescriptionData, resume: ResumeData, score: CandidateScore) -> str:
    headline = f"{resume.name or 'Candidate'} is a {score.recommendation} for {job.title or 'the role'}."
    notes = score.strengths[:2] + score.risks[:2]
    if not notes:
        notes = ["Profile was processed successfully but surfaced limited structured evidence."]
    return f"{headline} " + " ".join(notes)
