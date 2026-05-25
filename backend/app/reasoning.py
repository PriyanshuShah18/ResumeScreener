from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from app.schemas import CandidateScore, JobDescriptionData, ResumeData
from app.semantic_matching import get_semantic_matcher, lexical_similarity


class ReasoningOutput(BaseModel):
    fit_rationale: str = Field(default="")
    interview_focus_areas: list[str] = Field(default_factory=list)
    hidden_strengths: list[str] = Field(default_factory=list)


class EvidenceSnippet(BaseModel):
    source: str = Field(default="")
    text: str = Field(default="")
    matched_query: str = Field(default="")
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)


class ReasoningContext(BaseModel):
    fit_band: str = Field(default="")
    query_terms: list[str] = Field(default_factory=list)
    retrieved_evidence: list[EvidenceSnippet] = Field(default_factory=list)
    gap_evidence: list[str] = Field(default_factory=list)
    impact_evidence: list[str] = Field(default_factory=list)


_TOKEN_RE = re.compile(r"[a-z0-9\+\#\./]+")
_REASONING_STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "into",
    "the",
    "this",
    "that",
    "with",
    "will",
    "role",
    "team",
    "work",
    "build",
    "candidate",
    "engineer",
}
_IMPACT_RE = re.compile(
    r"(?i)\b(?:built|designed|led|owned|launched|improved|reduced|optimized|scaled|shipped)\b|"
    r"\b\d+(?:\.\d+)?\s*(?:%|k|m|users|requests|transactions|ms|seconds|monthly|daily)?\b"
)


def _compact_text(value: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip(" ,.;:") + "..."


def _append_unique(target: list[str], values: list[str]) -> None:
    seen = {item.lower() for item in target}
    for value in values:
        cleaned = _compact_text(value, limit=160)
        if not cleaned or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        target.append(cleaned)


def _sentence_join(values: list[str], limit: int = 2) -> str:
    cleaned: list[str] = []
    _append_unique(cleaned, values[:limit])
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return "; ".join(cleaned[:-1]) + "; and " + cleaned[-1]


def _token_set(value: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall((value or "").lower())
        if token not in _REASONING_STOPWORDS
    }


def _job_query_terms(job: JobDescriptionData, score: CandidateScore) -> list[str]:
    terms: list[str] = []
    _append_unique(terms, [job.title])
    _append_unique(terms, job.must_have_skills)
    _append_unique(terms, job.good_to_have_skills)
    _append_unique(terms, job.domain_keywords)
    _append_unique(terms, job.responsibilities[:6])
    _append_unique(terms, score.matched_skills)
    _append_unique(terms, score.additional_relevant_skills)
    return terms[:24]


def _resume_snippets(resume: ResumeData) -> list[tuple[str, str]]:
    snippets: list[tuple[str, str]] = []
    if resume.summary:
        snippets.append(("summary", f"Summary: {resume.summary}"))
    if resume.skills:
        snippets.append(("skills", "Skills: " + ", ".join(resume.skills[:25])))
    if resume.tools:
        snippets.append(("tools", "Tools: " + ", ".join(resume.tools[:20])))
    if resume.certifications:
        snippets.append(("certifications", "Certifications: " + ", ".join(resume.certifications[:10])))

    for entry in resume.experience_entries[:6]:
        role_bits = [entry.title, entry.company]
        role_text = " at ".join(bit for bit in role_bits if bit)
        if role_text:
            snippets.append(("experience_role", f"Experience role: {role_text}"))
        if entry.skills_used:
            snippets.append(("experience_skills", f"Experience skills: {', '.join(entry.skills_used[:15])}"))
        for highlight in entry.highlights[:4]:
            snippets.append(("experience_highlight", f"Experience evidence: {highlight}"))

    for project in resume.projects[:6]:
        snippets.append(("project", f"Project evidence: {project}"))

    for education in resume.education_entries[:4]:
        parts = [education.degree, education.field_of_study, education.institution]
        text = " ".join(part for part in parts if part)
        if text:
            snippets.append(("education", f"Education: {text}"))

    return [(source, _compact_text(text)) for source, text in snippets if _compact_text(text)]


def _lexical_overlap(query: str, text: str) -> float:
    query_tokens = _token_set(query)
    if not query_tokens:
        return 0.0
    text_tokens = _token_set(text)
    if not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def _rank_snippet(query_terms: list[str], text: str) -> tuple[str, float]:
    if not query_terms or not text:
        return "", 0.0

    matcher = get_semantic_matcher()
    best_query = ""
    best_score = 0.0
    for query in query_terms:
        overlap = _lexical_overlap(query, text)
        lexical_score = lexical_similarity(query, text)
        semantic_score = 0.0
        if matcher.enable_model and getattr(matcher, "_model", None) is not None:
            try:
                semantic_score = matcher.similarity(query, text)
            except Exception:
                semantic_score = 0.0
        score = max(overlap, lexical_score, semantic_score)
        if score > best_score:
            best_query = query
            best_score = score
    return best_query, min(round(best_score, 3), 1.0)


def _fit_band(score: CandidateScore) -> str:
    if score.total_score >= 80 and score.risk_score <= 45:
        return "strong"
    if score.total_score >= 60:
        return "moderate"
    if score.total_score >= 40:
        return "partial"
    return "limited"


def build_reasoning_context(
    job: JobDescriptionData,
    resume: ResumeData,
    score: CandidateScore,
    max_snippets: int = 6,
) -> ReasoningContext:
    """Retrieve resume evidence most relevant to the job and score signals."""
    query_terms = _job_query_terms(job, score)
    ranked: list[EvidenceSnippet] = []
    impact_evidence: list[str] = []

    for source, text in _resume_snippets(resume):
        matched_query, relevance = _rank_snippet(query_terms, text)
        if relevance >= 0.25:
            ranked.append(
                EvidenceSnippet(
                    source=source,
                    text=text,
                    matched_query=matched_query,
                    relevance=relevance,
                )
            )
        if source in {"experience_highlight", "project"} and _IMPACT_RE.search(text):
            impact_evidence.append(text)

    ranked.sort(key=lambda item: (-item.relevance, item.source, item.text))

    gap_evidence: list[str] = []
    if score.critical_missing_skills:
        gap_evidence.append("Critical missing skills: " + ", ".join(score.critical_missing_skills[:6]))
    elif score.missing_skills:
        gap_evidence.append("Missing required skills: " + ", ".join(score.missing_skills[:6]))
    for risk in score.risks[:3]:
        _append_unique(gap_evidence, [risk])

    return ReasoningContext(
        fit_band=_fit_band(score),
        query_terms=query_terms,
        retrieved_evidence=ranked[:max_snippets],
        gap_evidence=gap_evidence[:5],
        impact_evidence=impact_evidence[:4],
    )


def fallback_reasoning(job: JobDescriptionData, resume: ResumeData, score: CandidateScore) -> ReasoningOutput:
    context = build_reasoning_context(job, resume, score)
    top_evidence = [item.text for item in context.retrieved_evidence[:3]]
    primary_evidence = _sentence_join(top_evidence, limit=2)
    matched_requirements = ", ".join(score.matched_skills[:6])
    adjacent_skills = ", ".join(score.additional_relevant_skills[:5])
    gap_summary = _sentence_join(context.gap_evidence, limit=2)

    focus_areas: list[str] = []
    if score.critical_missing_skills:
        focus_areas.append(f"Validate missing critical skills: {', '.join(score.critical_missing_skills[:4])}")
    if score.missing_skills:
        focus_areas.append(f"Assess must-have skill depth: {', '.join(score.missing_skills[:4])}")
    if context.retrieved_evidence:
        focus_areas.append(
            "Probe scale, recency, and ownership behind: "
            + context.retrieved_evidence[0].text
        )
    for risk in score.risks[:2]:
        focus_areas.append(risk)

    hidden_strengths: list[str] = []
    if score.additional_relevant_skills:
        hidden_strengths.append(
            "Adjacent relevant skills: " + ", ".join(score.additional_relevant_skills[:5])
        )
    project_evidence = [item.text for item in context.retrieved_evidence if item.source == "project"]
    _append_unique(hidden_strengths, project_evidence[:2])
    _append_unique(hidden_strengths, context.impact_evidence[:2])
    _append_unique(hidden_strengths, score.strengths[:3])
    if not hidden_strengths and resume.projects:
        hidden_strengths.append("Project work indicates practical delivery experience.")

    rationale_parts = [
        (
            f"{resume.name or 'Candidate'} appears to be a {context.fit_band} fit for "
            f"{job.title or 'the role'} with a total score of {score.total_score}/100"
        ),
    ]
    if primary_evidence:
        rationale_parts.append(f"because the strongest evidence shows {primary_evidence}.")
    elif score.strengths:
        rationale_parts.append(f"because {score.strengths[0]}.")
    else:
        rationale_parts.append("based on the available structured resume evidence.")

    if matched_requirements:
        rationale_parts.append(f"The profile directly supports {matched_requirements}.")
    if adjacent_skills:
        rationale_parts.append(f"Adjacent evidence in {adjacent_skills} strengthens the match.")
    if gap_summary:
        rationale_parts.append(f"However, the interview should verify {gap_summary}.")
    elif score.risks:
        rationale_parts.append(f"However, the interview should verify {_sentence_join(score.risks, limit=2)}.")

    return ReasoningOutput(
        fit_rationale=" ".join(part for part in rationale_parts if part).strip(),
        interview_focus_areas=focus_areas[:4],
        hidden_strengths=hidden_strengths[:4],
    )


def build_reasoning_prompt(job: JobDescriptionData, resume: ResumeData, score: CandidateScore) -> str:
    schema_json = ReasoningOutput.model_json_schema()
    reasoning_context = build_reasoning_context(job, resume, score)
    return (
        "You are an expert technical HR reasoning assistant. Return valid JSON only that matches this schema: "
        f"{json.dumps(schema_json)}. "
        "Use the RetrievedEvidence, score, and structured data to give a concise rationale, interview focus areas, and hidden strengths. "
        "Do not make or imply any selection outcome. The HR team will decide separately.\n\n"
        "**CRITICAL REASONING & CONTEXT RULES:**\n"
        "0. **Grounding:** Treat RetrievedEvidence as the local RAG context. Prefer concrete evidence snippets over generic claims. "
        "If evidence is weak, say what should be validated rather than overstating fit.\n"
        "1. **Evaluate Modern Equivalents:** If a required skill is missing, actively check 'additional_relevant_skills'. "
        "If the candidate possesses highly synergistic or modern equivalents (e.g., Langchain/Generative AI for an ML role, "
        "or Next.js for a React role), HIGHLIGHT this as a strong compensatory factor in the rationale.\n"
        "2. **Strict Anti-Hallucination:** DO NOT invent skills. DO NOT claim unrelated skills are 'essential'. "
        "For example, if testing a React Native role, possessing Python/Java is NOT relevant frontend experience. "
        "Do not falsely compensate with functionally unrelated tech.\n"
        "3. **Objective Reality:** Only base your explanation on the EXACT data provided in CandidateScore and ResumeData.\n"
        "4. **Evaluate Impact Evidence vs. Skill Claims:** A candidate listing 'Python' as a skill "
        "is worth less than a candidate whose highlights show them using Python at scale with measurable outcomes. "
        "When populating strengths and hidden_strengths, prioritize evidence-based strengths (demonstrated at scale) "
        "over skill-based strengths (mentioned in skills list). When populating interview_focus_areas, include at "
        "least one probe for the *scale* of claimed experience.\n\n"
        f"RetrievedEvidence:\n{json.dumps(reasoning_context.model_dump(), ensure_ascii=True)}\n\n"
        f"JobDescriptionData:\n{json.dumps(job.model_dump(), ensure_ascii=True)}\n\n"
        f"ResumeData:\n{json.dumps(resume.model_dump(), ensure_ascii=True)}\n\n"
        f"CandidateScore:\n{json.dumps(score.model_dump(), ensure_ascii=True)}"
    )


from app.llm_understanding import llm_service

def generate_reasoning(
    job: JobDescriptionData,
    resume: ResumeData,
    score: CandidateScore,
    model_client: Any | None = None,
) -> ReasoningOutput:
    """Generate hiring reasoning from scored data.
    """
    prompt = build_reasoning_prompt(job, resume, score)

    if model_client is not None and getattr(model_client, "model_available", False):
        if hasattr(model_client, "generate_response") and hasattr(model_client, "parse_json"):
            try:
                raw_response = model_client.generate_response([{"role": "user", "content": prompt}])
                result = model_client.parse_json(raw_response)
            except Exception:
                result = None
        else:
            result = None
    elif llm_service.enabled:
        result, _ = llm_service._generate_json(prompt, {})
    else:
        return fallback_reasoning(job, resume, score)

    if not result:
        return fallback_reasoning(job, resume, score)

    try:
        return ReasoningOutput.model_validate(result)
    except Exception:
        # LLM returned malformed data — use what we can, fall back for the rest
        fb = fallback_reasoning(job, resume, score)
        return ReasoningOutput(
            fit_rationale=result.get("fit_rationale", fb.fit_rationale),
            interview_focus_areas=result.get("interview_focus_areas", fb.interview_focus_areas) if isinstance(result.get("interview_focus_areas"), list) else fb.interview_focus_areas,
            hidden_strengths=result.get("hidden_strengths", fb.hidden_strengths) if isinstance(result.get("hidden_strengths"), list) else fb.hidden_strengths,
        )
