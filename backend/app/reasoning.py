from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.schemas import CandidateScore, JobDescriptionData, ResumeData


class ReasoningOutput(BaseModel):
    fit_rationale: str = Field(default="")
    interview_focus_areas: list[str] = Field(default_factory=list)
    hidden_strengths: list[str] = Field(default_factory=list)


def fallback_reasoning(job: JobDescriptionData, resume: ResumeData, score: CandidateScore) -> ReasoningOutput:
    focus_areas: list[str] = []
    if score.critical_missing_skills:
        focus_areas.append(f"Validate missing critical skills: {', '.join(score.critical_missing_skills[:4])}")
    if score.missing_skills:
        focus_areas.append(f"Assess must-have skill depth: {', '.join(score.missing_skills[:4])}")
    for risk in score.risks[:2]:
        focus_areas.append(risk)

    hidden_strengths = score.strengths[:3]
    if not hidden_strengths and resume.projects:
        hidden_strengths.append("Project work indicates practical delivery experience.")

    rationale_parts = [
        f"{resume.name or 'Candidate'} was scored for {job.title or 'the role'} with a total score of {score.total_score}/100.",
    ]
    if score.strengths:
        rationale_parts.append("Strengths: " + " ".join(score.strengths[:2]))
    if score.risks:
        rationale_parts.append("Risks: " + " ".join(score.risks[:2]))

    return ReasoningOutput(
        fit_rationale=" ".join(part for part in rationale_parts if part).strip(),
        interview_focus_areas=focus_areas[:4],
        hidden_strengths=hidden_strengths[:4],
    )


def build_reasoning_prompt(job: JobDescriptionData, resume: ResumeData, score: CandidateScore) -> str:
    schema_json = ReasoningOutput.model_json_schema()
    return (
        "You are an expert technical HR reasoning assistant. Return valid JSON only that matches this schema: "
        f"{json.dumps(schema_json)}. "
        "Use the score and structured data to give a concise rationale, interview focus areas, and hidden strengths. "
        "Do not make or imply any selection outcome. The HR team will decide separately.\n\n"
        "**CRITICAL REASONING & CONTEXT RULES:**\n"
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
