from __future__ import annotations

import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.agent import HRExtractionService
from app.config import get_settings
from app.document_parser import DocumentParser
from app.schemas import ScreenResumesResponse, ScreeningResult
from app.scoring import build_recruiter_feedback, score_candidate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

settings = get_settings()
document_parser = DocumentParser(settings)
extraction_service = HRExtractionService(settings)

app = FastAPI(title="HR Resume Screening API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    if settings.preload_model_on_startup:
        extraction_service.preload()


@app.get("/")
def root() -> dict[str, object]:
    return {
        "message": "HR Resume Screening API",
        "docs": "/docs",
        "health": "/health",
        "screen_resumes": "/screen-resumes",
    }


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": extraction_service.model_available,
        "model_error": extraction_service.load_error,
    }


@app.post("/screen-resumes", response_model=ScreenResumesResponse)
async def screen_resumes(
    job_description: str = Form(...),
    resumes: list[UploadFile] = File(...),
    top_k: int = Form(5),
) -> ScreenResumesResponse:
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="job_description cannot be empty")
    if not resumes:
        raise HTTPException(status_code=400, detail="At least one resume is required")
    if len(resumes) > settings.max_resumes_per_request:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_resumes_per_request} resumes are allowed per request",
        )
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1")

    invalid_files = [resume.filename or "unnamed" for resume in resumes if not document_parser.is_supported(resume.filename or "")]
    if invalid_files:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported resume file types: {', '.join(invalid_files)}",
        )

    warnings: list[str] = []
    if extraction_service.load_error:
        warnings.append(f"Model unavailable, using heuristic extraction: {extraction_service.load_error}")

    try:
        job_summary = extraction_service.extract_job_description(job_description)
    except Exception as exc:
        logger.exception("Job description extraction failed")
        raise HTTPException(status_code=500, detail=f"Failed to parse job description: {exc}") from exc

    if extraction_service.load_error:
        model_warning = f"Model unavailable, using heuristic extraction: {extraction_service.load_error}"
        if model_warning not in warnings:
            warnings.append(model_warning)

    results: list[ScreeningResult] = []
    failed_count = 0

    for resume in resumes:
        filename = resume.filename or "unnamed"
        file_warnings: list[str] = []

        try:
            content = await resume.read()
            parsed = document_parser.parse_upload(filename, content)
            file_warnings.extend(parsed.warnings)

            resume_data = extraction_service.extract_resume(parsed.text, page_images=parsed.page_images)
            candidate_score = score_candidate(job_summary, resume_data)
            recruiter_feedback = build_recruiter_feedback(job_summary, resume_data, candidate_score)

            results.append(
                ScreeningResult(
                    source_file=filename,
                    resume_data=resume_data,
                    score=candidate_score,
                    recruiter_feedback=recruiter_feedback,
                    warnings=file_warnings,
                )
            )
            warnings.extend(f"{filename}: {warning}" for warning in file_warnings)
        except Exception as exc:
            failed_count += 1
            logger.exception("Resume processing failed for %s", filename)
            warnings.append(f"{filename}: {exc}")

    results.sort(key=lambda item: item.score.total_score, reverse=True)
    capped_top_k = min(top_k, len(results))

    return ScreenResumesResponse(
        job_summary=job_summary,
        results=results[:capped_top_k],
        warnings=warnings,
        processed_count=len(results),
        failed_count=failed_count,
    )
