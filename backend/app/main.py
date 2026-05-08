
import logging
import asyncio
import time
import uuid
import contextvars
from contextlib import asynccontextmanager

from typing import List, Annotated
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.agent import HRExtractionService
from app.config import get_settings
from app.constants import SUPPORTED_EXTENSIONS
from app.document_parser import DocumentParser
from app.ranking import rank_results
from app.reasoning import generate_reasoning
from app.schemas import ScreenResumesResponse, ScreeningResult
from app.scoring import build_recruiter_feedback, score_candidate
from app.llm_understanding import llm_service
from app import metrics

_request_id_var = contextvars.ContextVar("request_id", default="-")

logger = logging.getLogger(__name__)

# Structured logging: Include [request_id] in every line
log_format = "%(levelname)s:    %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = str(uuid.uuid4())[:8]
        _request_id_var.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

def log_info(msg, *args):
    logger.info(f"[{_request_id_var.get()}] {msg}", *args)

limiter = Limiter(key_func=get_remote_address)


class _MultipartBoundaryFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Skipping data after last boundary" not in record.getMessage()


logging.getLogger("python_multipart.multipart").addFilter(_MultipartBoundaryFilter())

settings = get_settings()
document_parser = DocumentParser(settings)
extraction_service = HRExtractionService(settings)


# ─── Modern Lifespan API (replaces deprecated on_event) ──────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    models = []
    if llm_service.groq_client: models.append("Groq")
    if llm_service.gemini_model: models.append("Gemini")
    if llm_service.openrouter_key: models.append("OpenRouter")
    if extraction_service._ollama_available: models.append("Ollama VLM")
    log_info(f"Backend started. Available extractors: {', '.join(models) if models else 'Heuristics only'}")

    # ── Pre-warm skill graph for common skills (3.2) ──
    try:
        await asyncio.to_thread(
            llm_service.generate_skill_graph,
            ["python", "javascript", "sql", "react", "aws", "docker"]
        )
        log_info("Skill graph pre-warmed.")
    except Exception as e:
        logger.warning(f"Failed to pre-warm skill graph: {e}")

    yield  # ── Application runs here ──

    # ── Shutdown — release GPU / model resources ──
    logger.info("Shutting down HR Screening API...")
    try:
        from app.semantic_matching import get_semantic_matcher
        matcher = get_semantic_matcher()
        if matcher._model:
            del matcher._model
            matcher._model = None
            logger.info("Semantic model resources released.")
    except Exception:
        pass
    logger.info("Shutdown complete.")


app = FastAPI(title="HR Resume Screening API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, object]:
    return {
        "message": "HR Resume Screening API (Production)",
        "docs": "/docs",
        "health": "/health",
        "screen_resumes": "/screen-resumes",
    }


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "ollama_connected": extraction_service._ollama_available,
        "groq_enabled": llm_service.enabled,
        "model_available": extraction_service.model_available,
    }


@app.get("/metrics")
def get_metrics() -> dict:
    return metrics.get_metrics()


@app.get("/supported-formats")
def supported_formats() -> dict:
    return {
        "formats": sorted(SUPPORTED_EXTENSIONS),
        "max_size_mb": settings.max_file_size_mb,
        "max_resumes_per_request": settings.max_resumes_per_request,
    }


@app.get("/debug-ping")
def debug_ping():
    log_info("🏓 Ping received")
    return {"status": "pong"}


@app.post("/screen-resumes", response_model=ScreenResumesResponse)
async def screen_resumes(
    request: Request,
    job_description: str = Form(...),
    resumes: List[UploadFile] = File(...),
    top_k: int = Form(5),
) -> ScreenResumesResponse:
    batch_start = time.perf_counter()
    log_info("🚀 Received screening request for %d resumes", len(resumes))

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
    if not extraction_service.model_available:
        warnings.append("No AI model available (Ollama/Groq/Gemini). Using heuristic extraction only.")

    # ── Step 1: Extract Job Description ──
    try:
        log_info("🔍 Extracting job description...")
        jd_start = time.perf_counter()
        job_summary = await asyncio.to_thread(extraction_service.extract_job_description, job_description)
        log_info("📋 JD extraction took %.1fs", time.perf_counter() - jd_start)
    except Exception as exc:
        logger.exception("Job description extraction failed")
        raise HTTPException(status_code=500, detail=f"Failed to parse job description: {exc}") from exc

    # ── Step 2: Pre-read all file bytes with size validation ──
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    file_contents: list[tuple[str, bytes]] = []
    for resume in resumes:
        filename = resume.filename or "unnamed"
        content = await resume.read()
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{filename} exceeds the {settings.max_file_size_mb}MB size limit.",
            )
        file_contents.append((filename, content))

    # ── Step 3: Process resumes with STAGGERED launch ──
    # We launch one resume every 0.5s to prevent "burst" rate limits (429s).
    semaphore = asyncio.Semaphore(8)
    results: list[ScreeningResult] = []
    failed_count = 0

    async def process_single_resume(filename: str, content: bytes) -> ScreeningResult | Exception:
        async with semaphore:
            try:
                def _sync_process() -> ScreeningResult:
                    resume_start = time.perf_counter()
                    file_warnings: list[str] = []
                    parsed = document_parser.parse_upload(filename, content)
                    file_warnings.extend(parsed.warnings)
                    resume_data = extraction_service.extract_resume(parsed.text, page_images=parsed.page_images)
                    candidate_score = score_candidate(job_summary, resume_data)
                    reasoning_output = generate_reasoning(job_summary, resume_data, candidate_score, extraction_service)
                    recruiter_feedback = build_recruiter_feedback(
                        job_summary,
                        resume_data,
                        candidate_score,
                        reasoning_summary=reasoning_output.fit_rationale,
                    )
                    log_info("📄 %s processed in %.1fs", filename, time.perf_counter() - resume_start)
                    return ScreeningResult(
                        source_file=filename,
                        resume_data=resume_data,
                        score=candidate_score,
                        recruiter_feedback=recruiter_feedback,
                        hiring_decision=reasoning_output.hiring_decision,
                        interview_focus_areas=reasoning_output.interview_focus_areas,
                        hidden_strengths=reasoning_output.hidden_strengths,
                        warnings=file_warnings,
                    )
                return await asyncio.to_thread(_sync_process)
            except Exception as exc:
                logger.exception("Resume processing failed for %s", filename)
                return exc

    # Launch tasks with a 0.3s gap between each to respect RPM limits
    tasks = []
    for filename, content in file_contents:
        tasks.append(asyncio.create_task(process_single_resume(filename, content)))
        await asyncio.sleep(0.3)  # Fast staggered launch ⚡

    outcomes = await asyncio.gather(*tasks)

    for (filename, _), outcome in zip(file_contents, outcomes):
        if isinstance(outcome, Exception):
            failed_count += 1
            warnings.append(f"{filename}: {outcome}")
        else:
            results.append(outcome)
            warnings.extend(f"{filename}: {w}" for w in outcome.warnings)

    results = rank_results(results)

    # Pool-relative recommendation normalization
    if len(results) >= 3:
        scores = [r.score.total_score for r in results]
        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        if std >= 3:  # Only normalize when there's meaningful spread
            for r in results:
                z = (r.score.total_score - mean) / max(std, 1.0)
                if z >= 1.0:
                    r.score.recommendation = "strong fit (top of pool)"
                elif z >= 0.25:
                    r.score.recommendation = "good fit"
                elif z >= -0.5:
                    r.score.recommendation = "review"
                else:
                    r.score.recommendation = "low fit"

    capped_top_k = min(top_k, len(results))

    total_elapsed = time.perf_counter() - batch_start
    log_info(
        "🏁 Batch complete: %d/%d resumes in %.1fs (%.1fs/resume avg)",
        len(results),
        len(file_contents),
        total_elapsed,
        total_elapsed / max(len(file_contents), 1),
    )

    # Record metrics for /metrics endpoint
    metrics.record_batch(len(results), failed_count, total_elapsed)

    return ScreenResumesResponse(
        job_summary=job_summary,
        results=results[:capped_top_k],
        warnings=warnings,
        processed_count=len(results),
        failed_count=failed_count,
    )
