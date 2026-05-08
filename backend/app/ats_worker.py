from __future__ import annotations

import argparse
import logging
import re
import time
import warnings
from datetime import datetime, timezone
from typing import Any


NOISY_DEPENDENCY_LOGGERS = (
    "httpx",
    "httpcore",
    "sentence_transformers",
    "huggingface_hub",
    "transformers",
)


def _suppress_noisy_worker_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"ARC4 has been moved.*",
        category=Warning,
        module=r"pypdf\._crypt_providers\._cryptography",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`resume_download` is deprecated.*",
        category=FutureWarning,
        module=r"huggingface_hub\.file_download",
    )


def _configure_worker_logging() -> None:
    _suppress_noisy_worker_warnings()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for logger_name in NOISY_DEPENDENCY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


_suppress_noisy_worker_warnings()

from app.agent import HRExtractionService
from app.ats_mongo import ATSMongoRepository
from app.ats_settings import ATSSettings, get_ats_settings
from app.config import get_settings
from app.document_parser import DocumentParser
from app.schemas import CandidateScore, JobDescriptionData, ResumeData, normalize_skill, normalize_whitespace
from app.scoring import score_candidate
from app.storage import ResumeDeletedError, S3ResumeStorage

logger = logging.getLogger(__name__)

ATS_STOPWORDS = {
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
    "using",
    "your",
    "you",
    "our",
    "job",
    "role",
    "work",
    "build",
    "modern",
    "applications",
}


class DeletedApplicationError(Exception):
    def __init__(self, reason: str, deleted_entity: str, details: dict[str, Any] | None = None):
        self.deleted_entity = deleted_entity
        self.details = details or {}
        super().__init__(reason)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalized_unique(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = normalize_skill(str(value))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _number(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else 0.0


def _extract_terms(text: str, limit: int = 30) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\+\#\.]{2,}", text):
        normalized = normalize_skill(token)
        if not normalized or normalized in ATS_STOPWORDS or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(normalized)
        if len(terms) >= limit:
            break
    return terms


def _job_text_from_hrms(job_doc: dict[str, Any]) -> str:
    parts = [
        f"Job title: {normalize_whitespace(job_doc.get('title', ''))}",
        f"Required skills: {', '.join(str(skill) for skill in _as_list(job_doc.get('skills')) if str(skill).strip())}",
        f"Minimum experience: {job_doc.get('experienceMinYears', 0)} years",
        f"Description: {normalize_whitespace(job_doc.get('description', ''))}",
    ]
    return "\n".join(part for part in parts if part.split(":", 1)[-1].strip())


def _job_from_hrms(job_doc: dict[str, Any]) -> JobDescriptionData:
    explicit_skills = _as_list(job_doc.get("skills"))
    description = normalize_whitespace(job_doc.get("description", ""))
    return JobDescriptionData(
        title=job_doc.get("title", ""),
        must_have_skills=explicit_skills,
        min_years_experience=job_doc.get("experienceMinYears", 0),
        responsibilities=[description] if description else [],
        domain_keywords=_extract_terms(f"{job_doc.get('title', '')} {description} {' '.join(str(s) for s in explicit_skills)}", limit=12),
    )


def _merge_job_context(job_doc: dict[str, Any], extracted_job: JobDescriptionData | None) -> JobDescriptionData:
    base = _job_from_hrms(job_doc)
    if not extracted_job:
        return base

    explicit_required = _normalized_unique(_as_list(job_doc.get("skills")))
    inferred_terms = (
        extracted_job.must_have_skills
        + extracted_job.good_to_have_skills
        + extracted_job.domain_keywords
    )
    inferred_preferred = [
        skill
        for skill in _normalized_unique(inferred_terms)
        if skill not in set(explicit_required)
    ]

    data = base.model_dump()
    data["must_have_skills"] = explicit_required or base.must_have_skills
    data["good_to_have_skills"] = _normalized_unique(base.good_to_have_skills + inferred_preferred)
    data["domain_keywords"] = _normalized_unique(
        base.domain_keywords
        + extracted_job.domain_keywords
        + inferred_preferred
        + _extract_terms(f"{base.title} {' '.join(base.responsibilities)}", limit=12)
    )
    data["required_education"] = _normalized_unique(extracted_job.required_education)
    data["required_certifications"] = _normalized_unique(extracted_job.required_certifications)
    data["responsibilities"] = base.responsibilities or extracted_job.responsibilities
    data["archetype"] = extracted_job.archetype or base.archetype
    return JobDescriptionData(**data)


def _job_from_hrms_with_extraction(job_doc: dict[str, Any], extraction_service: Any) -> JobDescriptionData:
    if not hasattr(extraction_service, "extract_job_description"):
        return _job_from_hrms(job_doc)

    try:
        extracted_job = extraction_service.extract_job_description(_job_text_from_hrms(job_doc))
    except Exception as exc:
        logger.warning("ATS job context extraction failed, using HRMS job fields only: %s", exc)
        return _job_from_hrms(job_doc)

    return _merge_job_context(
        job_doc,
        extracted_job if isinstance(extracted_job, JobDescriptionData) else None,
    )


def _resume_json_from_candidate(candidate_doc: dict[str, Any], parsed_json: Any | None) -> dict[str, Any]:
    if isinstance(parsed_json, ResumeData):
        return parsed_json.model_dump()
    if isinstance(parsed_json, dict):
        if isinstance(parsed_json.get("resume_data"), dict):
            return dict(parsed_json["resume_data"])
        return dict(parsed_json)

    latest_resume = candidate_doc.get("latestResume") or {}
    candidate_parsed = latest_resume.get("parsedJson")
    if isinstance(candidate_parsed, dict):
        return dict(candidate_parsed)
    return {}


def _resume_from_hrms_candidate(candidate_doc: dict[str, Any], parsed_json: Any | None = None) -> ResumeData:
    raw = _resume_json_from_candidate(candidate_doc, parsed_json)
    if "totalExperienceYears" in raw and "total_years_experience" not in raw:
        raw["total_years_experience"] = raw["totalExperienceYears"]

    profile_skills = _normalized_unique(_as_list(candidate_doc.get("skills")))
    resume = ResumeData(**raw)
    data = resume.model_dump()

    if not data.get("name") and candidate_doc.get("fullName"):
        data["name"] = candidate_doc.get("fullName")
    if not data.get("email") and candidate_doc.get("email"):
        data["email"] = candidate_doc.get("email")
    if not data.get("phone"):
        data["phone"] = candidate_doc.get("contact") or candidate_doc.get("phone") or ""
    if not data.get("location") and candidate_doc.get("location"):
        data["location"] = candidate_doc.get("location")
    data["skills"] = _normalized_unique(_as_list(data.get("skills")) + profile_skills)

    metadata = dict(data.get("metadata") or {})
    if candidate_doc.get("candidateId"):
        metadata["candidateId"] = candidate_doc.get("candidateId")
    if profile_skills:
        metadata["profileSkills"] = profile_skills
    data["metadata"] = metadata

    merged_resume = ResumeData(**data)
    profile_years = _number(candidate_doc.get("totalExperienceYears"))
    parsed_years = _number(raw.get("total_years_experience"))
    if profile_years or parsed_years:
        merged_resume.total_years_experience = max(
            merged_resume.total_years_experience,
            profile_years,
            parsed_years,
        )
    return merged_resume


def _dimension_percent(value: int, budget: int | None) -> int:
    if not budget:
        return 0
    return round(min(max(value / budget, 0.0), 1.0) * 100)


def _score_details(
    job_data: JobDescriptionData,
    resume_data: ResumeData,
    candidate_score: CandidateScore,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    budgets = candidate_score.budgets or {}
    return {
        "jobTitle": job_data.title,
        "skillMatchPercent": _dimension_percent(candidate_score.skills_score, budgets.get("skills")),
        "experienceMatchPercent": _dimension_percent(candidate_score.experience_score, budgets.get("experience")),
        "educationMatchPercent": _dimension_percent(candidate_score.education_score, budgets.get("education")),
        "keywordMatchPercent": _dimension_percent(candidate_score.keyword_score, budgets.get("keyword")),
        "completenessPercent": _dimension_percent(candidate_score.completeness_score, budgets.get("completeness")),
        "candidateExperienceYears": resume_data.total_years_experience,
        "requiredExperienceYears": job_data.min_years_experience,
        "matchedSkills": candidate_score.matched_skills,
        "missingSkills": candidate_score.missing_skills,
        "criticalMissingSkills": candidate_score.critical_missing_skills,
        "additionalRelevantSkills": candidate_score.additional_relevant_skills,
        "detectedDomainTags": candidate_score.detected_domain_tags,
        "skillsScore": candidate_score.skills_score,
        "experienceScore": candidate_score.experience_score,
        "educationScore": candidate_score.education_score,
        "keywordScore": candidate_score.keyword_score,
        "completenessScore": candidate_score.completeness_score,
        "confidenceScore": candidate_score.confidence_score,
        "riskScore": candidate_score.risk_score,
        "strengths": candidate_score.strengths,
        "risks": candidate_score.risks,
        "budgets": budgets,
        "semanticMatchDetails": candidate_score.semantic_match_details,
        "warnings": warnings or [],
    }


def _build_ats_log(
    organization_id: str,
    application: dict[str, Any],
    score: int,
    details: dict[str, Any],
    stage: str = "SCORE",
    created_at_iso: str | None = None,
    created_by: str = "ats-engine",
    created_by_name: str = "ATS Service",
) -> dict[str, Any]:
    created_at = created_at_iso or datetime.now(timezone.utc).isoformat()
    return {
        "organizationId": organization_id,
        "jobId": application.get("jobId"),
        "candidateId": application.get("candidateId"),
        "applicationId": application.get("applicationId"),
        "stage": stage,
        "score": score,
        "details": details,
        "createdBy": created_by,
        "createdByName": created_by_name,
        "createdAt": created_at,
    }


class ATSWorker:
    def __init__(
        self,
        repo: Any,
        storage: Any,
        document_parser: Any,
        extraction_service: Any,
        settings: ATSSettings,
    ):
        self.repo = repo
        self.storage = storage
        self.document_parser = document_parser
        self.extraction_service = extraction_service
        self.settings = settings

    def process_once(self) -> dict[str, int]:
        logger.info(
            "Mongo fetch pending applications: status=%s limit=%d",
            self.settings.applied_status,
            self.settings.batch_size,
        )
        applications = self.repo.fetch_pending_applications(
            limit=self.settings.batch_size,
            status=self.settings.applied_status,
        )
        logger.info("Mongo fetch pending applications complete: found=%d", len(applications))
        summary = {"found": len(applications), "processed": 0, "failed": 0, "deleted": 0}

        for application in applications:
            try:
                self.process_application(application)
                summary["processed"] += 1
            except DeletedApplicationError as exc:
                logger.warning(
                    "ATS scoring skipped for deleted application %s: %s",
                    application.get("applicationId"),
                    exc,
                )
                self._insert_deleted_log(application, exc)
                summary["deleted"] += 1
            except Exception as exc:
                logger.exception("ATS scoring failed for application %s", application.get("applicationId"))
                self._insert_failure_log(application, exc)
                summary["failed"] += 1

        return summary

    def process_application(self, application: dict[str, Any]) -> dict[str, Any]:
        application_id = application.get("applicationId")
        candidate_id = application.get("candidateId")
        job_id = application.get("jobId")
        if not application_id:
            raise ValueError("applicationId is required")
        if not candidate_id:
            raise ValueError(f"candidateId is required for application {application_id}")
        if not job_id:
            raise ValueError(f"jobId is required for application {application_id}")

        logger.info("Mongo fetch candidate: candidateId=%s applicationId=%s", candidate_id, application_id)
        candidate = self.repo.get_candidate(candidate_id)
        logger.info("Mongo fetch candidate complete: candidateId=%s found=%s", candidate_id, bool(candidate))
        if not candidate:
            raise DeletedApplicationError(
                f"Candidate not found for candidateId={candidate_id}",
                deleted_entity="candidate",
                details={"candidateId": candidate_id},
            )
        logger.info("Mongo fetch job: jobId=%s applicationId=%s", job_id, application_id)
        job = self.repo.get_job(job_id)
        logger.info("Mongo fetch job complete: jobId=%s found=%s", job_id, bool(job))
        if not job:
            raise ValueError(f"Job not found for jobId={job_id}")

        latest_resume = candidate.get("latestResume") or {}
        parsed_json = latest_resume.get("parsedJson")
        warnings: list[str] = []
        resume_source = "parsedJson" if parsed_json else "s3"
        parsed_resume: Any = parsed_json

        if not parsed_json:
            logger.info(
                "Mongo resume cache miss: candidateId=%s applicationId=%s",
                candidate_id,
                application_id,
            )
            logger.info(
                "S3 fetch resume: candidateId=%s storageKey=%s fileName=%s",
                candidate_id,
                latest_resume.get("storageKey"),
                latest_resume.get("fileName"),
            )
            try:
                stored_resume = self.storage.download_resume(latest_resume)
            except (ResumeDeletedError, FileNotFoundError) as exc:
                raise DeletedApplicationError(
                    str(exc),
                    deleted_entity="resume",
                    details={
                        "resumeSource": "s3",
                        "resumeStorageKey": latest_resume.get("storageKey"),
                        "resumeFileName": latest_resume.get("fileName"),
                    },
                ) from exc
            logger.info(
                "S3 fetch resume complete: candidateId=%s storageKey=%s bytes=%d",
                candidate_id,
                stored_resume.storage_key,
                len(stored_resume.content),
            )
            parsed_document = self.document_parser.parse_upload(stored_resume.file_name, stored_resume.content)
            warnings.extend(parsed_document.warnings)
            extracted_resume = self.extraction_service.extract_resume(
                parsed_document.text,
                page_images=parsed_document.page_images,
            )
            parsed_resume = extracted_resume
            if self.settings.cache_parsed_json:
                logger.info("Mongo write parsed resume cache: candidateId=%s", candidate_id)
                self.repo.cache_candidate_parsed_resume(candidate_id, extracted_resume.model_dump())
                logger.info("Mongo write parsed resume cache complete: candidateId=%s", candidate_id)
            else:
                logger.info("Mongo parsed resume cache disabled: candidateId=%s", candidate_id)
        else:
            logger.info(
                "Mongo resume cache hit: candidateId=%s applicationId=%s",
                candidate_id,
                application_id,
            )

        job_data = _job_from_hrms_with_extraction(job, self.extraction_service)
        resume_data = _resume_from_hrms_candidate(candidate, parsed_resume)
        candidate_score = score_candidate(job_data, resume_data)
        details = _score_details(job_data, resume_data, candidate_score, warnings=warnings)
        details["resumeSource"] = resume_source
        details["applicationStatus"] = application.get("status")
        if latest_resume.get("storageKey"):
            details["resumeStorageKey"] = latest_resume.get("storageKey")
        if latest_resume.get("fileName"):
            details["resumeFileName"] = latest_resume.get("fileName")

        log_doc = _build_ats_log(
            organization_id=self.settings.organization_id,
            application=application,
            score=candidate_score.total_score,
            details=details,
            stage=self.settings.score_stage,
            created_by=self.settings.created_by,
            created_by_name=self.settings.created_by_name,
        )
        self._write_score_log(log_doc)

        return log_doc

    def _insert_deleted_log(self, application: dict[str, Any], exc: DeletedApplicationError) -> None:
        details = {
            "status": "DELETED",
            "deletedEntity": exc.deleted_entity,
            "reason": str(exc),
            "warnings": [str(exc)],
            "applicationStatus": application.get("status"),
        }
        details.update(
            {
                key: value
                for key, value in exc.details.items()
                if value not in (None, "")
            }
        )
        log_doc = _build_ats_log(
            organization_id=self.settings.organization_id,
            application=application,
            score=0,
            details=details,
            stage=self.settings.score_stage,
            created_at_iso=datetime.now(timezone.utc).isoformat(),
            created_by=self.settings.created_by,
            created_by_name=self.settings.created_by_name,
        )
        self._write_score_log(log_doc)

    def _insert_failure_log(self, application: dict[str, Any], exc: Exception) -> None:
        log_doc = _build_ats_log(
            organization_id=self.settings.organization_id,
            application=application,
            score=0,
            details={
                "status": "FAILED",
                "error": str(exc),
                "warnings": [str(exc)],
            },
            stage=self.settings.score_stage,
            created_at_iso=datetime.now(timezone.utc).isoformat(),
            created_by=self.settings.created_by,
            created_by_name=self.settings.created_by_name,
        )
        self._write_score_log(log_doc)

    def _write_score_log(self, log_doc: dict[str, Any]) -> Any:
        details = log_doc.get("details") or {}
        logger.info(
            "Mongo write ATS score log: applicationId=%s stage=%s score=%s status=%s",
            log_doc.get("applicationId"),
            log_doc.get("stage"),
            log_doc.get("score"),
            details.get("status", "SCORED"),
        )
        result = self.repo.insert_score_log(log_doc)
        logger.info(
            "Mongo write ATS score log complete: applicationId=%s",
            log_doc.get("applicationId"),
        )
        return result


def create_worker(settings: ATSSettings | None = None) -> ATSWorker:
    ats_settings = settings or get_ats_settings()
    core_settings = get_settings()
    repo = ATSMongoRepository.from_settings(ats_settings)
    storage = S3ResumeStorage(ats_settings)
    document_parser = DocumentParser(core_settings)
    extraction_service = HRExtractionService(core_settings)
    return ATSWorker(
        repo=repo,
        storage=storage,
        document_parser=document_parser,
        extraction_service=extraction_service,
        settings=ats_settings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HRMS ATS scoring worker")
    parser.add_argument("--once", action="store_true", help="Process one batch and exit")
    parser.add_argument("--ensure-indexes", action="store_true", help="Create recommended MongoDB indexes before processing")
    parser.add_argument(
        "--clear-parsed-cache",
        action="store_true",
        help="Unset cached latestResume.parsedJson documents before processing",
    )
    args = parser.parse_args()

    _configure_worker_logging()
    worker = create_worker()
    if args.ensure_indexes:
        worker.repo.ensure_indexes()
    if args.clear_parsed_cache:
        cleared_count = worker.repo.clear_candidate_parsed_resume_cache()
        logger.info("Mongo cleared parsed resume cache: candidates=%d", cleared_count)

    while True:
        summary = worker.process_once()
        logger.info(
            "ATS worker batch complete: found=%d processed=%d failed=%d deleted=%d",
            summary["found"],
            summary["processed"],
            summary["failed"],
            summary.get("deleted", 0),
        )
        if args.once:
            break
        time.sleep(worker.settings.poll_interval_seconds)


if __name__ == "__main__":
    main()
