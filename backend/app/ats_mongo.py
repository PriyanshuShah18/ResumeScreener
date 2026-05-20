from __future__ import annotations

from typing import Any

from app.ats_settings import ATSSettings


class ATSMongoRepository:
    """
    Repository for ATS worker reads and writes.
    Supports both:
      - modern inline application writes (`write_ats_result`)
      - legacy score log collection writes (`insert_score_log`)
    """

    def __init__(self, db: Any):
        self.db = db
        self.applications = db["hrmsjobapplications"]
        self.candidates = db["hrmscandidates"]
        self.jobs = db["hrmsjobs"]
        try:
            self.score_logs = db["hrmsatslogs"]
        except Exception:
            self.score_logs = None

    @classmethod
    def from_settings(cls, settings: ATSSettings) -> "ATSMongoRepository":
        try:
            from pymongo import MongoClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pymongo is required for the ATS worker") from exc

        client = MongoClient(settings.require_mongodb_uri())
        return cls(client[settings.mongodb_db])

    def ensure_indexes(self) -> None:
        """Create recommended indexes for efficient polling."""
        self._ensure_index(
            self.applications,
            [("atsProcessed", 1), ("status", 1)],
        )
        self._ensure_index(self.candidates, [("candidateId", 1)])
        self._ensure_index(self.jobs, [("jobId", 1)])

    def _ensure_index(self, collection: Any, spec: list[tuple[str, int]], **kwargs: Any) -> None:
        expected_key = list(spec)
        for index in collection.index_information().values():
            if list(index.get("key", [])) == expected_key:
                return
        collection.create_index(spec, **kwargs)

    # ── Read ────────────────────────────────────────────────────────────────

    def fetch_pending_applications(self, limit: int, status: str = "APPLIED") -> list[dict[str, Any]]:
        """
        Return unprocessed applications in APPLIED status.
        Rules:
          - status must match
          - skip apps already atsProcessed=True
          - skip apps with an existing SCORE log unless that log is FAILED
        """
        applications = list(self.applications.find({"status": status}))
        pending: list[dict[str, Any]] = []
        for application in applications:
            if application.get("atsProcessed") is True:
                continue
            if self._has_completed_score_log(application.get("applicationId")):
                continue
            pending.append(application)
            if len(pending) >= limit:
                break
        return pending

    def _has_completed_score_log(self, application_id: Any) -> bool:
        if self.score_logs is None or not application_id:
            return False
        existing = self.score_logs.find_one({"applicationId": application_id, "stage": "SCORE"})
        if not existing:
            return False
        details = existing.get("details") if isinstance(existing, dict) else {}
        status = str((details or {}).get("status", "")).upper()
        return status != "FAILED"

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        return self.candidates.find_one({"candidateId": candidate_id})

    def get_job(self, job_id: Any) -> dict[str, Any] | None:
        job = self.jobs.find_one({"jobId": job_id})
        if job:
            return job

        # Fallback: try by MongoDB _id
        try:
            from bson import ObjectId
        except ImportError:  # pragma: no cover
            return None

        if isinstance(job_id, ObjectId):
            return self.jobs.find_one({"_id": job_id})

        job_id_text = str(job_id)
        if ObjectId.is_valid(job_id_text):
            return self.jobs.find_one({"_id": ObjectId(job_id_text)})

        return None

    # ── Write ───────────────────────────────────────────────────────────────

    def write_ats_result(
        self,
        doc_id: Any,
        score: int,
        details: dict[str, Any],
    ) -> None:
        """
        Write the ATS score back to the application document.
        Sets atsProcessed=True so the frontend ATS page can display it.
        """
        from datetime import datetime, timezone

        self.applications.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    "atsProcessed": True,
                    "atsScore": score,
                    "atsProcessedAt": datetime.now(timezone.utc),
                    "atsDetails": details,
                }
            },
        )

    def insert_score_log(self, log: dict[str, Any]) -> Any:
        """Legacy helper: write ATS score logs to hrmsatslogs."""
        if self.score_logs is None:
            return None
        result = self.score_logs.insert_one(log)
        return getattr(result, "inserted_id", None)

    def cache_candidate_parsed_resume(self, candidate_id: str, parsed_json: dict[str, Any]) -> None:
        """Cache the expensive LLM-parsed resume JSON on the candidate document."""
        self.candidates.update_one(
            {"candidateId": candidate_id},
            {"$set": {"latestResume.parsedJson": parsed_json}},
        )

    def update_application_status(self, application_id: str, status: str) -> None:
        self.applications.update_one(
            {"applicationId": application_id},
            {"$set": {"status": status}},
        )

    def clear_candidate_parsed_resume_cache(self) -> int:
        result = self.candidates.update_many(
            {"latestResume.parsedJson": {"$exists": True}},
            {"$unset": {"latestResume.parsedJson": ""}},
        )
        return int(getattr(result, "modified_count", 0) or 0)
