from __future__ import annotations

from typing import Any

from app.ats_settings import ATSSettings


class ATSMongoRepository:
    def __init__(self, db: Any):
        self.db = db
        self.applications = db["hrmsjobapplications"]
        self.candidates = db["hrmscandidates"]
        self.jobs = db["hrmsjobs"]
        self.logs = db["hrmsatslogs"]

    @classmethod
    def from_settings(cls, settings: ATSSettings) -> "ATSMongoRepository":
        try:
            from pymongo import MongoClient
        except ImportError as exc:  # pragma: no cover - depends on deployment env
            raise RuntimeError("pymongo is required for the ATS worker") from exc

        client = MongoClient(settings.require_mongodb_uri())
        return cls(client[settings.mongodb_db])

    def ensure_indexes(self) -> None:
        self._ensure_index(self.applications, [("status", 1), ("applicationId", 1)])
        self._ensure_index(self.logs, [("applicationId", 1), ("stage", 1)], unique=True)
        self._ensure_index(self.candidates, [("candidateId", 1)])
        self._ensure_index(self.jobs, [("jobId", 1)])

    def _ensure_index(self, collection: Any, spec: list[tuple[str, int]], **kwargs: Any) -> None:
        expected_key = list(spec)
        for index in collection.index_information().values():
            if list(index.get("key", [])) == expected_key:
                return
        collection.create_index(spec, **kwargs)

    def score_log_exists(self, application_id: str) -> bool:
        log = self.logs.find_one({"applicationId": application_id, "stage": "SCORE"})
        if not log:
            return False
        return (log.get("details") or {}).get("status") != "FAILED"

    def fetch_pending_applications(self, limit: int, status: str = "APPLIED") -> list[dict[str, Any]]:
        pending: list[dict[str, Any]] = []
        for application in self.applications.find({"status": status}):
            application_id = application.get("applicationId")
            if not application_id or self.score_log_exists(application_id):
                continue
            pending.append(application)
            if len(pending) >= limit:
                break
        return pending

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        return self.candidates.find_one({"candidateId": candidate_id})

    def get_job(self, job_id: Any) -> dict[str, Any] | None:
        job = self.jobs.find_one({"jobId": job_id})
        if job:
            return job

        try:
            from bson import ObjectId
        except ImportError:  # pragma: no cover - pymongo provides bson in deployment
            return None

        if isinstance(job_id, ObjectId):
            return self.jobs.find_one({"_id": job_id})

        job_id_text = str(job_id)
        if ObjectId.is_valid(job_id_text):
            return self.jobs.find_one({"_id": ObjectId(job_id_text)})

        return None

    def insert_score_log(self, log_doc: dict[str, Any]) -> Any:
        query = {"applicationId": log_doc.get("applicationId"), "stage": log_doc.get("stage")}
        if query["applicationId"] and query["stage"] and hasattr(self.logs, "replace_one"):
            result = self.logs.replace_one(query, log_doc, upsert=True)
            return result.upserted_id
        return self.logs.insert_one(log_doc).inserted_id

    def cache_candidate_parsed_resume(self, candidate_id: str, parsed_json: dict[str, Any]) -> None:
        self.candidates.update_one(
            {"candidateId": candidate_id},
            {"$set": {"latestResume.parsedJson": parsed_json}},
        )

    def clear_candidate_parsed_resume_cache(self) -> int:
        result = self.candidates.update_many(
            {"latestResume.parsedJson": {"$exists": True}},
            {"$unset": {"latestResume.parsedJson": ""}},
        )
        return int(getattr(result, "modified_count", 0) or 0)

    def update_application_status(self, application_id: str, status: str) -> None:
        self.applications.update_one(
            {"applicationId": application_id},
            {"$set": {"status": status}},
        )
