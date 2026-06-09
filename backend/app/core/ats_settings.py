from __future__ import annotations
#Testing
import os
from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int, minimum: int = 0) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(parsed, minimum)


@dataclass(frozen=True)
class ATSSettings:
    mongodb_uri: str | None = None
    mongodb_db: str = "Os"
    organization_id: str = "codnestx"
    s3_bucket: str = "harsh-gajjar-280"
    s3_region: str = "ap-southeast-2"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_endpoint_url: str | None = None
    poll_interval_seconds: int = 60
    batch_size: int = 25
    auto_shortlist: bool = False
    shortlist_threshold: int = 80
    cache_parsed_json: bool = False
    applied_status: str = "APPLIED"
    shortlisted_status: str = "SHORTLISTED"
    score_stage: str = "SCORE"
    created_by: str = "ats-engine"
    created_by_name: str = "ATS Service"

    def require_mongodb_uri(self) -> str:
        if not self.mongodb_uri:
            raise RuntimeError("MONGODB_URI or MONGO_URI is required for the ATS worker")
        return self.mongodb_uri

    def require_s3_credentials(self) -> tuple[str, str]:
        if not self.s3_access_key_id or not self.s3_secret_access_key:
            raise RuntimeError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required to download resumes")
        return self.s3_access_key_id, self.s3_secret_access_key


def get_ats_settings() -> ATSSettings:
    return ATSSettings(
        mongodb_uri=os.getenv("MONGODB_URI") or os.getenv("MONGO_URI"),
        mongodb_db=os.getenv("MONGODB_DB", "Os"),
        organization_id=os.getenv("DEFAULT_ORGANIZATION_ID", "codnestx"),
        s3_bucket=os.getenv("S3_BUCKET", "harsh-gajjar-280"),
        s3_region=os.getenv("AWS_REGION") or os.getenv("S3_REGION", "ap-southeast-2"),
        s3_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("S3_ACCESS_KEY_ID"),
        s3_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("S3_SECRET_ACCESS_KEY"),
        s3_endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        poll_interval_seconds=_get_int("ATS_POLL_INTERVAL_SECONDS", 60, minimum=1),
        batch_size=_get_int("ATS_BATCH_SIZE", 25, minimum=1),
        auto_shortlist=_get_bool("ATS_AUTO_SHORTLIST", False),
        shortlist_threshold=_get_int("ATS_SHORTLIST_THRESHOLD", 80, minimum=0),
        cache_parsed_json=_get_bool("ATS_CACHE_PARSED_JSON", False),
    )
