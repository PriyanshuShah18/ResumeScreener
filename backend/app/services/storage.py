from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.ats_settings import ATSSettings


@dataclass(frozen=True)
class StoredResume:
    storage_key: str
    file_name: str
    mime_type: str
    content: bytes


class ResumeDeletedError(FileNotFoundError):
    def __init__(self, storage_key: str):
        self.storage_key = storage_key
        super().__init__(f"Resume object not found in S3: {storage_key}")


def _is_missing_s3_object_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False

    error = response.get("Error") or {}
    code = str(error.get("Code") or "").strip()
    if code in {"NoSuchKey", "NoSuchVersion", "NotFound", "404"}:
        return True

    metadata = response.get("ResponseMetadata") or {}
    return metadata.get("HTTPStatusCode") == 404


class S3ResumeStorage:
    def __init__(self, settings: ATSSettings, client: Any | None = None):
        self.settings = settings
        self.client = client or self._build_client(settings)

    def _build_client(self, settings: ATSSettings):
        access_key, secret_key = settings.require_s3_credentials()
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on deployment env
            raise RuntimeError("boto3 is required for S3 resume downloads") from exc

        return boto3.client(
            "s3",
            region_name=settings.s3_region,
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def download_resume(self, latest_resume: dict[str, Any]) -> StoredResume:
        storage_key = str(latest_resume.get("storageKey") or "").strip()
        if not storage_key:
            raise ValueError("latestResume.storageKey is required when parsedJson is missing")

        file_name = str(latest_resume.get("fileName") or "").strip() or Path(storage_key).name or "resume"
        mime_type = str(latest_resume.get("mimeType") or "").strip() or "application/octet-stream"
        try:
            response = self.client.get_object(Bucket=self.settings.s3_bucket, Key=storage_key)
        except Exception as exc:
            if _is_missing_s3_object_error(exc):
                raise ResumeDeletedError(storage_key) from exc
            raise
        content = response["Body"].read()

        return StoredResume(
            storage_key=storage_key,
            file_name=file_name,
            mime_type=mime_type,
            content=content,
        )
