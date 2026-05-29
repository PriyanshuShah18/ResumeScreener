from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.ats_settings import ATSSettings

logger = logging.getLogger(__name__)


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


def _is_access_denied_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False

    error = response.get("Error") or {}
    code = str(error.get("Code") or "").strip()
    if code in {"AccessDenied", "403", "InvalidAccessKeyId", "SignatureDoesNotMatch"}:
        return True

    metadata = response.get("ResponseMetadata") or {}
    return metadata.get("HTTPStatusCode") == 403


def _mask_key(key: str | None) -> str:
    """Show first 4 and last 2 characters of a credential, mask the rest."""
    if not key:
        return "<not set>"
    if len(key) <= 8:
        return key[:2] + "***"
    return key[:4] + "***" + key[-2:]


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

        logger.info(
            "S3 client init: region=%s bucket=%s endpoint=%s accessKeyId=%s",
            settings.s3_region,
            settings.s3_bucket,
            settings.s3_endpoint_url or "<default>",
            _mask_key(access_key),
        )

        client = boto3.client(
            "s3",
            region_name=settings.s3_region,
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        # Quick connectivity check: verify the bucket is reachable
        try:
            client.head_bucket(Bucket=settings.s3_bucket)
            logger.info("S3 bucket verified: bucket=%s — accessible", settings.s3_bucket)
        except Exception as head_exc:
            # Log the error but don't fail — download attempts will surface the real issue
            logger.warning(
                "S3 bucket check failed: bucket=%s error=%s — downloads may fail",
                settings.s3_bucket,
                head_exc,
            )

        return client

    def download_resume(self, latest_resume: dict[str, Any]) -> StoredResume:
        storage_key = str(latest_resume.get("storageKey") or "").strip()
        if not storage_key:
            raise ValueError("latestResume.storageKey is required when parsedJson is missing")

        file_name = str(latest_resume.get("fileName") or "").strip() or Path(storage_key).name or "resume"
        mime_type = str(latest_resume.get("mimeType") or "").strip() or "application/octet-stream"

        logger.info(
            "S3 download start: bucket=%s key=%s fileName=%s",
            self.settings.s3_bucket,
            storage_key,
            file_name,
        )

        try:
            response = self.client.get_object(Bucket=self.settings.s3_bucket, Key=storage_key)
        except Exception as exc:
            self._log_download_error(exc, storage_key)
            if _is_missing_s3_object_error(exc):
                raise ResumeDeletedError(storage_key) from exc
            raise

        content = response["Body"].read()
        content_length = response.get("ContentLength", len(content))

        logger.info(
            "S3 download complete: bucket=%s key=%s bytes=%d contentType=%s",
            self.settings.s3_bucket,
            storage_key,
            content_length,
            response.get("ContentType", "unknown"),
        )

        return StoredResume(
            storage_key=storage_key,
            file_name=file_name,
            mime_type=mime_type,
            content=content,
        )

    def _log_download_error(self, exc: Exception, storage_key: str) -> None:
        """Classify and log the S3 error with actionable diagnostics."""
        error_response = getattr(exc, "response", None) or {}
        error_info = error_response.get("Error", {}) if isinstance(error_response, dict) else {}
        error_code = str(error_info.get("Code", "unknown"))
        error_message = str(error_info.get("Message", str(exc)))
        http_status = (
            error_response.get("ResponseMetadata", {}).get("HTTPStatusCode", "unknown")
            if isinstance(error_response, dict)
            else "unknown"
        )

        if _is_missing_s3_object_error(exc):
            logger.warning(
                "S3 object not found: bucket=%s key=%s errorCode=%s httpStatus=%s "
                "— the file may have been deleted or the key is incorrect",
                self.settings.s3_bucket,
                storage_key,
                error_code,
                http_status,
            )
        elif _is_access_denied_error(exc):
            logger.error(
                "S3 access denied: bucket=%s key=%s errorCode=%s httpStatus=%s accessKeyId=%s region=%s "
                "— check IAM permissions (s3:GetObject) and verify credentials are correct",
                self.settings.s3_bucket,
                storage_key,
                error_code,
                http_status,
                _mask_key(self.settings.s3_access_key_id),
                self.settings.s3_region,
            )
        else:
            logger.error(
                "S3 download failed: bucket=%s key=%s errorCode=%s httpStatus=%s message=%s "
                "region=%s endpoint=%s — this may be a network or configuration issue",
                self.settings.s3_bucket,
                storage_key,
                error_code,
                http_status,
                error_message,
                self.settings.s3_region,
                self.settings.s3_endpoint_url or "<default>",
                exc_info=True,
            )
