import os
from dataclasses import dataclass
from functools import lru_cache


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    model_id: str
    tesseract_cmd: str | None
    poppler_path: str | None
    max_resumes_per_request: int
    max_pages_per_resume: int
    max_file_size_mb: int
    min_pdf_text_chars: int
    preload_model_on_startup: bool
    gemini_api_key: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        model_id=os.getenv("MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct"),
        tesseract_cmd=os.getenv("TESSERACT_CMD"),
        poppler_path=os.getenv("POPPLER_PATH", r"D:\Work\Poppler\poppler-25.12.0\Library\bin"),
        max_resumes_per_request=_get_int("MAX_RESUMES_PER_REQUEST", 10),
        max_pages_per_resume=_get_int("MAX_PAGES_PER_RESUME", 5),
        max_file_size_mb=_get_int("MAX_FILE_SIZE_MB", 10),
        min_pdf_text_chars=_get_int("MIN_PDF_TEXT_CHARS", 300),
        preload_model_on_startup=_get_bool("PRELOAD_MODEL_ON_STARTUP", True),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
    )
