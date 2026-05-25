import os
from dataclasses import dataclass, field
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

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
    ollama_base_url: str
    tesseract_cmd: str | None
    poppler_path: str | None
    max_resumes_per_request: int
    max_pages_per_resume: int
    max_file_size_mb: int
    min_pdf_text_chars: int
    preload_model_on_startup: bool
    gemini_api_key: str | None
    groq_api_key: str | None
    openrouter_api_key: str | None
    hf_api_key: str | None
    scrub_pii_for_llm: bool
    allowed_origins: tuple[str, ...] = ("http://localhost:5173",)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings(
        model_id=os.getenv("MODEL_ID", "qwen2.5vl:3b"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        tesseract_cmd=os.getenv("TESSERACT_CMD"),
        poppler_path=os.getenv("POPPLER_PATH", r"D:\Work\Poppler\poppler-25.12.0\Library\bin"),
        max_resumes_per_request=_get_int("MAX_RESUMES_PER_REQUEST", 100),
        max_pages_per_resume=_get_int("MAX_PAGES_PER_RESUME", 5),
        max_file_size_mb=_get_int("MAX_FILE_SIZE_MB", 10),
        min_pdf_text_chars=_get_int("MIN_PDF_TEXT_CHARS", 300),
        preload_model_on_startup=_get_bool("PRELOAD_MODEL_ON_STARTUP", True),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        hf_api_key=os.getenv("HF_API_KEY"),
        scrub_pii_for_llm=_get_bool("SCRUB_PII_FOR_LLM", False),
        allowed_origins=tuple(
            o.strip()
            for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
            if o.strip()
        ),
    )

    if not settings.groq_api_key and not settings.gemini_api_key and not settings.openrouter_api_key:
        # Using print for console visibility during startup
        print("Warning: No LLM API Keys configured (Groq/Gemini/OpenRouter). Enrichment features will be disabled.")

    return settings

