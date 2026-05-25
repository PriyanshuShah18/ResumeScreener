import re
from typing import Any

WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(value: Any) -> str:
    if value is None:
        return ""
    return WHITESPACE_RE.sub(" ", str(value)).strip()


def normalize_skill(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\+\#\./ ]", " ", normalize_whitespace(value).lower())
    return WHITESPACE_RE.sub(" ", cleaned).strip(" .,:;")


def split_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[\n,;/|]+", str(value))

    seen: set[str] = set()
    result: list[str] = []
    for item in raw_items:
        normalized = normalize_skill(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
