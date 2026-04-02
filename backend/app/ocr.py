from __future__ import annotations

import re
from typing import Iterable

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:  # pragma: no cover
    pytesseract = None
from app.config import get_settings
from app.constants import CONTACT_HINTS, SECTION_PATTERNS


class OCRUnavailableError(RuntimeError):
    pass


def _configure_tesseract() -> None:
    settings = get_settings()
    if pytesseract and settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


def extract_with_boxes(image) -> str:
    if not pytesseract:
        raise OCRUnavailableError("pytesseract is not installed")

    _configure_tesseract()
    not_found_error = getattr(pytesseract, "TesseractNotFoundError", None)
    if not not_found_error and hasattr(pytesseract, "pytesseract"):
        not_found_error = getattr(pytesseract.pytesseract, "TesseractNotFoundError", None)

    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 11")
    except FileNotFoundError as exc:
        raise OCRUnavailableError("Tesseract binary was not found. Set TESSERACT_CMD or add Tesseract to PATH.") from exc
    except Exception as exc:
        if not_found_error and isinstance(exc, not_found_error):
            raise OCRUnavailableError("Tesseract binary was not found. Set TESSERACT_CMD or add Tesseract to PATH.") from exc
        raise
    words: list[dict[str, int | str]] = []

    for idx, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        try:
            confidence = float(data["conf"][idx])
        except (TypeError, ValueError):
            continue

        if not text or confidence < 15:
            continue

        words.append(
            {
                "text": text,
                "left": int(data["left"][idx]),
                "top": int(data["top"][idx]),
                "width": int(data["width"][idx]),
                "height": int(data["height"][idx]),
            }
        )

    if not words:
        return ""

    words.sort(key=lambda item: (item["top"], item["left"]))
    tolerance = max(12, int(sum(item["height"] for item in words) / max(len(words), 1)))

    lines: list[list[dict[str, int | str]]] = []
    current_line: list[dict[str, int | str]] = []
    current_top = None

    for word in words:
        if current_top is None or abs(int(word["top"]) - current_top) <= tolerance:
            current_line.append(word)
            current_top = int(word["top"]) if current_top is None else min(current_top, int(word["top"]))
            continue

        lines.append(current_line)
        current_line = [word]
        current_top = int(word["top"])

    if current_line:
        lines.append(current_line)

    return "\n".join(join_words(line) for line in lines if join_words(line))


def join_words(word_list: Iterable[dict[str, int | str]]) -> str:
    ordered = sorted(word_list, key=lambda item: int(item["left"]))
    if not ordered:
        return ""

    parts: list[str] = []
    previous_right = None
    for word in ordered:
        if previous_right is not None:
            gap = int(word["left"]) - previous_right
            parts.append("    " if gap > 70 else " ")
        parts.append(str(word["text"]))
        previous_right = int(word["left"]) + int(word["width"])
    return "".join(parts).strip()


def clean_ocr_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)

    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"^[\-\|\]\[()'`]+", "", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def annotate_resume_sections(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    annotated: list[str] = []
    current_marker = ""
    header_seen = False

    if looks_like_contact_block(lines[:4]):
        annotated.append("[CONTACT_INFO]")
        current_marker = "[CONTACT_INFO]"

    for line in lines:
        marker = match_section_marker(line)
        if marker:
            if marker != current_marker:
                annotated.append(marker)
                current_marker = marker
            header_seen = True
            continue

        if looks_like_link_line(line) and current_marker != "[LINKS]":
            annotated.append("[LINKS]")
            current_marker = "[LINKS]"
        elif not header_seen and current_marker != "[CONTACT_INFO]" and looks_like_contact_line(line):
            annotated.append("[CONTACT_INFO]")
            current_marker = "[CONTACT_INFO]"

        annotated.append(line)

    return "\n".join(annotated)


def match_section_marker(line: str) -> str:
    normalized = re.sub(r"[^a-z ]", "", line.lower()).strip()
    for marker, aliases in SECTION_PATTERNS:
        if normalized in aliases:
            return marker
    return ""


def looks_like_contact_block(lines: list[str]) -> bool:
    return any(looks_like_contact_line(line) for line in lines)


def looks_like_contact_line(line: str) -> bool:
    lowered = line.lower()
    return any(hint in lowered for hint in CONTACT_HINTS) or bool(re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", line))


def looks_like_link_line(line: str) -> bool:
    lowered = line.lower()
    return any(token in lowered for token in ("linkedin", "github", "portfolio", "www.", "http"))


def get_structured_ocr(image) -> str:
    text = extract_with_boxes(image)
    return annotate_resume_sections(clean_ocr_text(text))
