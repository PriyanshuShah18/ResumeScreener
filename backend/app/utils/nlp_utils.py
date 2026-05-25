import re

IMPACT_PATTERNS = [
    re.compile(r"\b(\d+(?:\.\d+)?)\s*%\s*(?:reduction|increase|improvement|decrease|faster|growth)", re.IGNORECASE),
    re.compile(r"\b(?:reduced|improved|increased|decreased|optimized)\b.{0,60}\b\d+", re.IGNORECASE),
    re.compile(r"\b(?:served|handling|processed|supporting)\b.{0,40}\b(\d[\d,]*)\s*(?:users|requests|events|transactions)", re.IGNORECASE),
]
OWNERSHIP_VERBS = {"led", "designed", "architected", "owned", "built", "launched", "shipped", "drove", "founded", "created", "established", "defined"}
CONTRIBUTOR_VERBS = {"worked", "assisted", "helped", "supported", "contributed", "participated"}


def extract_impact_signals(highlights: list[str]) -> dict:
    """Extract quantified impact presence from resume highlights."""
    quantified_count = 0
    for highlight in highlights:
        for pattern in IMPACT_PATTERNS:
            if pattern.search(highlight):
                quantified_count += 1
                break
    return {
        "quantified_bullets": quantified_count,
        "total_bullets": len(highlights),
        "quantification_rate": quantified_count / max(len(highlights), 1),
    }


def ownership_ratio(highlights: list[str]) -> float:
    """Measure ownership vs contributor language in resume highlights."""
    ownership_count = 0
    contributor_count = 0
    for highlight in highlights:
        words = highlight.strip().lower().split()
        if not words:
            continue
        first_word = words[0].rstrip("ed,.:;")
        if first_word in OWNERSHIP_VERBS or any(first_word.startswith(v) for v in OWNERSHIP_VERBS):
            ownership_count += 1
        elif first_word in CONTRIBUTOR_VERBS or any(first_word.startswith(v) for v in CONTRIBUTOR_VERBS):
            contributor_count += 1
    total = ownership_count + contributor_count
    if total == 0:
        return 0.5  # neutral
    return ownership_count / total
