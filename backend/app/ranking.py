from __future__ import annotations

from app.schemas import ScreeningResult


def _required_match_ratio(result: ScreeningResult) -> float:
    details = result.score.semantic_match_details or {}
    ratio = details.get("required_match_ratio")
    if isinstance(ratio, (int, float)):
        return max(0.0, min(float(ratio), 1.0))

    matched = result.score.matched_skills or []
    missing = result.score.missing_skills or []
    total_required = len(matched) + len(missing)
    if total_required == 0:
        return 1.0
    return len(matched) / total_required


def rank_results(results: list[ScreeningResult]) -> list[ScreeningResult]:
    return sorted(
        results,
        key=lambda item: (
            -item.score.total_score,
            -_required_match_ratio(item),
            -item.score.confidence_score,
            item.score.risk_score,
        ),
    )
