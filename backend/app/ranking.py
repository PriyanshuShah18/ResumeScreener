from __future__ import annotations

from app.schemas import ScreeningResult


def rank_results(results: list[ScreeningResult]) -> list[ScreeningResult]:
    return sorted(
        results,
        key=lambda item: (
            -item.score.total_score,
            -item.score.confidence_score,
            item.score.risk_score,
        ),
    )
