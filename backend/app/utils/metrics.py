"""Simple in-memory metrics for operational visibility."""

import threading
import time

_lock = threading.Lock()
_data = {"processed": 0, "failed": 0, "total_seconds": 0.0, "requests": 0}


def record_batch(processed: int, failed: int, elapsed: float) -> None:
    with _lock:
        _data["processed"] += processed
        _data["failed"] += failed
        _data["total_seconds"] += elapsed
        _data["requests"] += 1


def get_metrics() -> dict:
    with _lock:
        avg = _data["total_seconds"] / max(_data["requests"], 1)
        return {**_data, "avg_seconds_per_request": round(avg, 2)}
