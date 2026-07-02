"""Lightweight processing diagnostics for long document ingestion."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


LOG_PATH = Path(os.getenv("PROCESSING_DEBUG_LOG", "logs/processing_debug.log"))


def log_event(event: str, **fields: Any) -> None:
    """Append a JSONL debug event without interrupting the app."""

    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        "pid": os.getpid(),
        "memory_mb": _memory_mb(),
        **fields,
    }

    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return


def text_stats(texts: list[str]) -> dict[str, int | float]:
    lengths = [len(text) for text in texts]
    if not lengths:
        return {"count": 0, "min_chars": 0, "max_chars": 0, "avg_chars": 0.0, "total_chars": 0}

    return {
        "count": len(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "avg_chars": round(sum(lengths) / len(lengths), 2),
        "total_chars": sum(lengths),
    }


def _memory_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return None
