"""Write the active RAG configuration before holdout evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_config


SECRET_KEY_MARKERS = ("api_key", "apikey", "token", "secret", "password")
REDACTED_VALUE = "[REDACTED]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze active RAG config for evaluation reports.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/frozen_config.json"),
    )
    parser.add_argument("--stage", default="freeze_config")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _redact_secrets(_json_safe(asdict(get_config())))
    payload = {
        "stage": args.stage,
        "notes": args.notes,
        "environment": {
            "DOCUMENT_PROFILE": os.getenv("DOCUMENT_PROFILE") or os.getenv("QUERY_PROFILE") or "general",
            "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "gemini"),
            "LLM_MODEL": os.getenv("LLM_MODEL", "gemini-2.5-flash"),
        },
        "config": config,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote frozen config to {args.output}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if _is_secret_key(key):
                redacted[key] = REDACTED_VALUE if item else None
            else:
                redacted[key] = _redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(marker in normalized for marker in SECRET_KEY_MARKERS)


if __name__ == "__main__":
    main()
