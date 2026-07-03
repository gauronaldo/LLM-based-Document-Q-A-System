"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        parsed = int(value.strip())
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _env_optional_string(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    placeholder_values = {
        "none",
        "null",
        "false",
        "true",
        "your_openai_api_key_here",
        "your_gemini_api_key_here",
    }
    if cleaned.lower() in placeholder_values:
        return None
    return cleaned


@dataclass(frozen=True)
class AppConfig:
    """Runtime settings for the document QA application."""

    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    gemini_api_key: str | None = _env_optional_string("GEMINI_API_KEY")
    openai_api_key: str | None = _env_optional_string("OPENAI_API_KEY")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    embedding_batch_size: int = _env_int("EMBEDDING_BATCH_SIZE", 32)
    indexing_batch_size: int = _env_int("INDEXING_BATCH_SIZE", 128)
    parent_context_max_chars: int = _env_int("PARENT_CONTEXT_MAX_CHARS", 3000)
    chroma_persist_dir: Path = Path(os.getenv("CHROMA_PERSIST_DIR", "vector_db"))
    chroma_collection_name: str = os.getenv(
        "CHROMA_COLLECTION_NAME",
        "document_qa_vietnamese",
    )
    top_k: int = int(os.getenv("TOP_K", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    hybrid_alpha: float = float(os.getenv("HYBRID_ALPHA", "0.7"))
    use_hybrid_search: bool = _env_bool("USE_HYBRID_SEARCH", True)
    use_mmr: bool = _env_bool("USE_MMR", True)
    use_multi_query: bool = _env_bool("USE_MULTI_QUERY", False)
    multi_query_count: int = _env_int("MULTI_QUERY_COUNT", 4)
    reranker_model: str | None = _env_optional_string("RERANKER_MODEL")
    document_profile: str = os.getenv(
        "DOCUMENT_PROFILE",
        os.getenv("QUERY_PROFILE", "general"),
    ).strip().lower()
    raw_data_dir: Path = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
    processed_data_dir: Path = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))


def get_config() -> AppConfig:
    """Return application configuration."""

    return AppConfig()
