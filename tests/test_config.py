from app.config import _env_optional_string, get_config


def test_default_config_values() -> None:
    config = get_config()

    assert config.llm_provider
    assert config.llm_model
    assert config.embedding_model
    assert config.embedding_batch_size > 0
    assert config.indexing_batch_size > 0
    assert config.parent_context_max_chars > 0
    assert config.top_k > 0
    assert 0 <= config.similarity_threshold <= 1
    assert 0 <= config.hybrid_alpha <= 1
    assert isinstance(config.use_hybrid_search, bool)
    assert isinstance(config.use_mmr, bool)
    assert isinstance(config.use_multi_query, bool)
    assert config.multi_query_count > 0
    assert config.document_profile


def test_optional_env_string_ignores_placeholders(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_MODEL", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "your_openai_api_key_here")

    assert _env_optional_string("RERANKER_MODEL") is None
    assert _env_optional_string("OPENAI_API_KEY") is None
