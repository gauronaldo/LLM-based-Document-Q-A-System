from app.config import get_config


def test_default_config_values() -> None:
    config = get_config()

    assert config.llm_provider
    assert config.llm_model
    assert config.embedding_model
    assert config.top_k > 0
    assert 0 <= config.similarity_threshold <= 1

