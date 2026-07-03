from evaluation.freeze_config import REDACTED_VALUE, _redact_secrets


def test_redact_secrets_removes_api_keys_recursively() -> None:
    config = {
        "gemini_api_key": "secret-gemini",
        "nested": {
            "openai_api_key": "secret-openai",
            "normal_value": "visible",
        },
        "empty_token": None,
    }

    assert _redact_secrets(config) == {
        "gemini_api_key": REDACTED_VALUE,
        "nested": {
            "openai_api_key": REDACTED_VALUE,
            "normal_value": "visible",
        },
        "empty_token": None,
    }
