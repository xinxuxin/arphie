from personal_docs_qa.config import (
    get_answer_model,
    get_default_retrieval_mode,
    get_embedding_dimensions,
    get_embedding_model,
    get_openai_api_key,
    is_openai_available,
    resolve_retrieval_mode,
)


def test_openai_api_key_helpers(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert get_openai_api_key() is None
    assert is_openai_available() is False

    monkeypatch.setenv("OPENAI_API_KEY", " sk-test ")

    assert get_openai_api_key() == "sk-test"
    assert is_openai_available() is True


def test_embedding_defaults_and_env_overrides(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_EMBEDDING_DIMENSIONS", raising=False)

    assert get_embedding_model() == "text-embedding-3-small"
    assert get_embedding_dimensions() == 512

    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "custom-model")
    monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSIONS", "1536")

    assert get_embedding_model() == "custom-model"
    assert get_embedding_dimensions() == 1536


def test_answer_model_default_and_env_override(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_ANSWER_MODEL", raising=False)

    assert get_answer_model() == "gpt-4.1-mini"

    monkeypatch.setenv("OPENAI_ANSWER_MODEL", "custom-answer-model")

    assert get_answer_model() == "custom-answer-model"


def test_default_retrieval_mode_and_auto_fallback(monkeypatch) -> None:
    monkeypatch.delenv("DOCQA_RETRIEVAL_MODE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert get_default_retrieval_mode() == "auto"
    mode, warnings = resolve_retrieval_mode("auto")

    assert mode == "tfidf"
    assert warnings


def test_auto_uses_hybrid_when_openai_key_exists(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    mode, warnings = resolve_retrieval_mode("auto")

    assert mode == "hybrid"
    assert warnings == []
