"""Environment-backed configuration helpers."""

import os


VALID_RETRIEVAL_MODES = {"tfidf", "embedding", "hybrid", "auto"}
VALID_ANSWER_MODES = {"local", "openai", "auto"}
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 512
DEFAULT_RETRIEVAL_MODE = "auto"
DEFAULT_ANSWER_MODE = "auto"


def get_openai_api_key() -> str | None:
    """Return the OpenAI API key if one is configured."""
    value = os.getenv("OPENAI_API_KEY")
    return value.strip() if value and value.strip() else None


def is_openai_available() -> bool:
    """Return whether OpenAI-backed features can be used."""
    return get_openai_api_key() is not None


def get_embedding_model() -> str:
    """Return the configured OpenAI embedding model."""
    return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL


def get_embedding_dimensions() -> int:
    """Return the configured embedding dimensions.

    The default is 512 for `text-embedding-3-small` because this app targets a
    small take-home dataset and benefits from a smaller persisted local index.
    """
    raw_value = os.getenv("OPENAI_EMBEDDING_DIMENSIONS")
    if not raw_value:
        return DEFAULT_EMBEDDING_DIMENSIONS
    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_EMBEDDING_DIMENSIONS
    return value if value > 0 else DEFAULT_EMBEDDING_DIMENSIONS


def get_default_retrieval_mode() -> str:
    """Return the configured retrieval mode."""
    mode = os.getenv("DOCQA_RETRIEVAL_MODE", DEFAULT_RETRIEVAL_MODE).strip().lower()
    return mode if mode in VALID_RETRIEVAL_MODES else DEFAULT_RETRIEVAL_MODE


def get_default_answer_mode() -> str:
    """Return the configured answer mode."""
    mode = os.getenv("DOCQA_ANSWER_MODE", DEFAULT_ANSWER_MODE).strip().lower()
    return mode if mode in VALID_ANSWER_MODES else DEFAULT_ANSWER_MODE


def resolve_retrieval_mode(mode: str | None = None) -> tuple[str, list[str]]:
    """Resolve `auto` into a concrete retrieval mode and collect warnings."""
    requested = (mode or get_default_retrieval_mode()).strip().lower()
    warnings: list[str] = []

    if requested not in VALID_RETRIEVAL_MODES:
        warnings.append(f"Unknown retrieval mode '{requested}', falling back to tfidf.")
        return "tfidf", warnings

    if requested == "auto":
        if is_openai_available():
            return "hybrid", warnings
        warnings.append("OPENAI_API_KEY is not set; auto retrieval mode is using tfidf.")
        return "tfidf", warnings

    return requested, warnings
