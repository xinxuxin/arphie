"""Small OpenAI embeddings client wrapper."""

from __future__ import annotations

from typing import Sequence

from openai import OpenAI, OpenAIError

from personal_docs_qa.config import (
    get_embedding_dimensions,
    get_embedding_model,
    get_openai_api_key,
)


DEFAULT_BATCH_SIZE = 64


class OpenAIEmbeddingError(RuntimeError):
    """Friendly error raised for OpenAI embedding failures."""


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _batched(items: Sequence[str], batch_size: int) -> list[list[str]]:
    return [list(items[index : index + batch_size]) for index in range(0, len(items), batch_size)]


def embed_texts(
    texts: list[str],
    model: str | None = None,
    dimensions: int | None = None,
) -> list[list[float]]:
    """Embed a list of texts using the OpenAI embeddings API."""
    if not texts:
        return []

    cleaned = [_clean_text(text) for text in texts]
    if any(not text for text in cleaned):
        raise OpenAIEmbeddingError("Cannot embed empty text.")

    api_key = get_openai_api_key()
    if not api_key:
        raise OpenAIEmbeddingError("OPENAI_API_KEY is required for embedding retrieval mode.")

    selected_model = model or get_embedding_model()
    selected_dimensions = dimensions or get_embedding_dimensions()
    client = OpenAI(api_key=api_key)
    embeddings: list[list[float]] = []

    try:
        for batch in _batched(cleaned, DEFAULT_BATCH_SIZE):
            response = client.embeddings.create(
                model=selected_model,
                input=batch,
                dimensions=selected_dimensions,
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            embeddings.extend([list(item.embedding) for item in ordered])
    except OpenAIError as exc:
        raise OpenAIEmbeddingError(
            "OpenAI embedding request failed. Check that OPENAI_API_KEY is valid and has access to embeddings."
        ) from exc
    except Exception as exc:
        raise OpenAIEmbeddingError(f"Embedding request failed: {exc}") from exc

    if len(embeddings) != len(texts):
        raise OpenAIEmbeddingError("Embedding response count did not match input count.")

    return embeddings


def embed_query(
    query: str,
    model: str | None = None,
    dimensions: int | None = None,
) -> list[float]:
    """Embed a single search query."""
    embeddings = embed_texts([query], model=model, dimensions=dimensions)
    return embeddings[0]
