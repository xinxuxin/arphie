from types import SimpleNamespace

import pytest

from personal_docs_qa import openai_embeddings
from personal_docs_qa.openai_embeddings import OpenAIEmbeddingError, embed_query, embed_texts


class FakeEmbeddingsResource:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def create(self, *, model: str, input: list[str], dimensions: int):
        self.inputs.append(input)
        data = [
            SimpleNamespace(index=index, embedding=[float(len(text)), float(index), float(dimensions)])
            for index, text in enumerate(input)
        ]
        return SimpleNamespace(data=data)


class FakeOpenAI:
    resource = FakeEmbeddingsResource()

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.embeddings = self.resource


class FailingEmbeddingsResource:
    def create(self, *, model: str, input: list[str], dimensions: int):
        raise RuntimeError("network unavailable")


class FailingOpenAI:
    def __init__(self, api_key: str) -> None:
        self.embeddings = FailingEmbeddingsResource()


def test_embed_texts_batches_and_preserves_order(monkeypatch) -> None:
    FakeOpenAI.resource = FakeEmbeddingsResource()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai_embeddings, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(openai_embeddings, "DEFAULT_BATCH_SIZE", 2)

    embeddings = embed_texts(["alpha", "beta beta", "gamma"], dimensions=512)

    assert FakeOpenAI.resource.inputs == [["alpha", "beta beta"], ["gamma"]]
    assert embeddings == [[5.0, 0.0, 512.0], [9.0, 1.0, 512.0], [5.0, 0.0, 512.0]]


def test_embed_query_returns_single_embedding(monkeypatch) -> None:
    FakeOpenAI.resource = FakeEmbeddingsResource()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai_embeddings, "OpenAI", FakeOpenAI)

    assert embed_query("hello", dimensions=512) == [5.0, 0.0, 512.0]


def test_missing_api_key_is_friendly(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(OpenAIEmbeddingError, match="OPENAI_API_KEY"):
        embed_texts(["hello"])


def test_empty_input_returns_empty_list(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert embed_texts([]) == []


def test_empty_string_is_not_sent(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai_embeddings, "OpenAI", FakeOpenAI)

    with pytest.raises(OpenAIEmbeddingError, match="empty text"):
        embed_texts(["   "])


def test_api_failure_is_wrapped(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai_embeddings, "OpenAI", FailingOpenAI)

    with pytest.raises(OpenAIEmbeddingError, match="Embedding request failed"):
        embed_texts(["hello"])
