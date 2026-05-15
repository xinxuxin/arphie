from types import SimpleNamespace

import pytest

from personal_docs_qa import openai_answerer
from personal_docs_qa.models import Chunk, SearchResult
from personal_docs_qa.openai_answerer import OpenAIAnswerError, synthesize_answer


class FakeResponsesResource:
    def __init__(self) -> None:
        self.calls = []

    def create(self, *, model: str, input: str, max_output_tokens: int):
        self.calls.append(
            {
                "model": model,
                "input": input,
                "max_output_tokens": max_output_tokens,
            }
        )
        return SimpleNamespace(output_text="You worked on Doubao fine-tuning workflows. [1: resume.pdf]")


class FakeOpenAI:
    responses = FakeResponsesResource()

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.responses = self.__class__.responses


class FailingResponsesResource:
    def create(self, *, model: str, input: str, max_output_tokens: int):
        raise RuntimeError("boom")


class FailingOpenAI:
    def __init__(self, api_key: str) -> None:
        self.responses = FailingResponsesResource()


def make_result() -> SearchResult:
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        file_name="resume.pdf",
        path="/tmp/resume.pdf",
        text="ByteDance internship: built LoRA fine-tuning workflows for Doubao.",
        start_char=0,
        end_char=66,
        metadata={"extension": ".pdf", "page_number": 1},
    )
    return SearchResult(rank=1, score=0.5, chunk=chunk, retrieval_mode_used="hybrid")


def test_synthesize_answer_uses_responses_api(monkeypatch) -> None:
    FakeOpenAI.responses = FakeResponsesResource()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai_answerer, "OpenAI", FakeOpenAI)

    answer = synthesize_answer("What did I do at ByteDance?", [make_result()], confidence="high")

    assert "Doubao" in answer
    assert FakeOpenAI.responses.calls
    assert "ByteDance internship" in FakeOpenAI.responses.calls[0]["input"]


def test_synthesize_answer_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(OpenAIAnswerError, match="OPENAI_API_KEY"):
        synthesize_answer("Question?", [make_result()], confidence="low")


def test_synthesize_answer_wraps_failures_without_secret(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    monkeypatch.setattr(openai_answerer, "OpenAI", FailingOpenAI)

    with pytest.raises(OpenAIAnswerError) as exc_info:
        synthesize_answer("Question?", [make_result()], confidence="low")

    assert "sk-test-secret" not in str(exc_info.value)
    assert "OpenAI answer synthesis failed" in str(exc_info.value)
