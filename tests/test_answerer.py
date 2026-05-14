from personal_docs_qa.answerer import _local_answer_from_results, answer_question, answer_question_with_metadata
from personal_docs_qa.indexer import build_index
from personal_docs_qa.models import Chunk, SearchResult


def make_chunk(text: str, file_name: str, index: int) -> Chunk:
    return Chunk(
        id=f"chunk-{index}",
        document_id=f"doc-{index}",
        file_name=file_name,
        path=f"/tmp/{file_name}",
        text=text,
        start_char=0,
        end_char=len(text),
        metadata={"source_path": f"/tmp/{file_name}", "extension": ".txt"},
    )


def test_answer_includes_source_file_names() -> None:
    index = build_index([make_chunk("The grocery budget is 120 dollars per week.", "budget.txt", 1)])

    answer = answer_question(index, "What is the grocery budget?")

    assert "budget.txt" in answer.answer
    assert answer.sources[0].chunk.file_name == "budget.txt"


def test_weak_question_produces_cautious_answer() -> None:
    index = build_index([make_chunk("The car insurance renewal is due in April.", "insurance.txt", 1)])

    answer = answer_question(index, "What telescope should I buy?")

    assert answer.confidence == "low"
    assert "I found related passages, but they may not fully answer the question." in answer.answer
    assert answer.warnings


def test_answer_does_not_crash_with_no_index() -> None:
    answer = answer_question(None, "Anything?")

    assert answer.confidence == "low"
    assert "No index is available" in answer.answer
    assert answer.sources == []


def test_unrelated_question_does_not_invent_answer() -> None:
    index = build_index([make_chunk("The lease inspection is on Tuesday.", "lease.txt", 1)])

    answer = answer_question(index, "Which telescope should I buy?")

    assert answer.confidence == "low"
    assert answer.answer.startswith("I found related passages")
    assert "telescope" not in answer.answer.lower()


def test_high_score_without_query_overlap_stays_low_confidence() -> None:
    result = SearchResult(
        rank=1,
        score=1.0,
        chunk=make_chunk("The project risks include retrieval tradeoffs.", "project.md", 1),
        score_tfidf=0.0,
        score_embedding=0.85,
        retrieval_mode_used="hybrid",
    )

    answer = _local_answer_from_results("What does this folder say about travel plans to Japan?", [result])

    assert answer.confidence == "low"


def test_citations_come_from_retrieved_chunks() -> None:
    chunk = make_chunk("Bring a passport and printed itinerary for the trip.", "travel.md", 1)
    index = build_index([chunk])

    answer = answer_question(index, "What should I bring for the trip?")

    assert answer.sources[0].chunk == chunk
    assert "[1: travel.md]" in answer.answer


def test_answer_metadata_reports_modes_and_fallback() -> None:
    index = build_index([make_chunk("The inspection date is Tuesday.", "lease.md", 1)])

    result = answer_question_with_metadata(
        index,
        "What is the inspection date?",
        retrieval_mode="auto",
        answer_mode="openai",
    )

    assert result.retrieval_mode_requested == "auto"
    assert result.retrieval_mode_used == "tfidf"
    assert result.retrieval_fallback_used is True
    assert result.answer_mode_requested == "openai"
    assert result.answer_mode_used == "local"
    assert result.answer_fallback_used is True
    assert result.warnings
