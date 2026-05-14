from personal_docs_qa.answerer import answer_question
from personal_docs_qa.indexer import build_index
from personal_docs_qa.models import Chunk


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


def test_citations_come_from_retrieved_chunks() -> None:
    chunk = make_chunk("Bring a passport and printed itinerary for the trip.", "travel.md", 1)
    index = build_index([chunk])

    answer = answer_question(index, "What should I bring for the trip?")

    assert answer.sources[0].chunk == chunk
    assert "[1: travel.md]" in answer.answer
