import pytest
import numpy as np

from personal_docs_qa.indexer import build_index
from personal_docs_qa.models import Chunk
from personal_docs_qa import retriever
from personal_docs_qa.retriever import search


def make_chunk(text: str, index: int) -> Chunk:
    return Chunk(
        id=f"chunk-{index}",
        document_id=f"doc-{index}",
        file_name=f"doc-{index}.txt",
        path=f"/tmp/doc-{index}.txt",
        text=text,
        start_char=0,
        end_char=len(text),
        metadata={"source_path": f"/tmp/doc-{index}.txt", "extension": ".txt"},
    )


def test_relevant_chunk_ranks_above_irrelevant_chunk() -> None:
    chunks = [
        make_chunk("vacation itinerary hotel beach flights", 1),
        make_chunk("quarterly tax receipts and deductions", 2),
    ]
    index = build_index(chunks)

    results = search(index, "beach hotel", top_k=2)

    assert results[0].chunk == chunks[0]
    assert results[0].score > results[1].score


def test_top_k_respected() -> None:
    index = build_index([make_chunk("alpha beta", 1), make_chunk("alpha gamma", 2)])

    results = search(index, "alpha", top_k=1)

    assert len(results) == 1
    assert results[0].rank == 1


def test_empty_query_handled() -> None:
    index = build_index([make_chunk("some text", 1)])

    with pytest.raises(ValueError, match="cannot be empty"):
        search(index, "   ")


def test_unrelated_query_gives_low_scores() -> None:
    index = build_index([make_chunk("apple banana", 1), make_chunk("car train", 2)])

    results = search(index, "quantum nebula", top_k=2)

    assert len(results) == 2
    assert all(result.score == 0 for result in results)
    assert {result.chunk.id for result in results}.issubset({"chunk-1", "chunk-2"})


def test_embedding_search_uses_existing_chunks(monkeypatch) -> None:
    chunks = [make_chunk("alpha", 1), make_chunk("beta", 2)]
    index = build_index(chunks)
    index.retrieval_mode = "embedding"
    index.embedding_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    index.embedding_model = "text-embedding-3-small"
    index.embedding_dimensions = 512
    monkeypatch.setattr(retriever, "embed_query", lambda *args, **kwargs: [0.0, 1.0])

    results = search(index, "semantic beta", top_k=1)

    assert results[0].chunk == chunks[1]
