from personal_docs_qa.models import Chunk, SearchResult
from personal_docs_qa.source_format import (
    file_type_label,
    format_score,
    highlight_query_terms_html,
    source_payload,
)


def make_result(metadata: dict | None = None) -> SearchResult:
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        file_name="notes.md",
        path="/tmp/notes.md",
        text="The inspection date is Tuesday. Bring the signed lease.",
        start_char=0,
        end_char=55,
        metadata=metadata or {},
    )
    return SearchResult(
        rank=1,
        score=0.12345,
        score_tfidf=0.23456,
        score_embedding=None,
        retrieval_mode_used="tfidf",
        chunk=chunk,
    )


def test_score_formatting_uses_three_decimals() -> None:
    assert format_score(0.12345) == "0.123"
    assert format_score(1) == "1.000"
    assert format_score(None) == "-"


def test_source_payload_handles_missing_optional_fields() -> None:
    payload = source_payload(make_result(), query="inspection date")

    assert payload["file_type"] == "MD"
    assert payload["page_label"] is None
    assert payload["chunk_id"] == "chunk-1"
    assert payload["score_display"] == "0.123"
    assert payload["score_tfidf_display"] == "0.235"
    assert payload["score_embedding_display"] == "-"
    assert "<mark>inspection</mark>" in payload["excerpt_html"]
    assert "<mark>date</mark>" in payload["excerpt_html"]


def test_query_highlighting_handles_empty_query() -> None:
    assert highlight_query_terms_html("Use <safe> text.", "") == "Use &lt;safe&gt; text."
    assert highlight_query_terms_html("The date is Tuesday.", "What is the date?") == (
        "The <mark>date</mark> is Tuesday."
    )


def test_file_type_and_page_metadata_are_formatted() -> None:
    payload = source_payload(make_result({"extension": ".pdf", "page_number": 3}))

    assert file_type_label(make_result({"extension": ".txt"}).chunk) == "TXT"
    assert payload["file_type"] == "PDF"
    assert payload["page_number"] == 3
    assert payload["page_label"] == "p. 3"
