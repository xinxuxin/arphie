"""Lightweight eval harness for the sample document QA flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from personal_docs_qa.answerer import answer_question_with_metadata
from personal_docs_qa.config import is_openai_available
from personal_docs_qa.indexer import LocalIndex, index_folder_with_warnings


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "eval" / "eval_questions.json"
SAMPLE_DOCS_PATH = ROOT / "sample_docs"
EVAL_INDEX_DIR = ROOT / ".docqa"


def load_questions() -> list[dict[str, Any]]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def contains_all_terms(answer_text: str, excerpts: list[str], terms: list[str]) -> bool:
    haystack = " ".join([answer_text, *excerpts]).lower()
    return all(term.lower() in haystack for term in terms)


def build_eval_indexes(console: Console) -> dict[str, LocalIndex]:
    indexes: dict[str, LocalIndex] = {}
    tfidf_result = index_folder_with_warnings(
        SAMPLE_DOCS_PATH,
        EVAL_INDEX_DIR / "eval_tfidf_index.joblib",
        retrieval_mode="tfidf",
    )
    indexes["tfidf"] = tfidf_result.index

    if tfidf_result.warnings:
        console.print("[yellow]Warnings while building TF-IDF eval index:[/yellow]")
        for warning in tfidf_result.warnings:
            console.print(f"[yellow]- {warning}[/yellow]")

    if not is_openai_available():
        console.print("[yellow]OpenAI not configured; skipped embedding/hybrid eval.[/yellow]")
        return indexes

    try:
        hybrid_result = index_folder_with_warnings(
            SAMPLE_DOCS_PATH,
            EVAL_INDEX_DIR / "eval_hybrid_index.joblib",
            retrieval_mode="hybrid",
        )
    except ValueError as exc:
        console.print(f"[yellow]Hybrid eval skipped: {exc}[/yellow]")
        return indexes

    indexes["hybrid"] = hybrid_result.index
    if hybrid_result.warnings:
        console.print("[yellow]Warnings while building hybrid eval index:[/yellow]")
        for warning in hybrid_result.warnings:
            console.print(f"[yellow]- {warning}[/yellow]")
    return indexes


def main() -> None:
    console = Console()
    questions = load_questions()
    indexes = build_eval_indexes(console)

    table = Table(title="Sample Docs QA Eval")
    table.add_column("Question")
    table.add_column("Expected Source")
    table.add_column("Mode")
    table.add_column("Top Source")
    table.add_column("Expected Source Found")
    table.add_column("Expected Terms Found")
    table.add_column("Confidence")
    table.add_column("Retrieval Used")
    table.add_column("Answer Used")
    table.add_column("Fallback Used")

    for item in questions:
        for retrieval_mode, index in indexes.items():
            for answer_mode in ["local", "auto"]:
                result = answer_question_with_metadata(
                    index,
                    item["question"],
                    top_k=5,
                    retrieval_mode=retrieval_mode,
                    answer_mode=answer_mode,
                )
                answer = result.answer
                source_names = [source.chunk.file_name for source in answer.sources]
                excerpts = [source.chunk.text for source in answer.sources]
                expected_source = item["expected_source"]
                expected_source_found = expected_source in source_names
                expected_terms_found = contains_all_terms(answer.answer, excerpts, item["expected_terms"])
                top_source = source_names[0] if source_names else "none"
                fallback_used = result.retrieval_fallback_used or result.answer_fallback_used

                table.add_row(
                    item["question"],
                    expected_source,
                    f"{retrieval_mode}/{answer_mode}",
                    top_source,
                    "yes" if expected_source_found else "no",
                    "yes" if expected_terms_found else "no",
                    answer.confidence,
                    result.retrieval_mode_used,
                    result.answer_mode_used,
                    "yes" if fallback_used else "no",
                )

    console.print(table)


if __name__ == "__main__":
    main()
