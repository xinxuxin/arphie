"""Lightweight eval harness for the sample document QA flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from personal_docs_qa.answerer import answer_question
from personal_docs_qa.indexer import index_folder_with_warnings


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "eval" / "eval_questions.json"
SAMPLE_DOCS_PATH = ROOT / "sample_docs"
EVAL_INDEX_PATH = ROOT / ".docqa" / "eval_index.joblib"


def load_questions() -> list[dict[str, Any]]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def contains_all_terms(answer_text: str, excerpts: list[str], terms: list[str]) -> bool:
    haystack = " ".join([answer_text, *excerpts]).lower()
    return all(term.lower() in haystack for term in terms)


def main() -> None:
    console = Console()
    questions = load_questions()
    result = index_folder_with_warnings(SAMPLE_DOCS_PATH, EVAL_INDEX_PATH)

    if result.warnings:
        console.print("[yellow]Warnings while indexing sample docs:[/yellow]")
        for warning in result.warnings:
            console.print(f"[yellow]- {warning}[/yellow]")

    table = Table(title="Sample Docs QA Eval")
    table.add_column("Question")
    table.add_column("Expected Source Found")
    table.add_column("Expected Terms Found")
    table.add_column("Confidence")
    table.add_column("Top Source")

    for item in questions:
        answer = answer_question(result.index, item["question"], top_k=5)
        source_names = [source.chunk.file_name for source in answer.sources]
        excerpts = [source.chunk.text for source in answer.sources]
        expected_source_found = item["expected_source"] in source_names
        expected_terms_found = contains_all_terms(answer.answer, excerpts, item["expected_terms"])
        top_source = source_names[0] if source_names else "none"

        table.add_row(
            item["question"],
            "yes" if expected_source_found else "no",
            "yes" if expected_terms_found else "no",
            answer.confidence,
            top_source,
        )

    console.print(table)


if __name__ == "__main__":
    main()
