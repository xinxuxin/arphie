"""Command-line interface for Personal Docs QA."""

import re
from pathlib import Path

import typer
import uvicorn
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from personal_docs_qa.answerer import answer_question, answer_question_with_metadata
from personal_docs_qa.config import (
    VALID_ANSWER_MODES,
    VALID_RETRIEVAL_MODES,
    get_default_answer_mode,
    get_default_retrieval_mode,
)
from personal_docs_qa.indexer import DEFAULT_INDEX_PATH, index_folder_with_warnings, load_index
from personal_docs_qa.retriever import search
from personal_docs_qa.source_format import format_score, query_terms, source_payload


app = typer.Typer(
    help="Ask questions over a local folder of personal documents.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """Ask questions over a local folder of personal documents."""


def _load_default_index():
    try:
        return load_index(DEFAULT_INDEX_PATH)
    except FileNotFoundError:
        console.print("[yellow]No index found. Run `docqa ingest <folder>` first.[/yellow]")
        raise typer.Exit(code=1)


def _highlight_excerpt(excerpt: str, query: str, retrieval_mode: str) -> Text:
    text = Text(excerpt)
    if retrieval_mode not in {"tfidf", "hybrid"}:
        return text
    for term in query_terms(query):
        for match in re.finditer(re.escape(term), excerpt, flags=re.IGNORECASE):
            text.stylize("bold yellow", match.start(), match.end())
    return text


def _print_sources(results, query: str = "") -> None:
    results = [result for result in results if result.score > 0]
    if not results:
        console.print("[yellow]No positive-score sources found.[/yellow]")
        return

    console.print("[bold]Sources[/bold]")
    for result in results:
        source = source_payload(result, query=query)
        scores = (
            f"final {format_score(source['score'])} | "
            f"tfidf {source['score_tfidf_display']} | "
            f"embedding {source['score_embedding_display']}"
        )
        page = f" | {source['page_label']}" if source["page_label"] else ""
        meta = Text()
        meta.append(f"{source['file_type']} ", style="bold")
        meta.append(f"{source['retrieval_mode_used']} | {scores}{page}\n", style="cyan")
        meta.append(str(source["chunk_id"]), style="dim")

        console.print(
            Panel(
                Group(meta, _highlight_excerpt(source["excerpt"], query, source["retrieval_mode_used"])),
                title=f"#{source['rank']} {source['file_name']}",
                expand=False,
            )
        )


@app.command()
def ingest(
    folder: Path = typer.Argument(..., help="Folder of .txt, .md, and .pdf documents."),
    index_path: Path = typer.Option(DEFAULT_INDEX_PATH, help="Where to save the local index."),
    retrieval_mode: str = typer.Option(
        get_default_retrieval_mode(),
        help="Retrieval mode: tfidf, embedding, hybrid, or auto.",
    ),
) -> None:
    """Load documents, chunk them, and build a local searchable index."""
    if retrieval_mode not in VALID_RETRIEVAL_MODES:
        console.print(f"[red]Invalid retrieval mode: {retrieval_mode}[/red]")
        raise typer.Exit(code=1)
    try:
        result = index_folder_with_warnings(folder, index_path=index_path, retrieval_mode=retrieval_mode)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    index = result.index
    console.print(Panel.fit("Index built successfully", style="green"))
    console.print(f"Retrieval mode requested: [bold]{retrieval_mode}[/bold]")
    console.print(f"Retrieval mode built: [bold]{index.retrieval_mode_built}[/bold]")
    console.print(f"Embeddings created: [bold]{'yes' if index.chunk_embeddings is not None else 'no'}[/bold]")
    if index.embedding_model:
        console.print(f"Embedding model: [bold]{index.embedding_model}[/bold]")
    console.print(f"Documents loaded: [bold]{index.document_count}[/bold]")
    console.print(f"Chunks indexed: [bold]{index.chunk_count}[/bold]")
    console.print(f"Index path: [bold]{index_path}[/bold]")
    if result.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"[yellow]- {warning}[/yellow]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask over the local index."),
    top_k: int = typer.Option(5, help="Number of chunks to retrieve."),
    retrieval_mode: str = typer.Option(
        get_default_retrieval_mode(),
        help="Retrieval mode: auto, tfidf, embedding, or hybrid.",
    ),
    answer_mode: str = typer.Option(
        get_default_answer_mode(),
        help="Answer mode: auto, openai, or local.",
    ),
) -> None:
    """Ask a question against the saved local index."""
    if retrieval_mode not in VALID_RETRIEVAL_MODES:
        console.print(f"[red]Invalid retrieval mode: {retrieval_mode}[/red]")
        raise typer.Exit(code=1)
    if answer_mode not in VALID_ANSWER_MODES:
        console.print(f"[red]Invalid answer mode: {answer_mode}[/red]")
        raise typer.Exit(code=1)
    index = _load_default_index()
    result = answer_question_with_metadata(
        index,
        question,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        answer_mode=answer_mode,
    )
    answer = result.answer

    console.print(Panel(answer.answer, title=f"Answer ({answer.confidence} confidence)", expand=False))
    console.print(f"Retrieval mode requested: [bold]{result.retrieval_mode_requested}[/bold]")
    console.print(f"Retrieval mode used: [bold]{result.retrieval_mode_used}[/bold]")
    console.print(f"Retrieval fallback used: [bold]{'yes' if result.retrieval_fallback_used else 'no'}[/bold]")
    console.print(f"Answer mode requested: [bold]{result.answer_mode_requested}[/bold]")
    console.print(f"Answer mode used: [bold]{result.answer_mode_used}[/bold]")
    console.print(f"Answer fallback used: [bold]{'yes' if result.answer_fallback_used else 'no'}[/bold]")
    console.print(f"Confidence: [bold]{answer.confidence}[/bold]")
    if answer.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in answer.warnings:
            console.print(f"[yellow]- {warning}[/yellow]")
    if answer.sources:
        _print_sources(answer.sources, query=question)


@app.command(name="search")
def search_command(
    query: str = typer.Argument(..., help="Search query."),
    top_k: int = typer.Option(5, help="Number of chunks to retrieve."),
    retrieval_mode: str = typer.Option(
        get_default_retrieval_mode(),
        help="Retrieval mode: auto, tfidf, embedding, or hybrid.",
    ),
) -> None:
    """Search the saved local index and print matching chunks."""
    if retrieval_mode not in VALID_RETRIEVAL_MODES:
        console.print(f"[red]Invalid retrieval mode: {retrieval_mode}[/red]")
        raise typer.Exit(code=1)
    index = _load_default_index()
    try:
        results = search(index, query, top_k=top_k, retrieval_mode=retrieval_mode)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    _print_sources(results, query=query)


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", help="Host for the web server."),
    port: int = typer.Option(8000, help="Port for the web server."),
    reload: bool = typer.Option(False, help="Enable auto-reload for development."),
) -> None:
    """Launch the placeholder FastAPI web app."""
    console.print(f"[green]Starting Personal Docs QA web app at http://{host}:{port}[/green]")
    uvicorn.run("personal_docs_qa.web:app", host=host, port=port, reload=reload)


@app.command()
def demo() -> None:
    """Ingest sample docs and run example questions."""
    sample_folder = Path(__file__).resolve().parents[2] / "sample_docs"
    console.print(f"[bold]Ingesting sample docs from {sample_folder}[/bold]")

    try:
        result = index_folder_with_warnings(sample_folder, index_path=DEFAULT_INDEX_PATH)
    except ValueError as exc:
        console.print("[red]No sample chunks were produced.[/red]")
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    index = result.index
    console.print(f"[green]Indexed {index.document_count} documents into {index.chunk_count} chunks.[/green]")
    for warning in result.warnings:
        console.print(f"[yellow]- {warning}[/yellow]")

    questions = [
        "What is the inspection date?",
        "What ingredients are used in the coconut latte?",
        "What are the main project risks?",
        "How were AI tools used?",
    ]
    for question in questions:
        answer = answer_question(index, question, top_k=3)
        console.print(Panel(answer.answer, title=question, expand=False))
        _print_sources(answer.sources[:2], query=question)


if __name__ == "__main__":
    app()
