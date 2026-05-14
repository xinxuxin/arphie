"""Command-line interface for Personal Docs QA."""

from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from personal_docs_qa.answerer import answer_question
from personal_docs_qa.chunker import chunk_documents
from personal_docs_qa.indexer import DEFAULT_INDEX_PATH, build_index, load_index, save_index
from personal_docs_qa.loaders import load_folder
from personal_docs_qa.retriever import search


app = typer.Typer(
    help="Ask questions over a local folder of personal documents.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """Ask questions over a local folder of personal documents."""


def _excerpt(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _load_default_index():
    try:
        return load_index(DEFAULT_INDEX_PATH)
    except FileNotFoundError:
        console.print("[yellow]No index found. Run `docqa ingest <folder>` first.[/yellow]")
        raise typer.Exit(code=1)


def _print_sources(results) -> None:
    table = Table(title="Sources")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("File")
    table.add_column("Excerpt")

    for result in results:
        table.add_row(
            str(result.rank),
            f"{result.score:.3f}",
            result.chunk.file_name,
            _excerpt(result.chunk.text),
        )

    console.print(table)


@app.command()
def ingest(
    folder: Path = typer.Argument(..., help="Folder of .txt, .md, and .pdf documents."),
    index_path: Path = typer.Option(DEFAULT_INDEX_PATH, help="Where to save the local index."),
) -> None:
    """Load documents, chunk them, and build a local TF-IDF index."""
    documents, warnings = load_folder(folder)
    chunks = chunk_documents(documents)

    if not chunks:
        console.print("[red]No chunks were produced; index was not created.[/red]")
        for warning in warnings:
            console.print(f"[yellow]- {warning}[/yellow]")
        raise typer.Exit(code=1)

    try:
        index = build_index(chunks)
        save_index(index, index_path)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(Panel.fit("Index built successfully", style="green"))
    console.print(f"Documents loaded: [bold]{len(documents)}[/bold]")
    console.print(f"Chunks indexed: [bold]{index.chunk_count}[/bold]")
    console.print(f"Index path: [bold]{index_path}[/bold]")
    if warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"[yellow]- {warning}[/yellow]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask over the local index."),
    top_k: int = typer.Option(5, help="Number of chunks to retrieve."),
) -> None:
    """Ask a question against the saved local index."""
    index = _load_default_index()
    answer = answer_question(index, question, top_k=top_k)

    console.print(Panel(answer.answer, title=f"Answer ({answer.confidence} confidence)", expand=False))
    if answer.warnings:
        for warning in answer.warnings:
            console.print(f"[yellow]{warning}[/yellow]")
    if answer.sources:
        _print_sources(answer.sources)


@app.command(name="search")
def search_command(
    query: str = typer.Argument(..., help="Search query."),
    top_k: int = typer.Option(5, help="Number of chunks to retrieve."),
) -> None:
    """Search the saved local index and print matching chunks."""
    index = _load_default_index()
    try:
        results = search(index, query, top_k=top_k)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    _print_sources(results)


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

    documents, warnings = load_folder(sample_folder)
    chunks = chunk_documents(documents)
    if not chunks:
        console.print("[red]No sample chunks were produced.[/red]")
        raise typer.Exit(code=1)

    index = build_index(chunks)
    save_index(index, DEFAULT_INDEX_PATH)
    console.print(f"[green]Indexed {len(documents)} documents into {index.chunk_count} chunks.[/green]")
    for warning in warnings:
        console.print(f"[yellow]- {warning}[/yellow]")

    questions = [
        "What is the grocery budget?",
        "What documents should I bring for travel?",
        "What is the project decision?",
    ]
    for question in questions:
        answer = answer_question(index, question, top_k=3)
        console.print(Panel(answer.answer, title=question, expand=False))
        _print_sources(answer.sources[:2])


if __name__ == "__main__":
    app()
