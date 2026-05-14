"""Command-line interface for Personal Docs QA."""

import typer
import uvicorn
from rich.console import Console


app = typer.Typer(
    help="Ask questions over a local folder of personal documents.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """Ask questions over a local folder of personal documents."""


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", help="Host for the web server."),
    port: int = typer.Option(8000, help="Port for the web server."),
    reload: bool = typer.Option(False, help="Enable auto-reload for development."),
) -> None:
    """Launch the placeholder FastAPI web app."""
    console.print(f"[green]Starting Personal Docs QA web app at http://{host}:{port}[/green]")
    uvicorn.run("personal_docs_qa.web:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
