from typer.testing import CliRunner

from personal_docs_qa.cli import app
from personal_docs_qa.web import create_app


def test_cli_help() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "ask" in result.output
    assert "search" in result.output
    assert "web" in result.output


def test_web_help() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["web", "--help"])

    assert result.exit_code == 0
    assert "Launch the placeholder FastAPI web app." in result.output


def test_create_web_app() -> None:
    web_app = create_app()

    assert web_app.title == "Personal Docs QA"


def test_cli_ingest_and_ask(tmp_path) -> None:
    runner = CliRunner()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "budget.txt").write_text("The grocery budget is 120 dollars per week.", encoding="utf-8")

    with runner.isolated_filesystem():
        ingest_result = runner.invoke(app, ["ingest", str(docs)])
        ask_result = runner.invoke(app, ["ask", "What is the grocery budget?"])

    assert ingest_result.exit_code == 0
    assert "Documents loaded" in ingest_result.output
    assert ask_result.exit_code == 0
    assert "budget.txt" in ask_result.output


def test_cli_missing_index_message() -> None:
    runner = CliRunner()

    with runner.isolated_filesystem():
        result = runner.invoke(app, ["ask", "What is indexed?"])

    assert result.exit_code == 1
    assert "No index found. Run `docqa ingest <folder>` first." in result.output
