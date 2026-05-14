from typer.testing import CliRunner

from personal_docs_qa.cli import app
from personal_docs_qa.web import create_app


def test_cli_help() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "web" in result.output


def test_web_help() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["web", "--help"])

    assert result.exit_code == 0
    assert "Launch the placeholder FastAPI web app." in result.output


def test_create_web_app() -> None:
    web_app = create_app()

    assert web_app.title == "Personal Docs QA"
