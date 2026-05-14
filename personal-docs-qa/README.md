# Personal Docs QA

A small take-home project for asking natural-language questions over a local folder of personal documents.

This repository is intentionally modest: one shared Python core will power both a CLI and a lightweight FastAPI web demo. The first phase contains the project scaffold, dependency setup, placeholder interfaces, and documentation shape.

## Goals

- Load `.md`, `.txt`, and `.pdf` documents from a folder
- Chunk documents into searchable passages
- Build a local TF-IDF index
- Retrieve relevant chunks for a question
- Return concise answers with visible source citations
- Offer both a CLI and a web app using one shared engine
- Run locally with no API key by default

## Quick Start

```bash
cd personal-docs-qa
python -m venv .venv
source .venv/bin/activate
pip install -e .
docqa --help
docqa web
```

The web command currently starts a placeholder FastAPI app. Full document loading, indexing, retrieval, and answer generation will be implemented in later phases.

## Project Structure

```text
personal-docs-qa/
├── README.md
├── PROCESS.md
├── pyproject.toml
├── sample_docs/
├── eval/
├── tests/
└── src/
    └── personal_docs_qa/
        ├── __init__.py
        ├── models.py
        ├── loaders.py
        ├── chunker.py
        ├── indexer.py
        ├── retriever.py
        ├── answerer.py
        ├── cli.py
        ├── web.py
        └── static/
            ├── index.html
            ├── app.js
            └── styles.css
```

## Current CLI

```bash
docqa --help
docqa web
```

## Development

```bash
pip install -e .
pytest
```

The current tests only verify that the scaffold imports and the web app exists. They will expand as the shared core engine is implemented.

## Non-Goals

- Authentication
- Cloud deployment
- Database storage
- OCR
- Complex frontend framework
- External LLM or embedding API requirement

