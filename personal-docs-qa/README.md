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

The CLI and web app use the same local document QA engine. No API key is required by default.

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
docqa ingest sample_docs
docqa ask "What is the inspection date?"
docqa web
docqa demo
```

## Sample Documents

The `sample_docs/` folder contains small Markdown and text documents for demos. They support questions about lease inspection notes, coconut latte recipes, project risks, and AI tool usage.

PDF loading is supported through `pypdf`. This repo does not include a checked-in PDF fixture, but tests can create temporary PDF inputs where needed.

## Development

```bash
pip install -e .
pytest
python eval/run_eval.py
```

The tests cover the core loading, chunking, indexing, retrieval, answer, CLI, and web API behavior.

## Non-Goals

- Authentication
- Cloud deployment
- Database storage
- OCR
- Complex frontend framework
- External LLM or embedding API requirement
