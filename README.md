# Personal Docs QA

## Project Overview

Personal Docs QA is a local document question-answering app for small folders of personal documents. It supports both a Typer CLI and a lightweight FastAPI web app.

Users can index Markdown, plain text, and text-based PDF documents, then ask natural-language questions over the indexed content. The default no-key path retrieves relevant source chunks with TF-IDF, while optional OpenAI embedding and hybrid modes are available when an API key is configured. Answers include visible citations back to retrieved chunks.

The default path is fully local. No API key, external model, hosted database, or cloud service is required.

## Why Both CLI And Web

The CLI is the most reliable interface: it is scriptable, fast to test, and easy to use in automated checks. It also exposes the core workflow clearly: ingest documents, search, ask questions, and run a demo.

The web app is easier to demo and inspect. It lets a reviewer index a local folder or upload files, ask a question, and see the answer plus source snippets in a browser.

Both interfaces share the same core engine. Loading, chunking, indexing, retrieval, and answer generation live in `src/personal_docs_qa/`; the CLI and web app are thin wrappers around that shared code.

## Quickstart

```bash
git clone https://github.com/xinxuxin/arphie.git personal-docs-qa
cd personal-docs-qa
python -m venv .venv
source .venv/bin/activate
pip install -e .
docqa ingest sample_docs
docqa ask "What ingredients are used in the coconut latte?"
docqa web
```

## Web App Usage

Start the local web server:

```bash
docqa web
```

Open [http://localhost:8000](http://localhost:8000). From the UI, index `sample_docs` or upload files, ask a question, and review the answer sources. The web backend saves its local index to `.docqa/web_index.joblib`.

This web app is intentionally small: static HTML/CSS/JS served by FastAPI, with JSON endpoints for ingesting, uploading, and asking questions.

## CLI Usage

```bash
docqa ingest sample_docs
```

Loads supported documents, chunks them, builds a local index, and saves it to `.docqa/index.joblib`.

```bash
docqa ingest sample_docs --retrieval-mode tfidf
```

Forces local TF-IDF retrieval. Other modes are documented below.

```bash
docqa ask "What is the inspection date?"
```

Loads the saved index and prints a concise cited answer with source snippets.

```bash
docqa search "inspection date"
```

Prints the top retrieved chunks with rank, score, file name, and excerpt.

```bash
docqa demo
```

Indexes `sample_docs` and runs a few example questions.

```bash
docqa web
```

Starts the FastAPI web app on `localhost:8000`.

## Architecture

```text
Documents
  ↓
Loaders (.txt / .md / .pdf)
  ↓
Chunker
  ↓
Search Index
  ↓
Retriever
  ↓
Answerer with citations
  ↓
CLI + Web UI
```

The important design choice is that both user interfaces call the same modules:

- `loaders.py` reads supported files and returns normalized `Document` objects.
- `chunker.py` creates paragraph-aware chunks with overlap.
- `indexer.py` builds and persists the local search index with `joblib`.
- `retriever.py` searches the matrix with cosine similarity.
- `answerer.py` builds conservative extractive answers from retrieved chunks.

## Retrieval Choice

The default retrieval configuration is `auto`: if `OPENAI_API_KEY` is present, the app uses hybrid retrieval; otherwise it falls back to TF-IDF with a warning. This preserves the no-key fresh clone path while allowing an optional semantic retrieval upgrade.

Available modes:

- `tfidf`: local TF-IDF only
- `embedding`: OpenAI embeddings only, requires `OPENAI_API_KEY`
- `hybrid`: combines TF-IDF and OpenAI embedding scores, requires `OPENAI_API_KEY`
- `auto`: uses `hybrid` when an API key exists, otherwise falls back to `tfidf`

Environment variables:

- `DOCQA_RETRIEVAL_MODE`
- `OPENAI_API_KEY`
- `OPENAI_EMBEDDING_MODEL`
- `OPENAI_EMBEDDING_DIMENSIONS`

The default embedding model is `text-embedding-3-small`. The default embedding dimension is `512`, chosen to keep persisted local indexes small and fast for this take-home scale. OpenAI's `text-embedding-3-small` can use larger dimensions, but 512 is a practical speed/storage tradeoff for about 20 documents and 50,000 words.

I chose TF-IDF as the guaranteed baseline because it is local, deterministic, dependency-light, and fast. For the expected take-home size, it is more than adequate and keeps the system easy to reason about.

TF-IDF also avoids requiring API keys. A reviewer can clone the repo, install dependencies, and run the app without configuring credentials.

The main weakness is that TF-IDF is lexical, not semantic. It can miss paraphrases or questions that use very different wording from the documents. Embedding and hybrid modes help with that, but they are optional because they require an OpenAI API key. The app surfaces weak matches through low confidence and warnings rather than pretending the answer is stronger than the evidence.

## What I Built

- CLI with `ingest`, `ask`, `search`, `demo`, and `web`
- FastAPI web backend and static demo UI
- TXT, Markdown, and PDF loading
- Paragraph-aware chunking with overlap
- Local TF-IDF retrieval
- Optional OpenAI embedding and hybrid retrieval
- Extractive answers with visible citations
- Local index persistence with `joblib`
- Lightweight eval harness
- Focused pytest coverage

## Edge Cases Handled

- Empty files are skipped with warnings
- Unsupported file types are ignored
- Missing index produces a helpful error
- Weak matches produce cautious answers and warnings
- Unreadable files do not crash the whole ingest
- Malformed PDFs do not crash the whole ingest
- Duplicate file names in different folders keep distinct paths and IDs
- No API key is required

## What I Skipped And Why

- OCR for scanned PDFs: valuable, but too much scope for a compact take-home
- Authentication: unnecessary for a local demo
- Database storage: `joblib` persistence is simpler and sufficient here
- Cloud deployment: not needed for the assignment goals
- Vector database or advanced semantic retrieval pipeline: useful later, but more scope than needed here
- Production-grade LLM answer synthesis: I prioritized reliable citations and avoiding hallucination

## What I Would Do With 4 More Hours

- Hybrid BM25 plus embeddings retrieval
- Optional LLM synthesis with strict citation grounding
- Better PDF page-level citations in answers and UI
- Incremental indexing for changed files
- Persistent multi-index support
- Richer regression eval with expected answer checks
- Drag-and-drop folder UX improvements

## Weakest Part

Answer synthesis is conservative and extractive. That makes it safer and easier to inspect, but less fluent than a full LLM answer.

TF-IDF is not truly semantic, so paraphrased questions can be missed.

PDF support is limited to text-based PDFs via `pypdf`; scanned PDFs require OCR, which is intentionally out of scope.

## Evaluation

I used four layers of validation:

- `pytest` for loaders, chunking, indexing, retrieval, answer behavior, CLI, and web API routes
- `python eval/run_eval.py` as a lightweight regression harness over `sample_docs`
- Manual CLI checks such as `docqa ingest sample_docs`, `docqa ask ...`, and `docqa search ...`
- Manual web demo check via `docqa web` and `/api/health`

The eval harness is not a benchmark. It is a small sanity check that expected sources appear in retrieved results and expected terms appear in the answer or source excerpts.
