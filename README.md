# Personal Docs QA

## Overview

Personal Docs QA is a small personal document question-answering app with both a CLI and a lightweight web interface. It is designed for a folder of mixed personal documents and keeps the core indexing, retrieval, and answering logic shared between both interfaces.

The app supports:

- `.txt`, `.md`, and text-based `.pdf` files
- Local TF-IDF retrieval
- Optional OpenAI embedding retrieval
- Hybrid retrieval that combines TF-IDF and embeddings
- Answer modes for local extractive answers and OpenAI-grounded synthesis with local fallback
- Visible citations to retrieved source chunks

It runs without API keys by default. The fresh-clone path is fully local: load documents, chunk them, build a TF-IDF index, retrieve matching chunks, and answer conservatively from the retrieved evidence.

Note: the OpenAI answer mode is exposed through the CLI and web metadata, but this take-home currently keeps synthesis conservative and falls back to local extractive answering. The OpenAI-enhanced path is fully implemented for embeddings and hybrid retrieval.

## Quickstart: Local-Only

```bash
pip install -e .
docqa ingest sample_docs --retrieval-mode tfidf
docqa ask "What is the inspection date?" --retrieval-mode tfidf --answer-mode local
docqa web
```

Then open [http://localhost:8000](http://localhost:8000).

## Quickstart: OpenAI-Enhanced

```bash
export OPENAI_API_KEY=...
docqa ingest sample_docs --retrieval-mode hybrid
docqa ask "What is the inspection date?" --retrieval-mode hybrid --answer-mode auto
```

OpenAI is optional. If the key is missing or an OpenAI request fails in `auto` mode, the app falls back to local behavior and surfaces a warning instead of crashing.

## Retrieval Modes

- `tfidf`: local, deterministic, keyword-based retrieval using scikit-learn TF-IDF and cosine similarity.
- `embedding`: OpenAI semantic retrieval using stored chunk embeddings and a query embedding. Requires `OPENAI_API_KEY`.
- `hybrid`: combines normalized TF-IDF and embedding scores, so exact terms and semantic similarity can both contribute.
- `auto`: uses hybrid if OpenAI is available and the index has embeddings; otherwise falls back to TF-IDF.

The default embedding model is `text-embedding-3-small`. The default embedding dimension is `512`, chosen to keep local indexes compact and fast for the expected take-home scale.

## Answer Modes

- `local`: extractive local answer built only from retrieved chunks.
- `openai`: intended for grounded answer synthesis from retrieved chunks, with citations. In the current take-home implementation, it falls back to local extractive answering rather than inventing unsupported synthesis.
- `auto`: tries the strongest configured answer path first, then uses local fallback when OpenAI is unavailable or synthesis is not enabled.

The answer response always reports the requested mode, the mode actually used, whether fallback happened, confidence, warnings, and sources.

## Why This Design

TF-IDF keeps the fresh-clone experience reliable. A reviewer can install the package and run the app without credentials, network access, model downloads, or a vector database.

Embeddings improve semantic matching when the question uses different wording from the documents. Hybrid retrieval handles both exact terms and paraphrases by combining lexical and semantic signals.

OpenAI answer synthesis is the natural next step for fluency, but the system still treats retrieved evidence as the authority. The app is intentionally cautious: if the evidence is weak, it says so.

Fallback behavior keeps the app robust. Missing API keys, OpenAI failures, old indexes without embeddings, and weak matches all produce readable warnings rather than raw stack traces.

Personal documents may be sensitive, so local mode is always available. Users can choose the fully local path when privacy or reproducibility matters more than semantic retrieval.

## Architecture Diagram

```text
Documents
  -> Loaders
  -> Chunker
  -> Index Builder
      -> TF-IDF
      -> Optional OpenAI embeddings
  -> Retriever
      -> TF-IDF / Embedding / Hybrid
  -> Answerer
      -> OpenAI synthesis / Local fallback
  -> CLI + Web
```

Core modules live in `src/personal_docs_qa/`:

- `loaders.py`: reads TXT, Markdown, and text-based PDFs.
- `chunker.py`: creates paragraph-aware chunks with overlap.
- `indexer.py`: builds and persists TF-IDF indexes, optionally with OpenAI embeddings.
- `retriever.py`: runs TF-IDF, embedding, hybrid, or auto retrieval.
- `answerer.py`: produces conservative cited answers and reports mode/fallback metadata.
- `cli.py` and `web.py`: thin interfaces over the shared engine.

## Web Demo

```bash
docqa web
```

Open [http://localhost:8000](http://localhost:8000), then:

1. Select a retrieval mode.
2. Select an answer mode.
3. Index `sample_docs` or upload files.
4. Ask a question.
5. Inspect the pipeline, answer badges, fallback warnings, confidence, and source cards.

The web UI is static HTML/CSS/JS served by FastAPI. There is no React build step.

## Edge Cases Handled

- Missing API key
- OpenAI API failure
- Old index without embeddings
- Empty files
- Unsupported files
- Missing index
- Weak matches
- Unreadable PDFs

## What I Skipped And Why

- OCR for scanned PDFs: valuable, but too large for the take-home scope.
- Production authentication: unnecessary for a local demo.
- Cloud deployment: not required for the assignment.
- Persistent vector DB: `joblib` is simpler and enough for this document scale.
- Complex React frontend: static HTML/CSS/JS is faster to review and easier to run.
- Full benchmark suite: the eval harness is a sanity check, not a production benchmark.

## What I Would Do With 4 More Hours

- Query rewriting
- Reranking
- Streaming answers
- Better PDF page highlighting
- Larger regression eval set
- Per-folder index management
- Optional local embedding model
- Drag-and-drop folder UX

## Weakest Part

Embedding quality depends on chunking. If chunks are too broad or too narrow, semantic retrieval can still surface imperfect evidence.

Hybrid weights are heuristic. The current weights are reasonable for a small demo, but a larger eval set should tune them.

OpenAI synthesis depends on retrieved evidence. If retrieval misses the right chunk, a fluent answer would still be poorly grounded. For that reason, this version stays conservative and uses local fallback.

PDF extraction only handles text-based PDFs through `pypdf`; scanned documents need OCR.

The UI is demo-grade, not production-grade. It is useful for a two-minute walkthrough, but not a hardened document management product.

## Evaluation

Run tests:

```bash
pytest
```

Run the lightweight eval harness:

```bash
python eval/run_eval.py
```

The eval harness ingests `sample_docs`, checks expected sources and terms, compares TF-IDF against hybrid when OpenAI is configured, and includes a weak question to verify low-confidence behavior. It is intentionally small: its purpose is to show retrieval quality and fallback behavior beyond manual clicking, not to be a full benchmark.
