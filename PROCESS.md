# Process

## Tools Used

- ChatGPT for scoping, architecture discussion, README planning, and checking whether the project stayed inside the take-home constraints.
- Codex for implementation scaffolding, iterative coding, refactors, tests, and repeated local verification.
- Manual review for tradeoff decisions, edge-case selection, and deciding where to keep the implementation intentionally small.

## Where AI Helped

- Scaffolding the Typer CLI entry point and command structure.
- Creating the FastAPI route structure for folder ingest, upload ingest, ask, health, and static file serving.
- Drafting loader tests for `.txt`, `.md`, empty files, unsupported files, and broken paths.
- Suggesting edge cases such as empty folders, malformed PDFs, duplicate file names in different folders, missing indexes, and empty queries.
- Generating the first version of the static web UI so the app was demoable without introducing React or a frontend build step.
- Iterating on the README and eval harness so the reviewer can understand and validate the project quickly.

## Where AI Was Wrong Or Too Broad

- It initially leaned toward a heavier frontend direction, including ideas closer to a React app. I kept the web app as static HTML/CSS/JS served by FastAPI to fit the time limit.
- It suggested LLM or embedding-first retrieval in some planning paths. I kept TF-IDF as the default so the app runs locally without API keys.
- It suggested simple character-based chunking at first. I adjusted the chunker to prefer paragraph boundaries and avoid sentence splits when reasonable.
- It generated overly optimistic error handling in early placeholders. I manually checked missing index, empty input, unreadable file, and malformed PDF behavior.
- It occasionally created implementation details that looked fine but needed real smoke tests, such as CLI command shape and FastAPI test dependencies.

## Where I Overrode AI

- I chose one shared core engine instead of separate CLI and web implementations.
- I chose local-first retrieval and local index persistence.
- I prioritized citations, edge cases, and the eval harness over a more polished UI.
- I kept answer generation conservative and extractive to avoid hallucination.
- I avoided adding authentication, a database, cloud deployment, OCR, or external model calls.

## Verification

I verified the project with:

```bash
pytest
docqa ingest sample_docs
docqa ask "What ingredients are used in the coconut latte?"
docqa search "inspection date"
python eval/run_eval.py
docqa web
```

For the web app, I also checked the local health endpoint:

```bash
curl http://127.0.0.1:8000/api/health
```

## Remaining Concerns

- TF-IDF is lexical, so it can miss semantic paraphrases.
- Scanned PDFs are unsupported because there is no OCR.
- Source snippets are chunk-level citations, not perfect page-level citations in every answer.
- Web upload is local demo quality, not production-grade file handling.
- The extractive answerer is intentionally conservative and may be less fluent than an LLM-generated answer.
