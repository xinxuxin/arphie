# Process Notes

This project is being built as a compact, reliable take-home project rather than a production platform.

## Implementation Principles

- Keep the CLI and web app thin.
- Put shared behavior in the core package modules.
- Prefer simple, inspectable local dependencies.
- Make citations visible and deterministic.
- Handle edge cases clearly instead of hiding failures.
- Avoid API-key requirements by default.

## Planned Phases

1. Create the Python project scaffold, dependency metadata, CLI entry point, and placeholder web app.
2. Implement document loaders for Markdown, text, and PDFs.
3. Add chunking with source metadata.
4. Build and persist a local TF-IDF index.
5. Add retrieval and extractive answer generation.
6. Wire the shared engine into both CLI and web interfaces.
7. Add focused tests and a small eval harness.
8. Polish README instructions, edge cases, and demo flow.

## AI Tool Usage

AI assistance is being used to scaffold the project, generate initial code, draft documentation, and plan implementation steps. Human review should focus on whether the code remains small, understandable, reproducible, and honest about limitations.

When implementation proceeds beyond this scaffold, AI-generated code should be validated with:

- `pytest`
- Manual CLI smoke tests
- Manual web app smoke tests
- Sample document QA checks with visible source citations

## Current Status

The repository currently contains the initial project structure only. The full document QA logic has not yet been implemented.

