# Project Notes

The main project risks are overbuilding the document QA app, weak PDF text extraction, limited answer synthesis, and loose matches when retrieval scores are low. The goal is a small local demo, so authentication, databases, cloud deployment, OCR, and a complex frontend are intentionally out of scope.

The most important technical tradeoff is using local-first retrieval with TF-IDF instead of embeddings or an external language model. This keeps the app fast, transparent, and API-key free, but it can miss answers when the question uses vocabulary that does not overlap with the documents.

Weak spots include lack of OCR for scanned documents and the possibility that low-scoring retrieved chunks are only loosely related to the question.

The project should prioritize clear citations, deterministic behavior, simple tests, and honest warnings when evidence is weak.
