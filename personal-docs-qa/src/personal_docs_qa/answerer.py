"""Local extractive answer generation."""

import re

from personal_docs_qa.indexer import LocalIndex
from personal_docs_qa.models import Answer, SearchResult
from personal_docs_qa.retriever import search


LOW_CONFIDENCE_THRESHOLD = 0.12
HIGH_CONFIDENCE_THRESHOLD = 0.35
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[a-zA-Z0-9']+")


def _question_terms(question: str) -> set[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "did",
        "do",
        "does",
        "for",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "the",
        "to",
        "what",
        "when",
        "where",
        "who",
        "why",
    }
    return {word.lower() for word in WORD_RE.findall(question) if word.lower() not in stop_words}


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_RE.split(text.strip()) if sentence.strip()]


def _score_sentence(sentence: str, question_terms: set[str]) -> int:
    sentence_terms = {word.lower() for word in WORD_RE.findall(sentence)}
    return len(sentence_terms & question_terms)


def _citation(result: SearchResult) -> str:
    return f"[{result.rank}: {result.chunk.file_name}]"


def _extract_sentences(results: list[SearchResult], question: str, limit: int = 5) -> list[str]:
    question_terms = _question_terms(question)
    selected: list[str] = []

    for result in results:
        candidates = _sentences(result.chunk.text)
        if not candidates and result.chunk.text.strip():
            candidates = [result.chunk.text.strip()]

        ranked = sorted(
            candidates,
            key=lambda sentence: _score_sentence(sentence, question_terms),
            reverse=True,
        )
        for sentence in ranked[:1]:
            if sentence and sentence not in selected:
                selected.append(f"{sentence} {_citation(result)}")
                break
        if len(selected) >= limit:
            break

    return selected


def _confidence(results: list[SearchResult]) -> str:
    if not results:
        return "low"
    best_score = results[0].score
    if best_score >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if best_score >= LOW_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def answer_question(index: LocalIndex | None, question: str, top_k: int = 5) -> Answer:
    """Create a concise local extractive answer with citations."""
    if index is None:
        return Answer(
            question=question,
            answer="No index is available. Run `docqa ingest <folder>` first.",
            sources=[],
            warnings=["No index was provided."],
            confidence="low",
        )
    if not index.chunks:
        return Answer(
            question=question,
            answer="The index has no chunks to search.",
            sources=[],
            warnings=["The loaded index is empty."],
            confidence="low",
        )

    try:
        results = search(index, question, top_k=top_k)
    except ValueError as exc:
        return Answer(
            question=question,
            answer=str(exc),
            sources=[],
            warnings=[str(exc)],
            confidence="low",
        )

    if not results:
        return Answer(
            question=question,
            answer="I could not find any matching passages in the indexed documents.",
            sources=[],
            warnings=["No search results were returned."],
            confidence="low",
        )

    confidence = _confidence(results)
    warnings: list[str] = []
    positive_results = [result for result in results if result.score > 0]
    evidence_results = positive_results or results
    answer_sentences = _extract_sentences(evidence_results, question)

    if confidence == "low":
        warnings.append("Top retrieval scores are weak; the answer may be incomplete.")
        prefix = "I found related passages, but they may not fully answer the question."
        if answer_sentences:
            answer_text = f"{prefix} " + " ".join(answer_sentences[:2])
        else:
            answer_text = prefix
    else:
        answer_text = " ".join(answer_sentences[:5])

    # TODO: Optional LLM synthesis could be added here later, using only retrieved chunks as context.
    return Answer(
        question=question,
        answer=answer_text,
        sources=results,
        warnings=warnings,
        confidence=confidence,
    )
