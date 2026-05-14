"""Local extractive answer generation."""

import re
from dataclasses import dataclass

from personal_docs_qa.config import is_openai_available
from personal_docs_qa.indexer import LocalIndex
from personal_docs_qa.models import Answer, SearchResult
from personal_docs_qa.retriever import search


LOW_CONFIDENCE_THRESHOLD = 0.12
HIGH_CONFIDENCE_THRESHOLD = 0.35
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[a-zA-Z0-9']+")


@dataclass(frozen=True)
class AnswerResult:
    answer: Answer
    retrieval_mode_requested: str
    retrieval_mode_used: str
    retrieval_fallback_used: bool
    answer_mode_requested: str
    answer_mode_used: str
    answer_fallback_used: bool
    warnings: list[str]


def _question_terms(question: str) -> set[str]:
    stop_words = {
        "a",
        "about",
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
        "say",
        "says",
        "that",
        "the",
        "these",
        "this",
        "those",
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


def _has_query_overlap(results: list[SearchResult], question: str) -> bool:
    question_terms = _question_terms(question)
    if not question_terms:
        return True
    for result in results[:3]:
        chunk_terms = {word.lower() for word in WORD_RE.findall(result.chunk.text)}
        if question_terms & chunk_terms:
            return True
    return False


def _confidence(results: list[SearchResult], question: str) -> str:
    if not results:
        return "low"
    if not _has_query_overlap(results, question):
        return "low"
    best_score = results[0].score
    if best_score >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if best_score >= LOW_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def _local_answer_from_results(question: str, results: list[SearchResult], warnings: list[str] | None = None) -> Answer:
    warnings = list(warnings or [])
    if not results:
        return Answer(
            question=question,
            answer="I could not find any matching passages in the indexed documents.",
            sources=[],
            warnings=[*warnings, "No search results were returned."],
            confidence="low",
        )

    confidence = _confidence(results, question)
    useful_results = [result for result in results if result.score >= LOW_CONFIDENCE_THRESHOLD]
    positive_results = [result for result in results if result.score > 0]
    evidence_results = useful_results or positive_results or results
    answer_sentences = _extract_sentences(evidence_results, question)

    if confidence == "low":
        warnings.append("Top retrieval scores are weak; there may be insufficient evidence to answer fully.")
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


def _resolve_answer_mode(answer_mode: str) -> tuple[str, bool, list[str]]:
    requested = answer_mode or "auto"
    if requested == "local":
        return "local", False, []
    if requested == "openai":
        if is_openai_available():
            return "local", True, [
                "OpenAI answer synthesis is not implemented yet; using local extractive answerer."
            ]
        return "local", True, ["OPENAI_API_KEY is not set; using local extractive answerer."]
    if requested == "auto":
        if is_openai_available():
            return "local", True, [
                "OpenAI answer synthesis is not implemented yet; using local extractive answerer."
            ]
        return "local", True, ["OPENAI_API_KEY is not set; using local extractive answerer."]
    return "local", True, [f"Unknown answer mode '{requested}', using local extractive answerer."]


def answer_question(
    index: LocalIndex | None,
    question: str,
    top_k: int = 5,
    retrieval_mode: str = "auto",
) -> Answer:
    """Create a concise local extractive answer with citations."""
    return answer_question_with_metadata(
        index,
        question,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        answer_mode="local",
    ).answer


def answer_question_with_metadata(
    index: LocalIndex | None,
    question: str,
    top_k: int = 5,
    retrieval_mode: str = "auto",
    answer_mode: str = "auto",
) -> AnswerResult:
    """Create an answer and include mode/fallback metadata."""
    retrieval_mode_requested = retrieval_mode or "auto"
    answer_mode_requested = answer_mode or "auto"
    answer_mode_used, answer_fallback_used, answer_warnings = _resolve_answer_mode(answer_mode_requested)

    if index is None:
        answer = Answer(
            question=question,
            answer="No index is available. Run `docqa ingest <folder>` first.",
            sources=[],
            warnings=[*answer_warnings, "No index was provided."],
            confidence="low",
        )
        return AnswerResult(
            answer=answer,
            retrieval_mode_requested=retrieval_mode_requested,
            retrieval_mode_used="none",
            retrieval_fallback_used=False,
            answer_mode_requested=answer_mode_requested,
            answer_mode_used=answer_mode_used,
            answer_fallback_used=answer_fallback_used,
            warnings=answer.warnings,
        )
    if not index.chunks:
        answer = Answer(
            question=question,
            answer="The index has no chunks to search.",
            sources=[],
            warnings=[*answer_warnings, "The loaded index is empty."],
            confidence="low",
        )
        return AnswerResult(
            answer=answer,
            retrieval_mode_requested=retrieval_mode_requested,
            retrieval_mode_used="none",
            retrieval_fallback_used=False,
            answer_mode_requested=answer_mode_requested,
            answer_mode_used=answer_mode_used,
            answer_fallback_used=answer_fallback_used,
            warnings=answer.warnings,
        )

    try:
        results = search(index, question, top_k=top_k, retrieval_mode=retrieval_mode_requested)
    except ValueError as exc:
        answer = Answer(
            question=question,
            answer=str(exc),
            sources=[],
            warnings=[*answer_warnings, str(exc)],
            confidence="low",
        )
        return AnswerResult(
            answer=answer,
            retrieval_mode_requested=retrieval_mode_requested,
            retrieval_mode_used="none",
            retrieval_fallback_used=False,
            answer_mode_requested=answer_mode_requested,
            answer_mode_used=answer_mode_used,
            answer_fallback_used=answer_fallback_used,
            warnings=answer.warnings,
        )

    retrieval_mode_used = results[0].retrieval_mode_used if results else "none"
    retrieval_fallback_used = retrieval_mode_requested != retrieval_mode_used
    retrieval_warnings = []
    if retrieval_fallback_used:
        retrieval_warnings.append(
            f"Retrieval fell back from {retrieval_mode_requested} to {retrieval_mode_used}."
        )
    answer = _local_answer_from_results(question, results, warnings=[*answer_warnings, *retrieval_warnings])
    return AnswerResult(
        answer=answer,
        retrieval_mode_requested=retrieval_mode_requested,
        retrieval_mode_used=retrieval_mode_used,
        retrieval_fallback_used=retrieval_fallback_used,
        answer_mode_requested=answer_mode_requested,
        answer_mode_used=answer_mode_used,
        answer_fallback_used=answer_fallback_used,
        warnings=answer.warnings,
    )
