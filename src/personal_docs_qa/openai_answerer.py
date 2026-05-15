"""Small OpenAI grounded answer synthesis wrapper."""

from __future__ import annotations

from openai import OpenAI, OpenAIError

from personal_docs_qa.config import get_answer_model, get_openai_api_key
from personal_docs_qa.models import SearchResult


class OpenAIAnswerError(RuntimeError):
    """Friendly error raised for OpenAI answer synthesis failures."""


def _context_from_sources(results: list[SearchResult], max_chars_per_source: int = 1200) -> str:
    parts: list[str] = []
    for result in results:
        text = " ".join(result.chunk.text.split())
        if len(text) > max_chars_per_source:
            text = text[: max_chars_per_source - 3].rstrip() + "..."
        page = result.chunk.metadata.get("page_number")
        page_label = f"Page: {page}\n" if isinstance(page, int) else ""
        parts.append(
            f"[{result.rank}: {result.chunk.file_name}]\n"
            f"{page_label}"
            f"Score: {result.score:.3f}\n"
            f"{text}"
        )
    return "\n\n".join(parts)


def synthesize_answer(
    question: str,
    results: list[SearchResult],
    confidence: str,
    model: str | None = None,
) -> str:
    """Synthesize a concise answer using only retrieved chunks as evidence."""
    if not results:
        raise OpenAIAnswerError("No retrieved sources were available for OpenAI synthesis.")

    api_key = get_openai_api_key()
    if not api_key:
        raise OpenAIAnswerError("OPENAI_API_KEY is required for OpenAI answer synthesis.")

    selected_model = model or get_answer_model()
    client = OpenAI(api_key=api_key)
    context = _context_from_sources(results)
    prompt = (
        "Answer the question using only the provided source passages.\n"
        "Keep the answer concise: 2-5 sentences.\n"
        "Cite sources inline using exact bracket labels from the source headers, for example [1: file.pdf].\n"
        "Do not use citation shorthand like [1] or [2]; include the file name in each citation.\n"
        "If the passages do not fully answer the question, say that clearly before summarizing related evidence.\n"
        "Do not add facts that are not present in the sources.\n\n"
        f"Question: {question}\n"
        f"Retrieval confidence: {confidence}\n\n"
        f"Sources:\n{context}"
    )

    try:
        response = client.responses.create(
            model=selected_model,
            input=prompt,
            max_output_tokens=260,
        )
    except OpenAIError as exc:
        raise OpenAIAnswerError(
            "OpenAI answer synthesis failed. Check that OPENAI_API_KEY is valid and the answer model is available."
        ) from exc
    except Exception as exc:
        raise OpenAIAnswerError("OpenAI answer synthesis failed.") from exc

    answer_text = getattr(response, "output_text", None)
    if not answer_text:
        raise OpenAIAnswerError("OpenAI answer synthesis returned an empty response.")
    return answer_text.strip()
