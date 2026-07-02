"""Prompt construction for a conversational, grounded document chatbot."""

from __future__ import annotations

import re
from typing import Any

from app.query_intent import (
    INTENT_COMPARISON,
    INTENT_EXPLANATION,
    INTENT_EXTRACTION,
    INTENT_SUMMARY,
    is_vietnamese_query,
)


REFUSAL_VI = (
    "T\u00f4i kh\u00f4ng t\u00ecm th\u1ea5y th\u00f4ng tin n\u00e0y "
    "trong t\u00e0i li\u1ec7u \u0111\u01b0\u1ee3c cung c\u1ea5p."
)
REFUSAL_EN = "I could not find this information in the provided document."


def build_context(
    retrieved_chunks: list[dict[str, Any]],
    max_chunk_chars: int = 1200,
    max_total_chars: int = 6000,
) -> str:
    """Format retrieved chunks as source-labelled context."""

    context_blocks = []
    used_chars = 0
    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})
        page = metadata.get("page", "N/A")
        file_name = metadata.get("file_name", "unknown")
        chunk_id = chunk.get("chunk_id", "unknown")
        text = _compress_context_text(chunk.get("text", ""), max_chunk_chars)

        block = "\n".join(
            [
                f"[{index}]",
                f"File: {file_name}",
                f"Page: {page}",
                f"Chunk ID: {chunk_id}",
                f"Content: {text}",
            ]
        )
        if used_chars and used_chars + len(block) > max_total_chars:
            break

        context_blocks.append(block)
        used_chars += len(block)

    return "\n\n".join(context_blocks)


def _compress_context_text(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text

    head_chars = max(300, int(max_chars * 0.65))
    tail_chars = max(200, max_chars - head_chars - 20)
    head = text[:head_chars].rsplit(" ", 1)[0].strip()
    tail = text[-tail_chars:].split(" ", 1)[-1].strip()
    return (
        f"{head}\n"
        "[...context shortened...]\n"
        f"{tail}"
    )


def build_chat_history(chat_history: list[dict[str, str]] | None, max_turns: int = 4) -> str:
    """Format recent chat history for follow-up questions."""

    if not chat_history:
        return "No previous conversation."

    relevant_messages = [
        message
        for message in chat_history
        if message.get("role") in {"user", "assistant"} and message.get("content")
    ][-(max_turns * 2) :]

    if not relevant_messages:
        return "No previous conversation."

    return "\n".join(
        f"{message['role'].title()}: {message['content']}" for message in relevant_messages
    )


def build_prompt(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    chat_history: list[dict[str, str]] | None = None,
    intent: str = "qa",
) -> str:
    """Build a conversational prompt that still enforces document grounding."""

    context = build_context(retrieved_chunks)
    history = build_chat_history(chat_history)
    intent_instruction = _intent_instruction(intent)

    return f"""You are a helpful document-grounded chatbot.

You are chatting with the user about an uploaded document. Answer naturally and conversationally, but stay strictly grounded in the provided Context.

Core rules:
- Use the Conversation History only to understand follow-up references.
- Use only the Context as factual evidence for the answer.
- Never use previous assistant answers as evidence.
- Synthesize the answer in your own words; do not simply copy long raw passages.
- Do not invent details, numbers, names, requirements, or dates.
- For questions about a named person/entity, use only evidence that mentions that exact person/entity; ignore evidence about similarly named people/entities.
- Answer in exactly one language: the language of the current User Question.
- Do not include translations, bilingual explanations, or parenthetical translations.
- If the Context is insufficient, say exactly:
  - "{REFUSAL_VI}" for Vietnamese questions.
  - "{REFUSAL_EN}" for English questions.
- Match the language of the current User Question exactly.
- If the current User Question is Vietnamese, answer in Vietnamese.
- If the current User Question is English, answer in English.
- Do not switch languages just because the document or conversation history uses another language.
- Cite supporting evidence with compact citation labels like [1], [2].
- Do not write citation labels as [Source 1] or "Source 1"; use only [1].
- Every factual sentence or bullet must include at least one citation.
- Put citations at the end of the sentence or bullet they support. Avoid over-citing every short phrase.
- Keep the answer concise unless the user asks for detail.

Answer style for this request:
{intent_instruction}

Conversation History:
{history}

Context:
{context}

User Question:
{question}

Answer:
"""


def refusal_for_question(question: str) -> str:
    """Return a refusal message in the likely language of the question."""

    return REFUSAL_VI if is_vietnamese_query(question) else REFUSAL_EN


def normalize_answer(
    answer: str,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """Normalize answer language and citation format for the app contract."""

    normalized = _normalize_citation_labels(answer).strip()
    normalized = _keep_requested_language(normalized, question).strip()

    if _is_refusal(normalized):
        return normalized

    if retrieved_chunks and not _has_numeric_citation(normalized):
        normalized = _append_default_citation(normalized, source_id=1)

    return normalized


def answer_language_mismatch(answer: str, question: str) -> bool:
    """Return whether an answer appears to violate the requested language."""

    if _is_refusal(answer):
        return False

    question_is_vietnamese = is_vietnamese_query(question)
    answer_is_vietnamese = _looks_like_vietnamese_text(answer)
    if question_is_vietnamese:
        return False
    return answer_is_vietnamese


def build_language_repair_prompt(answer: str, question: str) -> str:
    """Build a strict rewrite prompt to repair language drift without adding facts."""

    if is_vietnamese_query(question):
        target_language = "Vietnamese"
    else:
        target_language = "English"

    return f"""Rewrite the answer below in {target_language} only.

Rules:
- Preserve the factual meaning.
- Preserve compact numeric citations like [1], [2].
- Do not add new facts or new citations.
- Do not include translations or bilingual explanations.
- Return only the rewritten answer.

Question:
{question}

Answer to rewrite:
{answer}
"""


def extract_sources(retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract source citations in the same order as prompt context labels."""

    sources = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})
        sources.append(
            {
                "source_id": index,
                "file_name": metadata.get("file_name", "unknown"),
                "page": metadata.get("page"),
                "chunk_id": chunk.get("chunk_id", "unknown"),
            }
        )

    return sources


def _normalize_citation_labels(answer: str) -> str:
    answer = re.sub(r"\[\s*Source\s+(\d+)\s*\]", r"[\1]", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\(\s*Source\s+(\d+)\s*\)", r"[\1]", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\bSource\s+(\d+)\b", r"[\1]", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", answer)
    return answer


def _keep_requested_language(answer: str, question: str) -> str:
    if is_vietnamese_query(question):
        return answer

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", answer) if paragraph.strip()]
    if len(paragraphs) <= 1:
        return answer

    english_paragraphs = [
        paragraph for paragraph in paragraphs if not _looks_like_vietnamese_text(paragraph)
    ]
    if english_paragraphs:
        return "\n\n".join(english_paragraphs)
    return answer


def _looks_like_vietnamese_text(text: str) -> bool:
    vietnamese_chars = sum(1 for char in text if char in _VIETNAMESE_CHARS)
    letters = sum(1 for char in text if char.isalpha())
    if letters == 0:
        return False
    if vietnamese_chars / letters >= 0.02:
        return True

    lowered = f" {text.lower()} "
    markers = (" theo ", " tài liệu ", " tác giả ", " được ", " của ", " và ", " không ")
    return any(marker in lowered for marker in markers)


def _has_numeric_citation(answer: str) -> bool:
    return bool(re.search(r"\[\d+(?:\s*,\s*\d+)*\]", answer))


def _append_default_citation(answer: str, source_id: int) -> str:
    if not answer:
        return answer

    citation = f"[{source_id}]"
    stripped = answer.rstrip()
    if stripped.endswith((".", "!", "?")):
        return f"{stripped} {citation}"
    return f"{stripped}. {citation}"


def _is_refusal(answer: str) -> bool:
    normalized = " ".join(answer.lower().split())
    return normalized in {REFUSAL_EN.lower(), REFUSAL_VI.lower()}


_VIETNAMESE_CHARS = set(
    "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    "ĂÂĐÊÔƠƯÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆ"
    "ÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
)


def _intent_instruction(intent: str) -> str:
    if intent == INTENT_SUMMARY:
        return (
            "Provide a short document summary first, then list the key points. "
            "Group related points together and cite the most relevant sources."
        )
    if intent == INTENT_COMPARISON:
        return (
            "Compare the requested items clearly. Use bullets or a small table if useful, "
            "and cite the sources for each comparison point."
        )
    if intent == INTENT_EXPLANATION:
        return (
            "Explain the concept step by step in plain language. Keep it grounded in the "
            "document and cite the relevant sources."
        )
    if intent == INTENT_EXTRACTION:
        return (
            "Extract the requested items as a clean list. Preserve important wording, "
            "numbers, requirements, and citations."
        )
    return (
        "Answer the question directly first, then add brief supporting details and source citations."
    )
