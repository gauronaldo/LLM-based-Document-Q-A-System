"""Prompt construction for a conversational, grounded document chatbot."""

from __future__ import annotations

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


def build_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as source-labelled context."""

    context_blocks = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})
        page = metadata.get("page", "N/A")
        file_name = metadata.get("file_name", "unknown")
        chunk_id = chunk.get("chunk_id", "unknown")
        text = chunk.get("text", "")

        context_blocks.append(
            "\n".join(
                [
                    f"[{index}]",
                    f"File: {file_name}",
                    f"Page: {page}",
                    f"Chunk ID: {chunk_id}",
                    f"Content: {text}",
                ]
            )
        )

    return "\n\n".join(context_blocks)


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
- Use only the Context and Conversation History below.
- Synthesize the answer in your own words; do not simply copy long raw passages.
- Do not invent details, numbers, names, requirements, or dates.
- If the Context is insufficient, say exactly:
  - "{REFUSAL_VI}" for Vietnamese questions.
  - "{REFUSAL_EN}" for English questions.
- If the user asks in Vietnamese, answer in Vietnamese.
- If the user asks in English, answer in English.
- Cite supporting evidence with compact citation labels like [1], [2].
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
