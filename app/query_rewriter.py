"""Rewrite follow-up questions into retrieval-friendly standalone queries."""

from __future__ import annotations

import re


FOLLOW_UP_MARKERS = {
    "it",
    "that",
    "this",
    "them",
    "they",
    "more",
    "explain more",
    "\u00fd \u0111\u00f3",
    "\u00fd n\u00e0y",
    "ph\u1ea7n \u0111\u00f3",
    "m\u1ee5c \u0111\u00f3",
    "gi\u1ea3i th\u00edch th\u00eam",
    "n\u00f3i r\u00f5 h\u01a1n",
    "y do",
    "y nay",
    "phan do",
    "muc do",
    "giai thich them",
    "noi ro hon",
}

STANDALONE_STARTERS = {
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "which",
    "list",
    "summarize",
    "extract",
    "compare",
    "ai",
    "các",
    "gì",
    "tên",
    "cac",
    "gi",
    "ten",
    "n\u00eau",
    "li\u1ec7t",
    "t\u00f3m",
    "so",
    "neu",
    "liet",
    "tom",
}
WORD_PATTERN = re.compile(r"[\w\u00c0-\u1ef9]+", re.UNICODE)


def rewrite_query_for_retrieval(
    question: str,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """Return a standalone retrieval query using recent chat context when useful."""

    clean_question = question.strip()
    if not clean_question or not chat_history:
        return clean_question

    lowered = clean_question.lower()
    if not _looks_like_follow_up(lowered):
        return clean_question

    previous_user_question = _last_user_message(chat_history)
    if not previous_user_question:
        return clean_question

    return f"{previous_user_question}\nFollow-up: {clean_question}"


def _looks_like_follow_up(lowered_question: str) -> bool:
    tokens = WORD_PATTERN.findall(lowered_question)
    if not tokens:
        return False
    single_word_markers = {marker for marker in FOLLOW_UP_MARKERS if " " not in marker}
    phrase_markers = FOLLOW_UP_MARKERS - single_word_markers

    if any(marker in tokens for marker in single_word_markers):
        return True
    if any(marker in lowered_question for marker in phrase_markers):
        return True
    if tokens[0] in STANDALONE_STARTERS:
        return False
    if len(tokens) <= 3:
        return True
    return False


def _last_user_message(chat_history: list[dict[str, str]]) -> str | None:
    for message in reversed(chat_history):
        if message.get("role") == "user" and message.get("content"):
            return message["content"].strip()
    return None
