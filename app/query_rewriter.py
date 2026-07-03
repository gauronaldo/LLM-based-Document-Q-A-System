"""Rewrite follow-up questions into retrieval-friendly standalone queries."""

from __future__ import annotations

import re

from app.query_profiles import QueryProfile, load_query_profile


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
DOCUMENT_SUBJECT_PATTERN = (
    r"(?:the\s+)?(?:paper|document|article|study|report|text|source|authors?|researchers?)"
)
REPORTING_VERB_PATTERN = (
    r"(?:say|state|claim|argue|mention|report|show|find|conclude|suggest|"
    r"allow|evaluate|use|study|estimate|discuss|describe|indicate|explain)"
)
CLAIM_PREFIX_PATTERNS = [
    re.compile(
        rf"^\s*(?:does|do|did)\s+{DOCUMENT_SUBJECT_PATTERN}\s+"
        rf"{REPORTING_VERB_PATTERN}\s+(?:that\s+)?",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^\s*(?:is|are|was|were)\s+it\s+"
        rf"(?:true|reported|shown|stated|claimed|argued|suggested)\s+(?:that\s+)?",
        re.IGNORECASE,
    ),
]
SUPPORT_CLAIM_PATTERN = re.compile(
    r"^\s*(?:does|do|did)\s+(.+?)\s+support\s+(?:the\s+)?(?:claim|idea|statement)\s+that\s+",
    re.IGNORECASE,
)


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


def build_retrieval_query(
    question: str,
    chat_history: list[dict[str, str]] | None = None,
    query_profile: QueryProfile | None = None,
) -> str:
    """Rewrite follow-ups and apply configurable retrieval-only term expansion."""

    rewritten = rewrite_query_for_retrieval(question, chat_history)
    rewritten = rewrite_claim_query_for_retrieval(rewritten)
    profile = query_profile or load_query_profile()
    return profile.expand_query(rewritten)


def rewrite_claim_query_for_retrieval(question: str) -> str:
    """Strip generic yes/no scaffolding so retrieval focuses on the claim content."""

    clean_question = question.strip()
    if not clean_question:
        return clean_question

    query = clean_question.rstrip(" ?")
    support_match = SUPPORT_CLAIM_PATTERN.match(query)
    if support_match:
        evidence_anchor = support_match.group(1).strip()
        claim = query[support_match.end() :].strip()
        return _clean_claim_query(f"{evidence_anchor} {claim}") or clean_question

    for pattern in CLAIM_PREFIX_PATTERNS:
        match = pattern.match(query)
        if match:
            claim = query[match.end() :].strip()
            return _clean_claim_query(claim) or clean_question

    return clean_question


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


def _clean_claim_query(claim: str) -> str:
    claim = re.sub(r"^\s*(?:that|whether|if)\s+", "", claim, flags=re.IGNORECASE)
    claim = re.sub(r"\bto\s+use\b", "use", claim, flags=re.IGNORECASE)
    claim = re.sub(r"\s+", " ", claim).strip(" .?")
    return claim
