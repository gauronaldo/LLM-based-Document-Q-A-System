"""Lightweight intent detection for document-grounded chat."""

from __future__ import annotations

from dataclasses import dataclass


INTENT_QA = "qa"
INTENT_SUMMARY = "summary"
INTENT_EXPLANATION = "explanation"
INTENT_COMPARISON = "comparison"
INTENT_EXTRACTION = "extraction"

VIETNAMESE_MARKERS = set(
    "\u0103\u00e2\u0111\u00ea\u00f4\u01a1\u01b0"
    "\u00e1\u00e0\u1ea3\u00e3\u1ea1\u1ea5\u1ea7\u1ea9\u1eab\u1ead"
    "\u1eaf\u1eb1\u1eb3\u1eb5\u1eb7\u00e9\u00e8\u1ebb\u1ebd\u1eb9"
    "\u1ebf\u1ec1\u1ec3\u1ec5\u1ec7\u00ed\u00ec\u1ec9\u0129\u1ecb"
    "\u00f3\u00f2\u1ecf\u00f5\u1ecd\u1ed1\u1ed3\u1ed5\u1ed7\u1ed9"
    "\u1edb\u1edd\u1edf\u1ee1\u1ee3\u00fa\u00f9\u1ee7\u0169\u1ee5"
    "\u1ee9\u1eeb\u1eed\u1eef\u1ef1\u00fd\u1ef3\u1ef7\u1ef9\u1ef5"
)

COMMON_VI_WORDS = {
    "sinh",
    "vien",
    "can",
    "bao",
    "nhieu",
    "tai",
    "lieu",
    "trong",
    "khong",
    "duoc",
}

SUMMARY_KEYWORDS = {
    "summarize",
    "summary",
    "summarise",
    "overview",
    "brief",
    "key points",
    "main points",
    "tom tat",
    "tong quan",
    "tong ket",
    "t\u00f3m t\u1eaft",
    "t\u1ed5ng quan",
    "t\u1ed5ng k\u1ebft",
}

EXPLANATION_KEYWORDS = {
    "explain",
    "explanation",
    "why",
    "how",
    "meaning",
    "giai thich",
    "tai sao",
    "nhu the nao",
    "gi\u1ea3i th\u00edch",
    "t\u1ea1i sao",
    "nh\u01b0 th\u1ebf n\u00e0o",
}

COMPARISON_KEYWORDS = {
    "compare",
    "comparison",
    "difference",
    "different",
    "versus",
    "vs",
    "so sanh",
    "khac nhau",
    "so s\u00e1nh",
    "kh\u00e1c nhau",
}

EXTRACTION_KEYWORDS = {
    "list",
    "extract",
    "requirements",
    "conditions",
    "steps",
    "bullet",
    "liet ke",
    "dieu kien",
    "cac buoc",
    "li\u1ec7t k\u00ea",
    "\u0111i\u1ec1u ki\u1ec7n",
    "c\u00e1c b\u01b0\u1edbc",
}


@dataclass(frozen=True)
class RetrievalPlan:
    """Retrieval overrides chosen from the user's intent."""

    top_k: int | None = None
    similarity_threshold: float | None = None


def detect_query_intent(question: str) -> str:
    """Detect the broad type of user request using transparent rules."""

    lowered = question.lower()
    if _contains_any(lowered, SUMMARY_KEYWORDS):
        return INTENT_SUMMARY
    if _contains_any(lowered, COMPARISON_KEYWORDS):
        return INTENT_COMPARISON
    if _contains_any(lowered, EXPLANATION_KEYWORDS):
        return INTENT_EXPLANATION
    if _contains_any(lowered, EXTRACTION_KEYWORDS):
        return INTENT_EXTRACTION
    return INTENT_QA


def is_vietnamese_query(question: str) -> bool:
    """Return whether the query likely uses Vietnamese."""

    lowered = question.lower()
    if any(character in VIETNAMESE_MARKERS for character in lowered):
        return True

    normalized = (
        lowered.replace("\u0111", "d")
        .replace("\u01a1", "o")
        .replace("\u01b0", "u")
    )
    words = set(normalized.split())
    return bool(words & COMMON_VI_WORDS)


def retrieval_plan_for_intent(intent: str, default_top_k: int, default_threshold: float) -> RetrievalPlan:
    """Return retrieval settings that match the request type."""

    if intent == INTENT_SUMMARY:
        return RetrievalPlan(top_k=max(default_top_k, 8), similarity_threshold=0.0)
    if intent in {INTENT_COMPARISON, INTENT_EXPLANATION, INTENT_EXTRACTION}:
        return RetrievalPlan(top_k=max(default_top_k, 7), similarity_threshold=min(default_threshold, 0.2))
    return RetrievalPlan()


def _contains_any(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)
