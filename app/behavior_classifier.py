"""General expected-behavior classification for document-grounded QA."""

from __future__ import annotations

from app.keyword_search import tokenize


YES_NO_STARTERS = {
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "is",
    "should",
    "was",
    "were",
}

CLAIM_VERIFICATION_MARKERS = {
    "claim",
    "confirm",
    "contradict",
    "evidence",
    "mention",
    "prove",
    "say",
    "state",
    "support",
    "verify",
}


def classify_expected_behavior(question: str) -> str:
    """Classify how the assistant should behave for a user question."""

    tokens = tokenize(question)
    if not tokens:
        return "answer"
    if tokens[0] in YES_NO_STARTERS:
        return "claim_verification"
    if set(tokens) & CLAIM_VERIFICATION_MARKERS:
        return "claim_verification"
    return "answer"


def behavior_instruction(expected_behavior: str) -> str:
    """Return a prompt instruction for the classified behavior."""

    if expected_behavior == "state_not_supported":
        return (
            "The user is asking whether a claim is supported. If the Context does not "
            'support the claim, say "The document does not support this claim." '
            "Then add one brief correction only if the Context provides direct contrary "
            "evidence. Do not guess."
        )
    if expected_behavior == "claim_verification":
        return (
            "If the question asks whether a claim is supported, answer yes/no only when "
            'the Context supports it. If the Context does not support the claim, say "The '
            'document does not support this claim." instead of guessing.'
        )
    return "Answer the user's question only from the provided Context."
