"""Post-process generated answers without relying on document-specific rules."""

from __future__ import annotations

import re
from typing import Literal

from app.context_support import NO_SUPPORT, SupportLevel
from app.prompt_template import REFUSAL_EN, REFUSAL_VI, is_vietnamese_query


ExpectedBehavior = Literal["answer", "claim_verification", "refuse", "state_not_supported"]


def postprocess_answer_behavior(
    answer: str,
    question: str,
    support_level: SupportLevel,
    expected_behavior: str = "answer",
) -> str:
    """Normalize refusal, unsupported-claim, and mixed answer behavior."""

    if not answer:
        return answer

    if expected_behavior == "refuse" and support_level == NO_SUPPORT:
        return canonical_refusal(question)

    if expected_behavior == "state_not_supported":
        return normalize_unsupported_claim_answer(answer, question)

    if contains_refusal_phrase(answer):
        if support_level == NO_SUPPORT:
            return canonical_refusal(question)
        cleaned = remove_refusal_sentences(answer)
        if contains_answer_content(cleaned):
            return cleaned

    if looks_like_refusal(answer) and support_level == NO_SUPPORT:
        return canonical_refusal(question)

    return answer


def canonical_refusal(question: str) -> str:
    return REFUSAL_VI if is_vietnamese_query(question) else REFUSAL_EN


def normalize_unsupported_claim_answer(answer: str, question: str) -> str:
    """Ensure unsupported-claim answers start with a clear claim-support statement."""

    cleaned = _strip_citations(answer).strip()
    prefix = _unsupported_prefix(question)
    if _starts_with_unsupported_statement(cleaned):
        return _strip_wrapping_parentheses(cleaned)

    if looks_like_refusal(cleaned):
        return prefix

    if cleaned.lower().startswith(("no.", "no,", "no ")):
        cleaned = re.sub(r"^no[.,]?\s*", "", cleaned, flags=re.IGNORECASE).strip()

    if not cleaned:
        return prefix
    return f"{prefix} {cleaned}"


def looks_like_refusal(answer: str) -> bool:
    normalized = _normalize_for_matching(answer)
    if not normalized:
        return False

    refusal_markers = (
        REFUSAL_EN.lower(),
        REFUSAL_VI.lower(),
        "i could not find",
        "i did not find",
        "i do not find",
        "i was unable to find",
        "i cannot find",
        "no information is provided",
        "not enough information",
        "insufficient information",
        "the context does not mention",
        "the document does not mention",
        "the provided document does not mention",
        "not found in the document",
        "there is no evidence in the context",
        "there is no mention of",
        "the retrieved context does not contain",
        "the information is not available in the provided context",
        "toi khong tim thay",
        "khong tim thay",
        "khong co thong tin",
        "khong du thong tin",
        "tôi không tìm thấy",
        "không tìm thấy",
        "không có thông tin",
        "không đủ thông tin",
    )
    return any(marker in normalized for marker in refusal_markers)


def contains_refusal_phrase(answer: str) -> bool:
    return looks_like_refusal(answer)


def contains_answer_content(answer: str) -> bool:
    non_refusal_parts = [part for part in _answer_units(answer) if not looks_like_refusal(part)]
    if not non_refusal_parts:
        return False
    tokens = re.findall(
        r"[\w\u00c0-\u1ef9]+",
        _strip_citations(" ".join(non_refusal_parts)),
        flags=re.UNICODE,
    )
    return len(tokens) >= 6


def remove_refusal_sentences(answer: str) -> str:
    """Remove no-information sentences while preserving useful answer sentences."""

    parts = _answer_units(answer)
    kept = [part.strip() for part in parts if part.strip() and not looks_like_refusal(part)]
    if kept:
        return " ".join(kept).strip()
    return _strip_citations(answer).strip()


def _answer_units(answer: str) -> list[str]:
    units: list[str] = []
    for paragraph in re.split(r"\n+", answer.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        pattern = re.compile(
            r".+?[.!?](?:\s*\[\d+(?:\s*,\s*\d+)*\])?|.+$",
            flags=re.UNICODE,
        )
        units.extend(match.group(0).strip() for match in pattern.finditer(paragraph) if match.group(0).strip())
    return units or [answer.strip()]


def _unsupported_prefix(question: str) -> str:
    if is_vietnamese_query(question):
        return "Tài liệu không hỗ trợ nhận định này."
    return "The document does not support this claim."


def _starts_with_unsupported_statement(answer: str) -> bool:
    normalized = _normalize_for_matching(answer)
    return normalized.startswith(
        (
            "the document does not support this claim",
            "the provided document does not support this claim",
            "tai lieu khong ho tro nhan dinh nay",
            "tài liệu không hỗ trợ nhận định này",
        )
    )


def _strip_citations(text: str) -> str:
    return re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", text).strip()


def _strip_wrapping_parentheses(answer: str) -> str:
    if answer.startswith("(") and answer.endswith(")"):
        return answer[1:-1].strip()
    return answer


def _normalize_for_matching(text: str) -> str:
    normalized = _strip_citations(" ".join(text.lower().split()))
    replacements = {
        "couldn't": "could not",
        "didn't": "did not",
        "don't": "do not",
        "can't": "cannot",
        "isn't": "is not",
        "doesn't": "does not",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized.strip(". ")
