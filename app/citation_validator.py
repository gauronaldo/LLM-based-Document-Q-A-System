"""Post-check compact numeric citations against retrieved evidence."""

from __future__ import annotations

import re
from typing import Any

from app.keyword_search import tokenize


CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]|\n|$)", re.UNICODE)


def post_check_citations(
    answer: str,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """Repair weak citations and add missing citations when support is clear."""

    if not answer:
        return answer
    if _looks_like_no_information_answer(answer):
        return _strip_citations(answer)
    if not retrieved_chunks:
        return answer

    repaired = answer
    for sentence_match in SENTENCE_PATTERN.finditer(answer):
        sentence = sentence_match.group(0)
        if not sentence:
            continue
        if _looks_like_no_information_answer(sentence):
            repaired = repaired.replace(sentence, _strip_citations(sentence), 1)
            continue
        citations = _citation_ids(sentence)
        trailing_citations = _following_citation_ids(answer, sentence_match.end())
        citations.extend(trailing_citations)
        best_id, best_score = _best_supporting_source(sentence, question, retrieved_chunks)
        if not citations:
            if _should_add_missing_citation(sentence, best_id, best_score):
                repaired = repaired.replace(sentence, _append_sentence_citation(sentence, best_id), 1)
            continue

        if best_id is None or best_score < 0.12:
            continue

        for citation_id in citations:
            if citation_id < 1 or citation_id > len(retrieved_chunks):
                repaired = repaired.replace(f"[{citation_id}]", f"[{best_id}]", 1)
                continue
            current_score = _support_score(sentence, question, retrieved_chunks[citation_id - 1])
            if best_id != citation_id and best_score >= current_score + 0.15:
                repaired = repaired.replace(f"[{citation_id}]", f"[{best_id}]", 1)

    return repaired


def _citation_ids(text: str) -> list[int]:
    ids: list[int] = []
    for match in CITATION_PATTERN.finditer(text):
        ids.extend(int(value.strip()) for value in match.group(1).split(",") if value.strip().isdigit())
    return ids


def _strip_citations(text: str) -> str:
    return re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", text).strip()


def _looks_like_no_information_answer(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    without_citations = CITATION_PATTERN.sub("", normalized).strip(". ")
    markers = (
        "i could not find this information",
        "i could not find information",
        "i couldn't find this information",
        "i couldn't find information",
        "i did not find this information",
        "i did not find information",
        "i didn't find this information",
        "i didn't find information",
        "i cannot find this information",
        "i cannot find information",
        "i can't find this information",
        "i can't find information",
        "the provided document does not mention",
        "the document does not mention",
        "the context does not mention",
        "there is no mention of",
        "could not find this information in the provided document",
        "not enough information",
        "insufficient information",
        "tôi không tìm thấy thông tin này",
        "toi khong tim thay thong tin nay",
        "không tìm thấy thông tin này",
        "khong tim thay thong tin nay",
    )
    if any(marker in without_citations for marker in markers):
        return True

    no_info_patterns = (
        r"^(no,\s*)?i\s+(did not|didn't|do not|don't|could not|couldn't|cannot|can't)\s+"
        r"(find|locate|see)\s+(this\s+)?(information|evidence|answer|details)\s*"
        r"(in|within)\s+(the\s+)?(document|paper|report|article|context|provided document|provided context)\b",
        r"^(the\s+)?(document|paper|report|article|context|provided document|provided context)\s+"
        r"(does not|doesn't|do not|don't)\s+"
        r"(address|mention|estimate|use|study|discuss|provide|include|contain|cover)\b",
        r"\b(not found|not available|not provided|not discussed|not mentioned)\s+"
        r"(in|within)\s+(the\s+)?(document|paper|report|article|context|provided document|provided context)\b",
    )
    return any(re.search(pattern, without_citations) for pattern in no_info_patterns)


def _following_citation_ids(answer: str, position: int) -> list[int]:
    following = answer[position:]
    match = re.match(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]", following)
    if not match:
        return []
    return [
        int(value.strip())
        for value in match.group(1).split(",")
        if value.strip().isdigit()
    ]


def _best_supporting_source(
    sentence: str,
    question: str,
    chunks: list[dict[str, Any]],
) -> tuple[int | None, float]:
    best_id = None
    best_score = 0.0
    for index, chunk in enumerate(chunks, start=1):
        score = _support_score(sentence, question, chunk)
        if score > best_score:
            best_id = index
            best_score = score
    return best_id, best_score


def _support_score(sentence: str, question: str, chunk: dict[str, Any]) -> float:
    sentence_terms = _content_terms(sentence)
    if not sentence_terms:
        sentence_terms = _content_terms(question)
    if not sentence_terms:
        return 0.0

    metadata = chunk.get("metadata", {})
    chunk_text = " ".join(
        str(part)
        for part in (
            metadata.get("section_title", ""),
            chunk.get("matched_text", ""),
            chunk.get("text", ""),
        )
        if part
    )
    chunk_terms = set(tokenize(chunk_text))
    if not chunk_terms:
        return 0.0

    sentence_weight = _weighted_term_coverage(sentence_terms, chunk_terms)
    question_terms = _content_terms(question)
    question_weight = _weighted_term_coverage(question_terms, chunk_terms) if question_terms else 0.0
    phrase_bonus = _phrase_support_bonus(sentence, chunk_text)
    return min(1.0, 0.75 * sentence_weight + 0.15 * question_weight + phrase_bonus)


def _weighted_term_coverage(terms: list[str], chunk_terms: set[str]) -> float:
    if not terms:
        return 0.0

    total_weight = 0.0
    matched_weight = 0.0
    for term in terms:
        weight = _term_weight(term)
        total_weight += weight
        if term in chunk_terms:
            matched_weight += weight
    return matched_weight / total_weight if total_weight else 0.0


def _term_weight(term: str) -> float:
    if any(character.isdigit() for character in term):
        return 2.0
    if len(term) >= 10:
        return 1.8
    if len(term) >= 7:
        return 1.4
    return 1.0


def _phrase_support_bonus(sentence: str, chunk_text: str) -> float:
    sentence_phrases = _content_phrases(sentence)
    if not sentence_phrases:
        return 0.0

    normalized_chunk = " ".join(tokenize(chunk_text))
    hits = sum(1 for phrase in sentence_phrases if phrase in normalized_chunk)
    return min(0.1, 0.05 * hits)


def _content_phrases(text: str) -> list[str]:
    terms = _content_terms(text)
    phrases = []
    for index in range(len(terms) - 1):
        left, right = terms[index], terms[index + 1]
        if len(left) >= 5 or len(right) >= 5:
            phrases.append(f"{left} {right}")
    return phrases


def _should_add_missing_citation(sentence: str, best_id: int | None, best_score: float) -> bool:
    if best_id is None or best_score < 0.5:
        return False
    terms = _content_terms(sentence)
    if len(terms) < 4:
        return False
    if _looks_like_heading_or_transition(sentence):
        return False
    return True


def _append_sentence_citation(sentence: str, citation_id: int) -> str:
    trailing_space = ""
    while sentence.endswith(" "):
        trailing_space += " "
        sentence = sentence[:-1]

    newline = ""
    if sentence.endswith("\n"):
        newline = "\n"
        sentence = sentence[:-1]

    stripped = sentence.rstrip()
    punctuation = ""
    if stripped and stripped[-1] in ".!?":
        punctuation = stripped[-1]
        stripped = stripped[:-1].rstrip()

    return f"{stripped} [{citation_id}]{punctuation}{newline}{trailing_space}"


def _looks_like_heading_or_transition(sentence: str) -> bool:
    stripped = sentence.strip()
    if not stripped:
        return True
    lowered = stripped.lower().strip(":")
    if lowered in {"sources", "retrieved evidence", "evidence", "answer", "summary"}:
        return True
    if stripped.endswith(":") and len(_content_terms(stripped)) <= 5:
        return True
    return False


def _content_terms(text: str) -> list[str]:
    return [
        token
        for token in tokenize(CITATION_PATTERN.sub("", text))
        if len(token) > 2 and token not in _STOPWORDS
    ]


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "according",
    "provided",
    "document",
    "paper",
}
