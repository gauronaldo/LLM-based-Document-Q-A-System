import unicodedata

from app.text_preprocessor import VietnameseTextPreprocessor
from app.text_splitter import TextSplitter


def test_clean_normalizes_unicode_and_preserves_vietnamese_accents() -> None:
    decomposed = "Sinh vie\u0302n pha\u0309i hoa\u0300n\ntha\u0300nh 65 % ti\u0301n chi\u0309."

    cleaned = VietnameseTextPreprocessor().clean(decomposed)

    assert cleaned == unicodedata.normalize(
        "NFC",
        "Sinh viên phải hoàn thành 65% tín chỉ.",
    )
    assert "viên" in cleaned
    assert "tín chỉ" in cleaned


def test_clean_keeps_paragraph_breaks_but_fixes_wrapped_lines() -> None:
    raw = "Điều 1\nQuy định chung.\n\n\nĐiều 2\nĐiều kiện thực tập."

    cleaned = VietnameseTextPreprocessor().clean(raw)

    assert cleaned == "Điều 1\n\nQuy định chung.\n\nĐiều 2\n\nĐiều kiện thực tập."


def test_clean_preserves_numbered_academic_headings_for_section_splitter() -> None:
    raw = (
        "Previous section final sentence.\n"
        "9\n"
        "Concluding Remarks\n"
        "This paper studies minimum wage effects in Colombia."
    )

    cleaned = VietnameseTextPreprocessor().clean(raw)

    assert cleaned == (
        "Previous section final sentence.\n\n"
        "9 Concluding Remarks\n\n"
        "This paper studies minimum wage effects in Colombia."
    )


def test_clean_preserves_common_data_sources_heading() -> None:
    raw = "Data Sources\nThe paper uses survey and administrative records."

    cleaned = VietnameseTextPreprocessor().clean(raw)

    assert cleaned == "Data Sources\n\nThe paper uses survey and administrative records."


def test_clean_does_not_preserve_numbered_sentences_as_headings() -> None:
    raw = (
        "70. I also show in Table B.2 that estimates are larger for some cities,\n"
        "although this relationship is weaker than the formal sector results."
    )

    cleaned = VietnameseTextPreprocessor().clean(raw)

    assert cleaned == (
        "70. I also show in Table B. 2 that estimates are larger for some cities, "
        "although this relationship is weaker than the formal sector results."
    )


def test_cleaned_pdf_text_keeps_sections_detectable_by_splitter() -> None:
    cleaned = VietnameseTextPreprocessor().clean(
        "Previous section final sentence.\n"
        "9\n"
        "Concluding Remarks\n"
        "This paper studies minimum wage effects in Colombia."
    )

    chunks = TextSplitter(chunk_size=160, chunk_overlap=0).split(
        [{"text": cleaned, "metadata": {"file_id": "paper", "file_name": "paper.pdf", "page": 34}}]
    )

    assert chunks[-1]["metadata"]["section_title"] == "9 Concluding Remarks"


def test_clean_query_uses_same_normalization_rules() -> None:
    query = "  Sinh viên cần bao nhiêu tín chỉ ?  "

    assert VietnameseTextPreprocessor().clean_query(query) == "Sinh viên cần bao nhiêu tín chỉ?"
