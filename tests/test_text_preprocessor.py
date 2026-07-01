import unicodedata

from app.text_preprocessor import VietnameseTextPreprocessor


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

    assert cleaned == "Điều 1 Quy định chung.\n\nĐiều 2 Điều kiện thực tập."


def test_clean_query_uses_same_normalization_rules() -> None:
    query = "  Sinh viên cần bao nhiêu tín chỉ ?  "

    assert VietnameseTextPreprocessor().clean_query(query) == "Sinh viên cần bao nhiêu tín chỉ?"
