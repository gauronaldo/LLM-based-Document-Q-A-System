from app.query_profiles import QueryProfile
from app.query_rewriter import build_retrieval_query, rewrite_query_for_retrieval


def test_rewrite_keeps_standalone_question() -> None:
    question = "What benefits does the internship offer?"

    assert rewrite_query_for_retrieval(question, [{"role": "user", "content": "Old question"}]) == question


def test_rewrite_short_follow_up_with_previous_user_question() -> None:
    rewritten = rewrite_query_for_retrieval(
        "Explain more",
        [
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "What benefits does the internship offer?"},
        ],
    )

    assert rewritten == "What benefits does the internship offer?\nFollow-up: Explain more"


def test_rewrite_vietnamese_follow_up() -> None:
    rewritten = rewrite_query_for_retrieval(
        "giải thích thêm ý đó",
        [{"role": "user", "content": "Ứng viên cần kỹ năng gì?"}],
    )

    assert rewritten == "Ứng viên cần kỹ năng gì?\nFollow-up: giải thích thêm ý đó"


def test_rewrite_unaccented_vietnamese_follow_up() -> None:
    rewritten = rewrite_query_for_retrieval(
        "giai thich them y do",
        [{"role": "user", "content": "Ung vien can ky nang gi?"}],
    )

    assert rewritten == "Ung vien can ky nang gi?\nFollow-up: giai thich them y do"


def test_rewrite_does_not_hardcode_section_synonyms_without_history() -> None:
    assert rewrite_query_for_retrieval("give me the conclusion") == "give me the conclusion"


def test_rewrite_keeps_vietnamese_standalone_question() -> None:
    question = "Ai là anh em kết nghĩa của Lưu Bị?"

    assert rewrite_query_for_retrieval(
        question,
        [{"role": "user", "content": "Câu hỏi trước"}],
    ) == question


def test_build_retrieval_query_applies_configurable_expansion() -> None:
    profile = QueryProfile(
        name="test",
        query_expansions={"source": ("data", "origin")},
        section_intents={},
    )

    rewritten = build_retrieval_query("Where does the source come from?", query_profile=profile)

    assert rewritten == "Where does the source come from?\nRelated retrieval terms: data, origin"


def test_rewrite_keeps_vietnamese_entity_attribute_question() -> None:
    question = "Các tên gọi khác của Quan Vũ là gì?"

    assert rewrite_query_for_retrieval(
        question,
        [{"role": "user", "content": "Ai là anh em kết nghĩa của Lưu Bị?"}],
    ) == question
