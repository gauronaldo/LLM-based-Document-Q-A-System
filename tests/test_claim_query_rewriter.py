from app.query_profiles import QueryProfile
from app.query_rewriter import build_retrieval_query, rewrite_claim_query_for_retrieval


def test_claim_rewrite_strips_document_scaffolding() -> None:
    assert (
        rewrite_claim_query_for_retrieval(
            "Does the paper say GitHub Copilot is powered by OpenAI Codex?"
        )
        == "GitHub Copilot is powered by OpenAI Codex"
    )


def test_claim_rewrite_handles_allow_questions_without_paper_terms() -> None:
    assert (
        rewrite_claim_query_for_retrieval(
            "Does the document allow the control group to use internet search and Stack Overflow?"
        )
        == "the control group use internet search and Stack Overflow"
    )


def test_claim_rewrite_keeps_support_anchor() -> None:
    assert (
        rewrite_claim_query_for_retrieval(
            "Does Figure 6 support the claim that treated participants completed the task faster?"
        )
        == "Figure 6 treated participants completed the task faster"
    )


def test_claim_rewrite_does_not_modify_general_yes_no_question() -> None:
    question = "Does this matter for the final answer?"

    assert rewrite_claim_query_for_retrieval(question) == question


def test_build_retrieval_query_rewrites_claim_before_profile_expansion() -> None:
    profile = QueryProfile(
        name="test",
        query_expansions={"control group": ("comparison group",)},
        section_intents={},
    )

    rewritten = build_retrieval_query(
        "Does the paper allow the control group to use internet search?",
        query_profile=profile,
    )

    assert rewritten == (
        "the control group use internet search\n"
        "Related retrieval terms: comparison group"
    )
