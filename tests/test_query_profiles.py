from app.query_profiles import QueryProfile, load_query_profile


def test_load_query_profile_falls_back_to_general_for_unknown_name() -> None:
    profile = load_query_profile("missing-profile")

    assert profile.name == "general"
    assert profile.prompt_extension


def test_academic_profile_extends_general_without_core_hardcoding() -> None:
    profile = load_query_profile("academic")

    expanded = profile.expand_query("What is the identification approach?")

    assert profile.name == "academic"
    assert "empirical strategy" in expanded
    assert "methodology" in expanded


def test_section_boost_uses_configured_terms() -> None:
    profile = QueryProfile(
        name="test",
        query_expansions={},
        section_intents={
            "ending": {
                "query_terms": ("ending",),
                "section_terms": ("finale",),
            }
        },
    )

    assert profile.section_boost(
        "What is the ending?",
        "Finale",
        "The ending is clear.",
        ["ending"],
    ) == 0.72


def test_academic_profile_scores_result_evidence_from_config() -> None:
    profile = load_query_profile("academic")

    boost = profile.evidence_boost(
        "What does the paper find for wages?",
        "Results",
        "The results show wage effects and estimates around the minimum wage.",
    )

    assert boost > 0.08


def test_academic_profile_scores_measurement_evidence_from_config() -> None:
    profile = load_query_profile("academic")

    boost = profile.evidence_boost(
        "What information creates variation in incidence?",
        "Empirical Strategy",
        "Incidence is measured as the fraction affected between the old and new threshold.",
    )

    assert boost > 0.1
