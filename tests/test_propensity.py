from slop_sftdiv.propensity import iter_feature_opportunities


def test_iter_feature_opportunities_marks_reference_initiations():
    text = (
        "Great question. It is not a shortcut, it is a discipline. "
        "Delve into robust tests. I hope this helps."
    )

    opportunities = iter_feature_opportunities(
        text,
        features=[
            "contrastive_negation",
            "slop_lexicon",
            "stock_openers",
            "stock_closers",
            "stock_openers_closers",
        ],
    )

    by_feature = {}
    for opportunity in opportunities:
        by_feature.setdefault(opportunity.feature, []).append(opportunity)

    assert by_feature["stock_openers"][0].char_offset == 0
    assert by_feature["stock_openers"][0].reference_initiates
    assert any(item.reference_initiates for item in by_feature["contrastive_negation"])
    assert any(item.reference_initiates for item in by_feature["slop_lexicon"])
    assert any(item.reference_initiates for item in by_feature["stock_closers"])
    assert sum(item.reference_initiates for item in by_feature["stock_openers_closers"]) == 2


def test_iter_feature_opportunities_can_cap_token_start_opportunities():
    text = "Delve into robust nuanced tests."

    opportunities = iter_feature_opportunities(
        text,
        features=["slop_lexicon"],
        max_token_start_opportunities=2,
    )

    assert len(opportunities) == 2
    assert [item.char_offset for item in opportunities] == [0, 6]


def test_iter_feature_opportunities_marks_neutral_controls():
    text = "For example, tests check behavior. Such as edge cases. As a result, bugs surface."

    opportunities = iter_feature_opportunities(
        text,
        features=[
            "neutral_for_example",
            "neutral_such_as",
            "neutral_as_a_result",
            "neutral_controls",
        ],
        max_token_start_opportunities=16,
    )

    by_feature = {}
    for opportunity in opportunities:
        by_feature.setdefault(opportunity.feature, []).append(opportunity)

    assert any(item.reference_initiates for item in by_feature["neutral_for_example"])
    assert any(item.reference_initiates for item in by_feature["neutral_such_as"])
    assert any(item.reference_initiates for item in by_feature["neutral_as_a_result"])
    assert sum(item.reference_initiates for item in by_feature["neutral_controls"]) == 3


def test_iter_feature_opportunities_excludes_unknown_or_deferred_features():
    text = "- Robust tests\nThis is not a shortcut, it is signal."

    opportunities = iter_feature_opportunities(
        text,
        features=["list_header_bold_lead_in", "punctuation_rhythm", "contrastive_negation"],
    )

    assert {item.feature for item in opportunities} == {"contrastive_negation"}


def test_iter_feature_opportunities_skips_tier1_hits_for_neutral_only(monkeypatch):
    def fail_iter_tier1_hits(*_args, **_kwargs):
        raise AssertionError("tier1 hits should not be needed for neutral-only extraction")

    monkeypatch.setattr("slop_sftdiv.propensity.iter_tier1_hits", fail_iter_tier1_hits)

    opportunities = iter_feature_opportunities(
        "For example, tests check behavior.",
        features=["neutral_controls"],
        max_token_start_opportunities=8,
    )

    assert any(item.reference_initiates for item in opportunities)
