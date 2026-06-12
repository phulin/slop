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


def test_iter_feature_opportunities_excludes_unknown_or_deferred_features():
    text = "- Robust tests\nThis is not a shortcut, it is signal."

    opportunities = iter_feature_opportunities(
        text,
        features=["list_header_bold_lead_in", "punctuation_rhythm", "contrastive_negation"],
    )

    assert {item.feature for item in opportunities} == {"contrastive_negation"}
