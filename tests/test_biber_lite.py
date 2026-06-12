from slop_sftdiv.features import extract_biber_lite_features


def test_biber_lite_counts_surface_register_features():
    features = extract_biber_lite_features(
        "I think you should carefully consider that the result was reported because it may help."
    )

    assert features.token_count > 0
    assert features.counts["biber_lite_first_person_pronouns"] == 1
    assert features.counts["biber_lite_second_person_pronouns"] == 1
    assert features.counts["biber_lite_necessity_modals"] == 1
    assert features.counts["biber_lite_possibility_modals"] == 1
    assert features.counts["biber_lite_private_verbs"] == 1
    assert features.counts["biber_lite_passive_voice_approx"] == 1
    assert features.rates_per_1k_tokens["biber_lite_private_verbs"] > 0
