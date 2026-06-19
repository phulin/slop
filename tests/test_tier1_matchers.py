from slop_sftdiv.features import extract_tier1_features, iter_tier1_hits
from slop_sftdiv.features.tier1_matchers import count_tokens


def test_extract_tier1_features_counts_main_feature_families():
    text = """Great question.

**Plan**: Delve into a robust, nuanced, and multifaceted landscape.
- It is not brittle but adaptable and useful.
1. Moreover, this is worth noting: test; inspect... ship!

I hope this helps."""

    features = extract_tier1_features(text)

    assert features.helpers["token_count"] == count_tokens(text)
    assert features.item_counts["stock_openers"]["great_question"] == 1
    assert features.item_counts["stock_closers"]["hope_this_helps"] == 1
    assert features.item_counts["contrastive_negation"]["not_x_but_y"] == 1
    assert features.item_counts["slop_lexicon"]["delve"] == 1
    assert features.item_counts["slop_lexicon"]["robust"] == 1
    assert features.item_counts["slop_lexicon"]["nuanced"] == 1
    assert features.item_counts["slop_lexicon"]["multifaceted"] == 1
    assert "structure" not in features.item_counts
    assert "list_header_bold_lead_in" not in features.counts
    assert features.item_counts["punctuation"]["colon"] >= 1
    assert features.item_counts["punctuation"]["semicolon"] == 1
    assert features.item_counts["punctuation"]["ellipsis"] == 1
    assert features.item_counts["punctuation"]["exclamation"] == 1
    assert features.counts["stock_openers_closers"] == 2
    assert features.rates_per_1k_tokens["slop_lexicon"] > 0
    assert features.helpers["paragraph_count"] == 3
    rhythm_uniformity = features.helpers["paragraph_rhythm_uniformity"]
    assert isinstance(rhythm_uniformity, float)
    assert rhythm_uniformity > 0


def test_contrastive_negation_distinguishes_not_only_from_plain_not_but():
    text = "It is not only fast but also reliable. It is not fragile but steady."

    features = extract_tier1_features(text)

    assert features.item_counts["contrastive_negation"]["not_just_but"] == 1
    assert features.item_counts["contrastive_negation"]["not_x_but_y"] == 1
    assert features.item_counts["contrastive_negation"]["not_x_its_y"] == 0
    assert features.counts["contrastive_negation"] == 2


def test_contrastive_negation_counts_not_x_its_y_frame():
    text = "It is not a shortcut, it is a discipline. It's not noise; it's signal."

    features = extract_tier1_features(text)
    hits = [hit for hit in iter_tier1_hits(text) if hit.feature == "contrastive_negation"]

    assert features.item_counts["contrastive_negation"]["not_x_its_y"] == 2
    assert features.item_counts["contrastive_negation"]["not_x_but_y"] == 0
    assert features.item_counts["contrastive_negation"]["not_just_but"] == 0
    assert {hit.subtype for hit in hits} == {"not_x_its_y"}


def test_empty_text_has_stable_zero_counts_and_denominators():
    features = extract_tier1_features("")

    assert features.helpers["token_count"] == 0
    assert features.helpers["sentence_count"] == 1
    assert features.helpers["paragraph_count"] == 0
    assert all(count == 0 for count in features.counts.values())
    assert all(rate == 0.0 for rate in features.rates_per_1k_tokens.values())


def test_iter_tier1_hits_returns_spans_and_context():
    text = "Great question. It is not just fast but reliable. Delve deeper."

    hits = iter_tier1_hits(text, context_chars=8)

    features = {hit.feature for hit in hits}
    assert {"stock_openers", "contrastive_negation", "slop_lexicon"} <= features
    delve = next(hit for hit in hits if hit.subtype == "delve")
    assert delve.text == "Delve"
    assert text[delve.start : delve.end] == "Delve"
    assert delve.context.startswith("...")


def test_eqbench_slop_score_feature_uses_vendored_lists_and_components():
    text = (
        "This is not just a tool, but a paradigm shift. "
        "He took deep breath with unwavering resolve and meticulous care."
    )

    features = extract_tier1_features(text)
    hits = iter_tier1_hits(text, features=["eqbench_slop_score"])

    assert features.counts["eqbench_slop_score"] > 0
    assert features.item_counts["eqbench_slop_score"]["word_hits"] >= 3
    assert features.item_counts["eqbench_slop_score"]["trigram_hits"] == 1
    assert features.item_counts["eqbench_slop_score"]["contrast_hits"] == 1
    word_rate = features.helpers["eqbench_slop_word_matches_per_1k_words"]
    assert isinstance(word_rate, float)
    assert word_rate > 0
    assert {hit.subtype for hit in hits} >= {
        "word:meticulous",
        "word:paradigm",
        "word:unwavering",
        "trigram:took deep breath",
    }


def test_stock_closer_overall_requires_final_discourse_frame():
    ordinary = extract_tier1_features("The specific materials differ, but the overall process is the same.")
    framed = extract_tier1_features("The details vary. Overall, this improves the system.")

    assert ordinary.item_counts["stock_closers"]["overall"] == 0
    assert framed.item_counts["stock_closers"]["overall"] == 1


def test_rule_of_three_skips_code_and_table_lines():
    text = """Here is code:
```python
read arrays from standard input, computes their valid convolution, and prints the result
```

It supports empathy, clarity, and rigor.
"""

    features = extract_tier1_features(text)

    assert features.item_counts["rule_of_three"]["approx_triple"] == 1
