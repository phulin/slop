from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


PYBIBER_FEATURES: tuple[str, ...] = (
    "f_01_past_tense",
    "f_02_perfect_aspect",
    "f_03_present_tense",
    "f_04_place_adverbials",
    "f_05_time_adverbials",
    "f_06_first_person_pronouns",
    "f_07_second_person_pronouns",
    "f_08_third_person_pronouns",
    "f_09_pronoun_it",
    "f_10_demonstrative_pronoun",
    "f_11_indefinite_pronouns",
    "f_12_proverb_do",
    "f_13_wh_question",
    "f_14_nominalizations",
    "f_15_gerunds",
    "f_16_other_nouns",
    "f_17_agentless_passives",
    "f_18_by_passives",
    "f_19_be_main_verb",
    "f_20_existential_there",
    "f_21_that_verb_comp",
    "f_22_that_adj_comp",
    "f_23_wh_clause",
    "f_24_infinitives",
    "f_25_present_participle",
    "f_26_past_participle",
    "f_27_past_participle_whiz",
    "f_28_present_participle_whiz",
    "f_29_that_subj",
    "f_30_that_obj",
    "f_31_wh_subj",
    "f_32_wh_obj",
    "f_33_pied_piping",
    "f_34_sentence_relatives",
    "f_35_because",
    "f_36_though",
    "f_37_if",
    "f_38_other_adv_sub",
    "f_39_prepositions",
    "f_40_adj_attr",
    "f_41_adj_pred",
    "f_42_adverbs",
    "f_43_type_token",
    "f_44_mean_word_length",
    "f_45_conjuncts",
    "f_46_downtoners",
    "f_47_hedges",
    "f_48_amplifiers",
    "f_49_emphatics",
    "f_50_discourse_particles",
    "f_51_demonstratives",
    "f_52_modal_possibility",
    "f_53_modal_necessity",
    "f_54_modal_predictive",
    "f_55_verb_public",
    "f_56_verb_private",
    "f_57_verb_suasive",
    "f_58_verb_seem",
    "f_59_contractions",
    "f_60_that_deletion",
    "f_61_stranded_preposition",
    "f_62_split_infinitive",
    "f_63_split_auxiliary",
    "f_64_phrasal_coordination",
    "f_65_clausal_coordination",
    "f_66_neg_synthetic",
    "f_67_neg_analytic",
)

PYBIBER_RATE_FEATURES: frozenset[str] = frozenset(PYBIBER_FEATURES) - {
    "f_43_type_token",
    "f_44_mean_word_length",
}


@dataclass(frozen=True)
class PybiberFullFeatures:
    """Full pybiber feature vectors for one document."""

    doc_id: str
    counts: dict[str, float]
    rates_or_values: dict[str, float]


def extract_pybiber_full_features(
    text: str,
    *,
    doc_id: str = "doc",
    model: str = "en_core_web_sm",
) -> PybiberFullFeatures:
    """Extract full pybiber features for one text.

    This is convenient for tests and small probes. Use the CLI batch path for
    real corpora/generation caches so spaCy parsing is batched.
    """

    normalized_rows, count_rows = cast(
        tuple[list[dict[str, Any]], list[dict[str, Any]]],
        run_pybiber_full(
            [{"doc_id": doc_id, "text": text}],
            model=model,
            return_counts=True,
        ),
    )
    normalized = normalized_rows[0]
    counts = count_rows[0]
    return PybiberFullFeatures(
        doc_id=doc_id,
        counts={feature: float(counts.get(feature, 0.0)) for feature in PYBIBER_FEATURES},
        rates_or_values={
            feature: float(normalized.get(feature, 0.0)) for feature in PYBIBER_FEATURES
        },
    )


def run_pybiber_full(
    records: list[dict[str, Any]],
    *,
    model: str = "en_core_web_sm",
    n_process: int = 1,
    batch_size: int = 25,
    return_counts: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run full pybiber extraction over records with ``doc_id`` and ``text`` keys."""

    try:
        import polars as pl
        import pybiber as pb
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "Full pybiber extraction requires optional dependencies: pybiber, polars, spacy"
        ) from exc

    try:
        nlp = spacy.load(model, disable=["ner"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model {model!r} is required for full pybiber extraction. "
            f"Install it with: python -m spacy download {model}"
        ) from exc

    corpus = pl.DataFrame(
        {
            "doc_id": [str(record["doc_id"]) for record in records],
            "text": [str(record["text"]) for record in records],
        }
    )
    parsed = pb.spacy_parse(
        corpus,
        nlp,
        n_process=n_process,
        batch_size=batch_size,
        disable_ner=True,
    )
    normalized = pb.biber(parsed, normalize=True).to_dicts()
    if not return_counts:
        return normalized
    counts = pb.biber(parsed, normalize=False).to_dicts()
    return normalized, counts
