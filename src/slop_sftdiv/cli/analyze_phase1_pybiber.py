from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from slop_sftdiv.features.pybiber_full import PYBIBER_FEATURES


@dataclass(frozen=True)
class Phase1PybiberSample:
    name: str
    role: str
    pybiber_csv: Path
    manifest_parquet: Path
    manifest_role: str | None = None


DEFAULT_SELECTED_FEATURES: tuple[str, ...] = (
    "f_01_past_tense",
    "f_03_present_tense",
    "f_06_first_person_pronouns",
    "f_07_second_person_pronouns",
    "f_08_third_person_pronouns",
    "f_14_nominalizations",
    "f_24_infinitives",
    "f_39_prepositions",
    "f_40_adj_attr",
    "f_42_adverbs",
    "f_43_type_token",
    "f_44_mean_word_length",
    "f_47_hedges",
    "f_48_amplifiers",
    "f_49_emphatics",
    "f_51_demonstratives",
    "f_52_modal_possibility",
    "f_53_modal_necessity",
    "f_54_modal_predictive",
    "f_64_phrasal_coordination",
    "f_65_clausal_coordination",
    "f_66_neg_synthetic",
    "f_67_neg_analytic",
)


DEFAULT_SAMPLES: tuple[Phase1PybiberSample, ...] = (
    Phase1PybiberSample(
        name="dolma_pretrain",
        role="pretrain_document",
        pybiber_csv=Path("artifacts/stage1/census/olmo3_dolma3_80k_scan_pybiber_full.csv"),
        manifest_parquet=Path("artifacts/stage1/corpora/olmo3_dolma3_80k_scan_sample_manifest.parquet"),
    ),
    Phase1PybiberSample(
        name="sft_target",
        role="target_response",
        pybiber_csv=Path("artifacts/stage1/census/olmo3_dolci_sft_40k_pybiber_full.csv"),
        manifest_parquet=Path("artifacts/stage1/corpora/olmo3_dolci_sft_40k_sample_manifest.parquet"),
    ),
    Phase1PybiberSample(
        name="dpo_chosen",
        role="chosen",
        pybiber_csv=Path("artifacts/stage1/census/olmo3_dolci_dpo_40k_retained_pybiber_full.csv"),
        manifest_parquet=Path("artifacts/stage1/corpora/olmo3_dolci_dpo_40k_sample_manifest.parquet"),
        manifest_role="chosen",
    ),
    Phase1PybiberSample(
        name="dpo_rejected",
        role="rejected",
        pybiber_csv=Path("artifacts/stage1/census/olmo3_dolci_dpo_40k_retained_pybiber_full.csv"),
        manifest_parquet=Path("artifacts/stage1/corpora/olmo3_dolci_dpo_40k_sample_manifest.parquet"),
        manifest_role="rejected",
    ),
)


COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("sft_minus_dolma", "sft_target", "dolma_pretrain"),
    ("dpo_chosen_minus_rejected", "dpo_chosen", "dpo_rejected"),
    ("dpo_chosen_minus_sft", "dpo_chosen", "sft_target"),
    ("dpo_rejected_minus_sft", "dpo_rejected", "sft_target"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze full pybiber register shifts in the Phase 1 OLMo samples."
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        metavar="NAME,ROLE,PYBIBER_CSV,MANIFEST_PARQUET[,MANIFEST_ROLE]",
        help=(
            "Override/add a sample. Defaults to Phase 1 Dolma, SFT, DPO chosen, "
            "and DPO rejected. MANIFEST_ROLE filters manifest text_role when set."
        ),
    )
    parser.add_argument(
        "--output-means",
        type=Path,
        default=Path("artifacts/stage1/census/phase1_pybiber_register_means.csv"),
    )
    parser.add_argument(
        "--output-deltas",
        type=Path,
        default=Path("artifacts/stage1/census/phase1_pybiber_register_deltas.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("docs/experiments/phase1_pybiber_register_analysis.md"),
    )
    parser.add_argument(
        "--output-intervals",
        type=Path,
        default=Path("artifacts/stage1/census/phase1_pybiber_register_intervals.csv"),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=500,
        help=(
            "Document-bootstrap resamples for selected-feature pybiber intervals. "
            "Set to 0 to skip interval output."
        ),
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--bootstrap-chunk-size",
        type=int,
        default=100,
        help="Number of bootstrap replicates to process per matrix chunk.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of largest absolute deltas to include in the summary per comparison.",
    )
    return parser


def _parse_sample(value: str) -> Phase1PybiberSample:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in {4, 5}:
        raise argparse.ArgumentTypeError(
            "--sample must be NAME,ROLE,PYBIBER_CSV,MANIFEST_PARQUET[,MANIFEST_ROLE]"
        )
    return Phase1PybiberSample(
        name=parts[0],
        role=parts[1],
        pybiber_csv=Path(parts[2]),
        manifest_parquet=Path(parts[3]),
        manifest_role=parts[4] if len(parts) == 5 and parts[4] else None,
    )


def _manifest_with_pybiber_doc_ids(path: Path) -> pd.DataFrame:
    manifest = pd.read_parquet(path, columns=["doc_id", "token_count", "text_role"])
    manifest = manifest.reset_index(drop=True)
    counts = manifest["doc_id"].value_counts()
    duplicate = manifest["doc_id"].map(counts) > 1
    manifest["pybiber_doc_id"] = manifest["doc_id"]
    manifest.loc[duplicate, "pybiber_doc_id"] = manifest.loc[duplicate].apply(
        lambda row: f"{row['doc_id']}#row{int(row.name)}",
        axis=1,
    )
    return manifest


def _joined_sample_frame(sample: Phase1PybiberSample) -> tuple[pd.DataFrame, pd.Series]:
    pybiber = pd.read_csv(sample.pybiber_csv)
    manifest = _manifest_with_pybiber_doc_ids(sample.manifest_parquet)
    if sample.manifest_role is not None:
        manifest = manifest[manifest["text_role"] == sample.manifest_role].copy()
    joined = pybiber.merge(
        manifest[["pybiber_doc_id", "token_count", "text_role"]],
        left_on="doc_id",
        right_on="pybiber_doc_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(manifest):
        raise ValueError(
            f"{sample.name} join returned {len(joined):,} rows for {len(manifest):,} manifest rows"
        )
    weights = joined["token_count"].clip(lower=1)
    return joined, weights


def _weighted_feature_means(sample: Phase1PybiberSample) -> tuple[pd.Series, dict[str, Any]]:
    joined, weights = _joined_sample_frame(sample)
    means = joined[list(PYBIBER_FEATURES)].multiply(weights, axis=0).sum() / weights.sum()
    meta = {
        "sample": sample.name,
        "role": sample.role,
        "rows": int(len(joined)),
        "tokens": int(weights.sum()),
        "pybiber_csv": str(sample.pybiber_csv),
        "manifest_parquet": str(sample.manifest_parquet),
    }
    return means, meta


def _means_frame(samples: list[Phase1PybiberSample]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows = []
    metas = []
    for sample in samples:
        means, meta = _weighted_feature_means(sample)
        metas.append(meta)
        for feature, value in means.items():
            rows.append(
                {
                    "sample": sample.name,
                    "role": sample.role,
                    "feature": feature,
                    "token_weighted_mean": float(value),
                    "rows": meta["rows"],
                    "tokens": meta["tokens"],
                }
            )
    return pd.DataFrame(rows), metas


def _bootstrap_selected_means(
    sample: Phase1PybiberSample,
    *,
    features: tuple[str, ...],
    bootstrap_samples: int,
    seed: int,
    chunk_size: int,
) -> pd.DataFrame:
    if bootstrap_samples <= 0:
        return pd.DataFrame()
    if chunk_size <= 0:
        raise ValueError("--bootstrap-chunk-size must be positive")

    joined, weights = _joined_sample_frame(sample)
    weights_array = weights.to_numpy(dtype=np.float64)
    feature_values = joined[list(features)].to_numpy(dtype=np.float64)
    weighted_values = feature_values * weights_array[:, None]
    sample_size = len(joined)
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    remaining = bootstrap_samples
    probabilities = np.full(sample_size, 1 / sample_size, dtype=np.float64)
    while remaining > 0:
        current = min(chunk_size, remaining)
        counts = rng.multinomial(sample_size, probabilities, size=current)
        numerators = counts @ weighted_values
        denominators = counts @ weights_array
        draws.append(numerators / denominators[:, None])
        remaining -= current

    matrix = np.vstack(draws)
    rows = []
    for feature_index, feature in enumerate(features):
        values = matrix[:, feature_index]
        rows.append(
            {
                "sample": sample.name,
                "role": sample.role,
                "feature": feature,
                "bootstrap_samples": bootstrap_samples,
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def _intervals_frame(
    samples: list[Phase1PybiberSample],
    means: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
    chunk_size: int,
) -> pd.DataFrame:
    if bootstrap_samples <= 0:
        return pd.DataFrame()

    mean_interval_frames = [
        _bootstrap_selected_means(
            sample,
            features=DEFAULT_SELECTED_FEATURES,
            bootstrap_samples=bootstrap_samples,
            seed=seed + index,
            chunk_size=chunk_size,
        )
        for index, sample in enumerate(samples)
    ]
    if not mean_interval_frames:
        return pd.DataFrame()

    mean_intervals = pd.concat(mean_interval_frames, ignore_index=True)
    mean_lookup = means.set_index(["sample", "feature"])["token_weighted_mean"]
    rows = []
    for row in mean_intervals.itertuples(index=False):
        rows.append(
            {
                "statistic": "mean",
                "comparison": "",
                "sample": row.sample,
                "role": row.role,
                "left_sample": "",
                "right_sample": "",
                "feature": row.feature,
                "point_estimate": float(mean_lookup.loc[(row.sample, row.feature)]),
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "bootstrap_samples": row.bootstrap_samples,
            }
        )

    wide_low = mean_intervals.pivot(index="feature", columns="sample", values="ci_low")
    wide_high = mean_intervals.pivot(index="feature", columns="sample", values="ci_high")
    wide_point = means.pivot(index="feature", columns="sample", values="token_weighted_mean")
    for comparison, left, right in COMPARISONS:
        if left not in wide_point.columns or right not in wide_point.columns:
            continue
        for feature in DEFAULT_SELECTED_FEATURES:
            if feature not in wide_point.index:
                continue
            # Conservative endpoint interval; samples are not paired across
            # corpora, so avoid implying a paired bootstrap.
            rows.append(
                {
                    "statistic": "delta",
                    "comparison": comparison,
                    "sample": "",
                    "role": "",
                    "left_sample": left,
                    "right_sample": right,
                    "feature": feature,
                    "point_estimate": float(
                        wide_point.loc[feature, left] - wide_point.loc[feature, right]
                    ),
                    "ci_low": float(wide_low.loc[feature, left] - wide_high.loc[feature, right]),
                    "ci_high": float(wide_high.loc[feature, left] - wide_low.loc[feature, right]),
                    "bootstrap_samples": bootstrap_samples,
                }
            )
    return pd.DataFrame(rows)


def _deltas_frame(means: pd.DataFrame) -> pd.DataFrame:
    wide = means.pivot(index="feature", columns="sample", values="token_weighted_mean")
    rows = []
    for comparison, left, right in COMPARISONS:
        if left not in wide.columns or right not in wide.columns:
            continue
        delta = wide[left] - wide[right]
        for feature, value in delta.items():
            rows.append(
                {
                    "comparison": comparison,
                    "left_sample": left,
                    "right_sample": right,
                    "feature": feature,
                    "delta": float(value),
                    "abs_delta": float(abs(value)),
                    "left_value": float(wide.loc[feature, left]),
                    "right_value": float(wide.loc[feature, right]),
                }
            )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    if frame.empty:
        return ["_No rows._"]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for row in frame[columns].itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_summary(
    *,
    path: Path,
    means: pd.DataFrame,
    deltas: pd.DataFrame,
    intervals: pd.DataFrame,
    metas: list[dict[str, Any]],
    top_n: int,
) -> None:
    selected = means[means["feature"].isin(DEFAULT_SELECTED_FEATURES)].copy()
    selected_wide = (
        selected.pivot(index="feature", columns="sample", values="token_weighted_mean")
        .reindex(DEFAULT_SELECTED_FEATURES)
        .reset_index()
    )
    selected_wide = selected_wide.rename_axis(None, axis=1)

    lines = [
        "# Phase 1 Full Pybiber Register Analysis",
        "",
        "This analysis uses the full 67-feature `pybiber` outputs for the",
        "English-filtered OLMo Phase 1 samples. Means are token-weighted over",
        "retained documents, using sample manifest token counts.",
        "",
        "## Inputs",
        "",
    ]
    for meta in metas:
        lines.append(
            f"- `{meta['sample']}`: {meta['rows']:,} rows, {meta['tokens']:,} tokens, "
            f"`{meta['pybiber_csv']}`"
        )
    lines.extend(
        [
            "",
            "## Selected Token-Weighted Means",
            "",
            *_markdown_table(
                selected_wide.round(3),
                ["feature", "dolma_pretrain", "sft_target", "dpo_chosen", "dpo_rejected"],
            ),
            "",
            "## Largest Register Deltas",
            "",
        ]
    )
    for comparison in [item[0] for item in COMPARISONS]:
        subset = (
            deltas[deltas["comparison"] == comparison]
            .sort_values("abs_delta", ascending=False)
            .head(top_n)
            .copy()
        )
        lines.extend(
            [
                f"### {comparison}",
                "",
                *_markdown_table(
                    subset[["feature", "left_value", "right_value", "delta"]].round(3),
                    ["feature", "left_value", "right_value", "delta"],
                ),
                "",
            ]
        )
    if not intervals.empty:
        mean_intervals = intervals[
            (intervals["statistic"] == "mean")
            & (intervals["sample"].isin(["dolma_pretrain", "sft_target", "dpo_chosen", "dpo_rejected"]))
            & (intervals["feature"].isin(DEFAULT_SELECTED_FEATURES[:10]))
        ].copy()
        mean_intervals["interval"] = mean_intervals.apply(
            lambda row: f"{row['point_estimate']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]",
            axis=1,
        )
        interval_wide = (
            mean_intervals.pivot(index="feature", columns="sample", values="interval")
            .reindex(DEFAULT_SELECTED_FEATURES[:10])
            .reset_index()
        )
        interval_wide = interval_wide.rename_axis(None, axis=1)
        lines.extend(
            [
                "## Selected Bootstrap Intervals",
                "",
                "Nonparametric document-bootstrap intervals over retained rows, preserving",
                "token weighting within each resample. Values are point estimates with",
                "95% intervals for the first selected paper-facing features.",
                "",
                *_markdown_table(
                    interval_wide,
                    ["feature", "dolma_pretrain", "sft_target", "dpo_chosen", "dpo_rejected"],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Substantive Read",
            "",
            "- SFT is far less narrative/personal than the Dolma pretraining sample:",
            "  past tense, first-person pronouns, third-person pronouns, adverbs,",
            "  emphatics, and clausal coordination all drop sharply.",
            "- SFT is more nominalized and expository: nominalizations, attributive",
            "  adjectives, present tense, prepositions, and broad noun density rise.",
            "- DPO does not simply continue SFT in one direction. Both chosen and",
            "  rejected responses differ from SFT on broader register dimensions,",
            "  including more coordination, contractions, nominalizations, and",
            "  analytic negation.",
            "- DPO chosen responses are more descriptive/expository than rejected",
            "  responses on this register layer: chosen is higher on attributive",
            "  adjectives, adverbs, prepositions, and broad noun density.",
            "- DPO rejected responses are more personal/narrative/mental-state framed:",
            "  rejected is higher on first-person pronouns, third-person pronouns,",
            "  past tense, private verbs, infinitives, nominalizations, and analytic",
            "  negation.",
            "- Hedges, amplifiers, and emphatics are higher in the Dolma pretraining",
            "  sample than in SFT. So the register result does not support a generic",
            "  claim that alignment simply adds hedging; it supports a narrower claim",
            "  that alignment data shift toward polished, nominalized, answer-like",
            "  exposition while specific slop lexemes and rhetorical forms move on top",
            "  of that register shift.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    samples = [_parse_sample(item) for item in args.sample] if args.sample else list(DEFAULT_SAMPLES)
    means, metas = _means_frame(samples)
    deltas = _deltas_frame(means)
    intervals = _intervals_frame(
        samples,
        means,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        chunk_size=args.bootstrap_chunk_size,
    )
    args.output_means.parent.mkdir(parents=True, exist_ok=True)
    args.output_deltas.parent.mkdir(parents=True, exist_ok=True)
    means.to_csv(args.output_means, index=False)
    deltas.to_csv(args.output_deltas, index=False)
    if not intervals.empty:
        args.output_intervals.parent.mkdir(parents=True, exist_ok=True)
        intervals.to_csv(args.output_intervals, index=False)
    _write_summary(
        path=args.summary_output,
        means=means,
        deltas=deltas,
        intervals=intervals,
        metas=metas,
        top_n=args.top_n,
    )
    return {
        "samples": len(samples),
        "mean_rows": len(means),
        "delta_rows": len(deltas),
        "interval_rows": len(intervals),
        "summary_output": str(args.summary_output),
    }


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(
        f"Wrote {summary['mean_rows']} means, {summary['delta_rows']} deltas, "
        f"and {summary['interval_rows']} intervals; "
        f"summary: {summary['summary_output']}"
    )


if __name__ == "__main__":
    main()
