from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


EXCLUDED_REVISED_PHASE1_FEATURES = {
    "generic_hedging",
    # Legacy artifact guard: older stage-1 tables may still contain this retired
    # formatting feature, but new matcher runs no longer emit it.
    "list_header_bold_lead_in",
    "punctuation_rhythm",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble revised Phase 1 census artifacts.")
    parser.add_argument(
        "--feature-rates",
        action="append",
        required=True,
        type=Path,
        help="Feature-rate CSV from slop-census. Repeat for multiple corpora.",
    )
    parser.add_argument("--pair-deltas", type=Path, required=True)
    parser.add_argument(
        "--output-feature-corpus",
        type=Path,
        default=Path("artifacts/stage1/census/feature_rates_by_corpus.parquet"),
    )
    parser.add_argument(
        "--output-feature-stratum",
        type=Path,
        default=Path("artifacts/stage1/census/feature_rates_by_stratum.parquet"),
    )
    parser.add_argument(
        "--output-pair-deltas",
        type=Path,
        default=Path("artifacts/stage1/census/preference_pair_deltas.parquet"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("artifacts/stage1/census/census_summary.md"),
    )
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase1-census")
    parser.add_argument("--wandb-job-type", default="assemble-phase1")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _read_feature_rates(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["input_path"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _aggregate_rates(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = (
        frame.groupby(group_columns, dropna=False, as_index=False)
        .agg({"count": "sum", "docs": "sum", "tokens": "sum"})
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    grouped["per_1k_tokens"] = 1000.0 * grouped["count"] / grouped["tokens"].clip(lower=1)
    grouped["per_doc"] = grouped["count"] / grouped["docs"].clip(lower=1)
    return grouped


def _write_summary(
    *,
    path: Path,
    feature_rates: pd.DataFrame,
    by_corpus: pd.DataFrame,
    by_stratum: pd.DataFrame,
    pair_deltas: pd.DataFrame,
    feature_rate_inputs: list[Path],
    pair_delta_input: Path,
) -> None:
    excluded_present = sorted(set(feature_rates["feature"]) & EXCLUDED_REVISED_PHASE1_FEATURES)
    core_features = sorted(feature_rates["feature"].unique())
    lines = [
        "# Phase 1 Census Summary",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "Scope: revised Phase 1 close-out. Core deterministic features are contrastive",
        "negation, rule-of-three approximation, slop lexicon, and stock",
        "openers/closers. Full pybiber extraction is handled by the separate",
        "`slop-pybiber-full` batched register pipeline. Punctuation/rhythm,",
        "generic hedging, and full SmolLM replication remain exploratory or",
        "out of this assembly surface.",
        "",
        "## Inputs",
        "",
        *[f"- Feature rates: `{item}`" for item in feature_rate_inputs],
        f"- Pair deltas: `{pair_delta_input}`",
        "",
        "## Output Shapes",
        "",
        f"- `feature_rates_by_corpus.parquet`: {len(by_corpus):,} rows",
        f"- `feature_rates_by_stratum.parquet`: {len(by_stratum):,} rows",
        f"- `preference_pair_deltas.parquet`: {len(pair_deltas):,} rows",
        f"- Unique pair IDs: {pair_deltas['pair_id'].nunique():,}",
        "",
        "## Feature Scope",
        "",
        f"- Revised core/table features: {', '.join(core_features)}",
        "- Full pybiber register features: separate `slop-pybiber-full` outputs",
        f"- Removed Phase 1 features present in assembled tables: {excluded_present or 'none'}",
        "",
        "## Caveats",
        "",
        "- Current manual precision labels cover only an interim contrastive-negation",
        "  subset. Do not treat all revised core features as validated until the",
        "  precision report passes the 200-hit gate or explicitly demotes features.",
        "- Dolci DPO pair deltas are measured on the retained package with unique",
        "  row-qualified pair IDs. Interpret chosen/rejected differences by",
        "  preference type and model-pair metadata, not as a pure human-preference",
        "  effect.",
        "- Dolma 3 is a bounded retained sample; rare strata may remain",
        "  underfilled relative to the nominal per-stratum cap.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_assemble_phase1(args: argparse.Namespace) -> dict[str, Any]:
    feature_rates = _read_feature_rates(args.feature_rates)
    if feature_rates.empty:
        raise ValueError("no feature-rate rows loaded")
    missing = {"source", "subset", "role", "feature", "count", "docs", "tokens"} - set(
        feature_rates.columns
    )
    if missing:
        raise ValueError(f"feature-rate inputs missing required columns: {sorted(missing)}")

    by_corpus = _aggregate_rates(feature_rates, ["source", "role", "feature"])
    by_stratum = _aggregate_rates(feature_rates, ["source", "subset", "role", "feature"])
    pair_deltas = pd.read_csv(args.pair_deltas)

    for output_path, frame in (
        (args.output_feature_corpus, by_corpus),
        (args.output_feature_stratum, by_stratum),
        (args.output_pair_deltas, pair_deltas),
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)

    _write_summary(
        path=args.summary_output,
        feature_rates=feature_rates,
        by_corpus=by_corpus,
        by_stratum=by_stratum,
        pair_deltas=pair_deltas,
        feature_rate_inputs=args.feature_rates,
        pair_delta_input=args.pair_deltas,
    )

    summary = {
        "feature_rate_input_rows": len(feature_rates),
        "feature_rates_by_corpus_rows": len(by_corpus),
        "feature_rates_by_stratum_rows": len(by_stratum),
        "preference_pair_delta_rows": len(pair_deltas),
        "preference_pair_ids": pair_deltas["pair_id"].nunique(),
        "features": feature_rates["feature"].nunique(),
        "removed_features_present": len(set(feature_rates["feature"]) & EXCLUDED_REVISED_PHASE1_FEATURES),
    }
    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage1", "phase1", "assemble", *args.wandb_tag],
        config={
            "feature_rate_inputs": [str(path) for path in args.feature_rates],
            "pair_deltas": str(args.pair_deltas),
            "output_feature_corpus": str(args.output_feature_corpus),
            "output_feature_stratum": str(args.output_feature_stratum),
            "output_pair_deltas": str(args.output_pair_deltas),
            "summary_output": str(args.summary_output),
        },
    )
    try:
        run.log({f"assembly/{key}": value for key, value in summary.items()})
        log_summary_table(run, "assembly_summary", [summary])
    finally:
        run.finish()
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_assemble_phase1(args)
    print(f"Wrote Phase 1 assembly with {summary['preference_pair_delta_rows']} pair rows")


if __name__ == "__main__":
    main()
