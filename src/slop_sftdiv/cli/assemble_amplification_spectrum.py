from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


DEFAULT_FEATURES = (
    "contrastive_negation",
    "rule_of_three_approx",
    "slop_lexicon",
    "stock_openers",
    "stock_closers",
    "stock_openers_closers",
)
STAGE_ORDER = ("base", "sft", "dpo", "final")
STAGE_LABELS = {
    "base": "Base",
    "sft": "SFT",
    "dpo": "DPO",
    "final": "Final/RLVR",
}
DATA_RATE_COLUMNS = {
    "pretrain_document": "pretrain_per_1k_tokens",
    "mid_target_response": "mid_target_per_1k_tokens",
    "target_response": "sft_target_per_1k_tokens",
    "chosen": "dpo_chosen_per_1k_tokens",
    "rejected": "dpo_rejected_per_1k_tokens",
}


def _normalized_data_rate_role(source: str, role: str) -> str:
    if role != "text":
        return role
    source_lower = source.lower()
    if "dolma" in source_lower or "pretrain" in source_lower:
        return "pretrain_document"
    if "sft" in source_lower:
        return "target_response"
    return role


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble existing Phase 1/2 artifacts into an amplification-spectrum table."
    )
    parser.add_argument("--feature-rate", action="append", default=[])
    parser.add_argument(
        "--weighted-pretrain-baseline",
        action="append",
        type=Path,
        default=[],
        help=(
            "Coverage-aware weighted pretraining baseline CSV from "
            "slop-assemble-weighted-pretrain-baseline. Its covered-only rate "
            "overrides the aggregate pretrain_per_1k_tokens field."
        ),
    )
    parser.add_argument("--propensity-grid", action="append", type=Path, default=[])
    parser.add_argument("--generation-grid", type=Path, default=None)
    parser.add_argument("--compounding", type=Path, default=None)
    parser.add_argument("--denominator-support", action="append", type=Path, default=[])
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase2/analysis/olmo3_amplification_spectrum.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("artifacts/phase2/analysis/olmo3_amplification_spectrum_summary.md"),
    )
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase2-analysis")
    parser.add_argument("--wandb-job-type", default="assemble-amplification-spectrum")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def _stage_order(stage: str) -> int:
    return STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)


def _safe_column_part(value: str) -> str:
    parts = []
    for char in value.lower():
        if char.isalnum():
            parts.append(char)
        elif parts and parts[-1] != "_":
            parts.append("_")
    return "".join(parts).strip("_") or "unknown"


def _source_rate_column(role: str, source: str) -> str | None:
    source_part = _safe_column_part(source)
    if role == "pretrain_document":
        return f"pretrain_{source_part}_per_1k_tokens"
    if role == "mid_target_response":
        return f"mid_{source_part}_per_1k_tokens"
    return None


def _weighted_pretrain_rates(paths: list[Path]) -> dict[str, dict[str, float | None]]:
    rates: dict[str, dict[str, float | None]] = {}
    for path in paths:
        for row in _read_rows(path):
            feature = str(row.get("feature", ""))
            role = str(row.get("role", ""))
            if not feature or role != "pretrain_document":
                continue
            covered_only = _float_or_none(row.get("weighted_per_1k_tokens_covered_only"))
            if covered_only is None:
                continue
            out = rates.setdefault(feature, {})
            out["pretrain_per_1k_tokens"] = covered_only
            out["pretrain_weighted_covered_recipe_share"] = _float_or_none(
                row.get("covered_recipe_share")
            )
            out["pretrain_weighted_exact_covered_recipe_share"] = _float_or_none(
                row.get("exact_covered_recipe_share")
            )
            out["pretrain_weighted_proxy_covered_recipe_share"] = _float_or_none(
                row.get("proxy_covered_recipe_share")
            )
            out["pretrain_weighted_missing_recipe_share"] = _float_or_none(
                row.get("missing_recipe_share")
            )
            out["pretrain_weighted_missing_as_zero_per_1k_tokens"] = _float_or_none(
                row.get("weighted_per_1k_tokens_missing_as_zero")
            )
    return rates


def _data_rates(paths: list[str], weighted_pretrain_paths: list[Path]) -> dict[str, dict[str, float | None]]:
    rates: dict[str, dict[str, float | None]] = {}
    aggregate_counts: dict[tuple[str, str], float] = {}
    aggregate_tokens: dict[tuple[str, str], float] = {}
    source_counts: dict[tuple[str, str], float] = {}
    source_tokens: dict[tuple[str, str], float] = {}
    source_fallback_rates: dict[tuple[str, str], list[float]] = {}
    for raw_path in paths:
        for row in _read_rows(Path(raw_path)):
            feature = str(row.get("feature", ""))
            source = str(row.get("source", ""))
            role = _normalized_data_rate_role(source, str(row.get("role", "")))
            if not feature or role not in DATA_RATE_COLUMNS:
                continue
            if row.get("per_1k_tokens") in (None, ""):
                continue
            column = DATA_RATE_COLUMNS[role]
            source_column = _source_rate_column(role, source)
            count = _float_or_none(row.get("count"))
            tokens = _float_or_none(row.get("tokens"))
            per_1k_tokens = float(row["per_1k_tokens"])
            if source_column:
                source_key = (feature, source_column)
                if count is not None and tokens is not None and tokens > 0:
                    source_counts[source_key] = source_counts.get(source_key, 0.0) + count
                    source_tokens[source_key] = source_tokens.get(source_key, 0.0) + tokens
                else:
                    source_fallback_rates.setdefault(source_key, []).append(per_1k_tokens)
            if count is None or tokens is None or tokens <= 0:
                rates.setdefault(feature, {})[column] = per_1k_tokens
                continue
            key = (feature, role)
            aggregate_counts[key] = aggregate_counts.get(key, 0.0) + count
            aggregate_tokens[key] = aggregate_tokens.get(key, 0.0) + tokens

    for (feature, source_column), count in source_counts.items():
        tokens = source_tokens[(feature, source_column)]
        if tokens > 0:
            rates.setdefault(feature, {})[source_column] = count / tokens * 1000.0
    for (feature, source_column), values in source_fallback_rates.items():
        if values and source_column not in rates.setdefault(feature, {}):
            rates[feature][source_column] = sum(values) / len(values)
    for (feature, role), count in aggregate_counts.items():
        tokens = aggregate_tokens[(feature, role)]
        if tokens > 0:
            rates.setdefault(feature, {})[DATA_RATE_COLUMNS[role]] = count / tokens * 1000.0
    for feature, weighted_rates in _weighted_pretrain_rates(weighted_pretrain_paths).items():
        rates.setdefault(feature, {}).update(weighted_rates)
    return rates


def _propensity(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for path in paths:
        for row in _read_rows(path):
            out[(str(row["feature"]), str(row["stage"]))] = {
                "teacher_forced_opportunities": _int_or_none(row.get("opportunities")),
                "teacher_forced_reference_initiations": _int_or_none(
                    row.get("reference_initiations")
                ),
                "teacher_forced_reference_rate": _float_or_none(row.get("reference_rate")),
                "teacher_forced_mean_prob_mass": _float_or_none(row.get("mean_prob_mass")),
                "teacher_forced_af": _float_or_none(row.get("amplification_factor")),
                "teacher_forced_af_ci_low": _float_or_none(
                    row.get("amplification_factor_ci_low")
                ),
                "teacher_forced_af_ci_high": _float_or_none(
                    row.get("amplification_factor_ci_high")
                ),
                "normalization_feature": row.get("normalization_feature") or "",
                "teacher_forced_normalized_af": _float_or_none(
                    row.get("normalized_amplification_factor")
                ),
                "teacher_forced_normalized_af_ci_low": _float_or_none(
                    row.get("normalized_amplification_factor_ci_low")
                ),
                "teacher_forced_normalized_af_ci_high": _float_or_none(
                    row.get("normalized_amplification_factor_ci_high")
                ),
            }
    return out


def _generation(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    out = {}
    for row in _read_rows(path):
        out[(str(row["feature"]), str(row["stage"]))] = {
            "free_run_per_1k_tokens": _float_or_none(row.get("per_1k_tokens")),
            "free_run_per_1k_tokens_ci_low": _float_or_none(
                row.get("per_1k_tokens_ci_low")
            ),
            "free_run_per_1k_tokens_ci_high": _float_or_none(
                row.get("per_1k_tokens_ci_high")
            ),
            "free_run_count": _int_or_none(row.get("count")),
            "free_run_repeated_count": _int_or_none(row.get("repeated_count")),
            "free_run_generations": _int_or_none(row.get("generations")),
            "free_run_generated_tokens": _int_or_none(row.get("generated_tokens")),
            "free_run_share_generations_with_feature": _float_or_none(
                row.get("share_generations_with_feature")
            ),
            "free_run_share_repeat_generations": _float_or_none(
                row.get("share_repeat_generations")
            ),
        }
    return out


def _compounding(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    out = {}
    for row in _read_rows(path):
        out[(str(row["feature"]), str(row["stage"]))] = {
            "compounding_observed_per_1k_opportunities": _float_or_none(
                row.get("observed_per_1k_opportunities")
            ),
            "compounding_observed_per_1k_opportunities_ci_low": _float_or_none(
                row.get("observed_per_1k_opportunities_ci_low")
            ),
            "compounding_observed_per_1k_opportunities_ci_high": _float_or_none(
                row.get("observed_per_1k_opportunities_ci_high")
            ),
            "compounding_expected_per_1k_opportunities": _float_or_none(
                row.get("expected_per_1k_opportunities")
            ),
            "compounding_excess_per_1k_opportunities": _float_or_none(
                row.get("excess_per_1k_opportunities")
            ),
            "compounding_excess_per_1k_opportunities_ci_low": _float_or_none(
                row.get("excess_per_1k_opportunities_ci_low")
            ),
            "compounding_excess_per_1k_opportunities_ci_high": _float_or_none(
                row.get("excess_per_1k_opportunities_ci_high")
            ),
            "compounding_realized_af": _float_or_none(row.get("realized_af")),
            "compounding_realized_af_ci_low": _float_or_none(
                row.get("realized_af_ci_low")
            ),
            "compounding_realized_af_ci_high": _float_or_none(
                row.get("realized_af_ci_high")
            ),
            "compounding_repeat_generations": _int_or_none(row.get("repeat_generations")),
            "compounding_risk_diff_after_prior": _float_or_none(
                row.get("risk_diff_after_prior")
            ),
            "compounding_risk_diff_after_prior_ci_low": _float_or_none(
                row.get("risk_diff_after_prior_ci_low")
            ),
            "compounding_risk_diff_after_prior_ci_high": _float_or_none(
                row.get("risk_diff_after_prior_ci_high")
            ),
            "compounding_risk_ratio_after_prior": _float_or_none(
                row.get("risk_ratio_after_prior")
            ),
            "compounding_risk_ratio_after_prior_ci_low": _float_or_none(
                row.get("risk_ratio_after_prior_ci_low")
            ),
            "compounding_risk_ratio_after_prior_ci_high": _float_or_none(
                row.get("risk_ratio_after_prior_ci_high")
            ),
        }
    return out


def _denominators(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out = {}
    for path in paths:
        for row in _read_rows(path):
            out[str(row["feature"])] = {
                "denominator_opportunities_5k": _int_or_none(row.get("opportunities")),
                "denominator_reference_initiations_5k": _int_or_none(
                    row.get("reference_initiations")
                ),
                "denominator_reference_rate_5k": _float_or_none(row.get("reference_rate")),
                "denominator_documents_with_reference_5k": _int_or_none(
                    row.get("documents_with_reference")
                ),
            }
    return out


def _coverage_note(row: dict[str, Any]) -> str:
    notes = []
    if row.get("teacher_forced_af") is None and row["feature"] == "rule_of_three_approx":
        notes.append("teacher-forced omitted: initiation contract not frozen")
    elif row.get("teacher_forced_af") is None:
        notes.append("teacher-forced missing")
    elif row["feature"] == "rule_of_three_approx":
        notes.append("teacher-forced proxy: comma-pair extension")
    refs = row.get("denominator_reference_initiations_5k")
    if row["feature"] == "contrastive_negation" and refs is not None and refs < 20:
        notes.append(f"sparse held-out references: {refs}")
    if row.get("free_run_per_1k_tokens") is None:
        notes.append("free-running missing")
    if row.get("compounding_excess_per_1k_opportunities") is None:
        notes.append("compounding missing or not opportunity-defined")
    return "; ".join(notes) if notes else "measured in supplied artifacts"


def assemble_spectrum(args: argparse.Namespace) -> list[dict[str, Any]]:
    features = tuple(args.feature or DEFAULT_FEATURES)
    data_rates = _data_rates(args.feature_rate, args.weighted_pretrain_baseline)
    propensity = _propensity(args.propensity_grid)
    generation = _generation(args.generation_grid)
    compounding = _compounding(args.compounding)
    denominators = _denominators(args.denominator_support)

    rows: list[dict[str, Any]] = []
    for feature in features:
        for stage in STAGE_ORDER:
            row: dict[str, Any] = {
                "feature": feature,
                "stage": stage,
                "stage_order": _stage_order(stage),
                "stage_label": STAGE_LABELS.get(stage, stage),
                "pretrain_per_1k_tokens": None,
                "mid_target_per_1k_tokens": None,
                "sft_target_per_1k_tokens": None,
                "dpo_chosen_per_1k_tokens": None,
                "dpo_rejected_per_1k_tokens": None,
            }
            row.update(data_rates.get(feature, {}))
            row.update(denominators.get(feature, {}))
            row.update(propensity.get((feature, stage), {}))
            row.update(generation.get((feature, stage), {}))
            row.update(compounding.get((feature, stage), {}))
            row["coverage_note"] = _coverage_note(row)
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "feature",
        "stage",
        "stage_order",
        "stage_label",
        "coverage_note",
        "pretrain_per_1k_tokens",
        "pretrain_weighted_covered_recipe_share",
        "pretrain_weighted_exact_covered_recipe_share",
        "pretrain_weighted_proxy_covered_recipe_share",
        "pretrain_weighted_missing_recipe_share",
        "pretrain_weighted_missing_as_zero_per_1k_tokens",
        "mid_target_per_1k_tokens",
        "sft_target_per_1k_tokens",
        "dpo_chosen_per_1k_tokens",
        "dpo_rejected_per_1k_tokens",
        "teacher_forced_af",
        "teacher_forced_normalized_af",
        "free_run_per_1k_tokens",
        "free_run_per_1k_tokens_ci_low",
        "free_run_per_1k_tokens_ci_high",
        "compounding_excess_per_1k_opportunities",
        "compounding_excess_per_1k_opportunities_ci_low",
        "compounding_excess_per_1k_opportunities_ci_high",
        "compounding_realized_af",
        "compounding_realized_af_ci_low",
        "compounding_realized_af_ci_high",
    ]
    columns = [column for column in preferred if column in columns] + [
        column for column in columns if column not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Amplification Spectrum",
        "",
        "This table assembles existing bounded measurement artifacts. Blank cells are missing",
        "measurements, not zero effects.",
        "",
        "| Feature | Stage | Pretrain /1k | Mid /1k | SFT data /1k | DPO chosen /1k | DPO rejected /1k | TF norm AF | Free-run /1k | Compounding excess /1k opp | Note |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {feature} | {stage} | {pretrain} | {mid} | {sft} | {chosen} | {rejected} | {tf} | {free} | {excess} | {note} |".format(
                feature=row["feature"],
                stage=row["stage_label"],
                pretrain=_fmt(row.get("pretrain_per_1k_tokens")),
                mid=_fmt(row.get("mid_target_per_1k_tokens")),
                sft=_fmt(row.get("sft_target_per_1k_tokens")),
                chosen=_fmt(row.get("dpo_chosen_per_1k_tokens")),
                rejected=_fmt(row.get("dpo_rejected_per_1k_tokens")),
                tf=_fmt(row.get("teacher_forced_normalized_af") or row.get("teacher_forced_af")),
                free=_fmt(row.get("free_run_per_1k_tokens")),
                excess=_fmt(row.get("compounding_excess_per_1k_opportunities")),
                note=row["coverage_note"],
            )
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- `rule_of_three_approx` teacher-forced values, when present, use the comma-pair extension proxy rather than open-vocabulary construction initiation.",
            "- Sparse held-out target support should be treated as a coverage caveat, not as evidence of zero effect.",
            "- Free-running and compounding columns are blank when the supplied inputs do not include those measurement layers.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = assemble_spectrum(args)
    _write_csv(args.output, rows)
    _write_summary(args.summary_output, rows)
    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage2", "phase2", "amplification-spectrum", *args.wandb_tag],
        config={
            "feature_rates": args.feature_rate,
            "weighted_pretrain_baseline": [
                str(path) for path in args.weighted_pretrain_baseline
            ],
            "propensity_grid": [str(path) for path in args.propensity_grid],
            "generation_grid": str(args.generation_grid) if args.generation_grid else None,
            "compounding": str(args.compounding) if args.compounding else None,
            "denominator_support": [str(path) for path in args.denominator_support],
            "features": list(args.feature or DEFAULT_FEATURES),
            "output": str(args.output),
            "summary_output": str(args.summary_output),
        },
    )
    try:
        run.log(
            {
                "amplification_spectrum/rows": len(rows),
                "amplification_spectrum/features": len({row["feature"] for row in rows}),
                "amplification_spectrum/stages": len({row["stage"] for row in rows}),
                "amplification_spectrum/teacher_forced_cells": sum(
                    row.get("teacher_forced_af") is not None for row in rows
                ),
                "amplification_spectrum/free_running_cells": sum(
                    row.get("free_run_per_1k_tokens") is not None for row in rows
                ),
            }
        )
        log_summary_table(run, "amplification_spectrum", rows)
    finally:
        run.finish()
    return rows


def main() -> None:
    args = build_parser().parse_args()
    rows = run(args)
    print(f"Wrote {len(rows)} amplification-spectrum rows to {args.output}")


if __name__ == "__main__":
    main()
