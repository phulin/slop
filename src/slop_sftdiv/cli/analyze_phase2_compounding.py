from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slop_sftdiv.features.tier1_matchers import TOKEN_RE, iter_tier1_hits
from slop_sftdiv.propensity import PHASE2_OPPORTUNITY_SPECS, iter_feature_opportunities
from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


DEFAULT_FEATURES = [
    "contrastive_negation",
    "rule_of_three_approx",
    "slop_lexicon",
    "stock_openers",
    "stock_closers",
]


@dataclass(frozen=True)
class GenerationSpec:
    stage: str
    path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Phase 2 free-running self-conditioning/compounding."
    )
    parser.add_argument(
        "--generation-cache",
        action="append",
        required=True,
        help="Stage-labelled generation JSONL in the form stage=path.",
    )
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument(
        "--window-tokens",
        type=int,
        default=32,
        help="Token-window denominator for empirical conditional hit rates.",
    )
    parser.add_argument(
        "--propensity-grid",
        type=Path,
        default=None,
        help="Optional assembled teacher-forced Phase 2 grid for expected-vs-observed columns.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase2-compounding")
    parser.add_argument("--wandb-job-type", default="compounding-analysis")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _parse_generation_specs(raw_specs: list[str]) -> list[GenerationSpec]:
    specs: list[GenerationSpec] = []
    seen: set[str] = set()
    for raw_spec in raw_specs:
        if "=" not in raw_spec:
            raise ValueError(f"generation cache must be stage=path, got {raw_spec!r}")
        stage, raw_path = raw_spec.split("=", 1)
        stage = stage.strip()
        if not stage:
            raise ValueError(f"empty stage in generation cache spec {raw_spec!r}")
        if stage in seen:
            raise ValueError(f"duplicate stage {stage!r}")
        seen.add(stage)
        specs.append(GenerationSpec(stage=stage, path=Path(raw_path)))
    return specs


def _selected_features(features: list[str]) -> list[str]:
    return sorted(set(features or DEFAULT_FEATURES))


def _load_generation_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def _token_windows(text: str, *, window_tokens: int) -> list[tuple[int, int]]:
    if window_tokens <= 0:
        raise ValueError("--window-tokens must be positive")
    spans = [match.span() for match in TOKEN_RE.finditer(text)]
    if not spans:
        return [(0, len(text))]
    windows = []
    for offset in range(0, len(spans), window_tokens):
        chunk = spans[offset : offset + window_tokens]
        windows.append((chunk[0][0], chunk[-1][1]))
    return windows


def _hits_by_feature(text: str, selected_features: list[str]) -> dict[str, list[int]]:
    starts: dict[str, list[int]] = {feature: [] for feature in selected_features}
    for hit in iter_tier1_hits(text, features=selected_features):
        starts.setdefault(hit.feature, []).append(hit.start)
    return {feature: sorted(feature_starts) for feature, feature_starts in starts.items()}


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.inf if numerator > 0 else 0.0
    return numerator / denominator


def _empty_counter() -> Counter[str]:
    return Counter(
        {
            "generations": 0,
            "windows": 0,
            "prior_windows": 0,
            "no_prior_windows": 0,
            "hit_windows_after_prior": 0,
            "hit_windows_without_prior": 0,
            "feature_hits": 0,
            "repeated_hits": 0,
            "generations_with_feature": 0,
            "repeat_generations": 0,
            "generation_opportunities": 0,
            "observed_initiations": 0,
        }
    )


def _analyze_generation(
    *,
    text: str,
    selected_features: list[str],
    window_tokens: int,
) -> dict[str, Counter[str]]:
    windows = _token_windows(text, window_tokens=window_tokens)
    hits = _hits_by_feature(text, selected_features)
    feature_counts: dict[str, Counter[str]] = {}
    for feature in selected_features:
        starts = hits.get(feature, [])
        opportunities = (
            iter_feature_opportunities(text, features=[feature])
            if feature in PHASE2_OPPORTUNITY_SPECS
            else []
        )
        counter = _empty_counter()
        counter["generations"] = 1
        counter["feature_hits"] = len(starts)
        counter["repeated_hits"] = max(0, len(starts) - 1)
        counter["generations_with_feature"] = 1 if starts else 0
        counter["repeat_generations"] = 1 if len(starts) > 1 else 0
        counter["generation_opportunities"] = len(opportunities)
        counter["observed_initiations"] = sum(
            1 for opportunity in opportunities if opportunity.reference_initiates
        )
        for window_start, window_end in windows:
            counter["windows"] += 1
            has_prior = any(start < window_start for start in starts)
            has_hit = any(window_start <= start < window_end for start in starts)
            if has_prior:
                counter["prior_windows"] += 1
                if has_hit:
                    counter["hit_windows_after_prior"] += 1
            else:
                counter["no_prior_windows"] += 1
                if has_hit:
                    counter["hit_windows_without_prior"] += 1
        feature_counts[feature] = counter
    return feature_counts


def _row_key(stage: str, row: dict[str, Any], feature: str) -> tuple[str, str, float, float, str]:
    return (
        stage,
        str(row.get("source", "")),
        float(row.get("temperature", 0.0)),
        float(row.get("top_p", 0.0)),
        feature,
    )


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _int_or_zero(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def _load_propensity_grid(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            stage = str(row.get("stage", ""))
            feature = str(row.get("feature", ""))
            if not stage or not feature:
                continue
            rows[(stage, feature)] = {
                "teacher_forced_opportunities": _int_or_zero(row.get("opportunities")),
                "teacher_forced_reference_initiations": _int_or_zero(
                    row.get("reference_initiations")
                ),
                "reference_rate": _float_or_none(row.get("reference_rate")),
                "mean_prob_mass": _float_or_none(row.get("mean_prob_mass")),
                "amplification_factor": _float_or_none(row.get("amplification_factor")),
                "normalized_amplification_factor": _float_or_none(
                    row.get("normalized_amplification_factor")
                ),
                "normalization_feature": row.get("normalization_feature") or "",
            }
    return rows


def _summarize_counts(
    counts: dict[tuple[str, str, float, float, str], Counter[str]],
    *,
    window_tokens: int,
    propensity_grid: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for (stage, source, temperature, top_p, feature), counter in sorted(counts.items()):
        prior_windows = int(counter["prior_windows"])
        no_prior_windows = int(counter["no_prior_windows"])
        p_after_prior = _safe_rate(int(counter["hit_windows_after_prior"]), prior_windows)
        p_without_prior = _safe_rate(int(counter["hit_windows_without_prior"]), no_prior_windows)
        generation_opportunities = int(counter["generation_opportunities"])
        observed_initiations = int(counter["observed_initiations"])
        observed_rate = _safe_rate(observed_initiations, generation_opportunities)
        propensity = propensity_grid.get((stage, feature), {})
        expected_rate = propensity.get("mean_prob_mass")
        reference_rate = propensity.get("reference_rate")
        excess_rate = (
            observed_rate - expected_rate
            if expected_rate is not None
            else None
        )
        observed_over_expected = (
            _safe_ratio(observed_rate, expected_rate)
            if expected_rate is not None
            else None
        )
        realized_af = (
            _safe_ratio(observed_rate, reference_rate)
            if reference_rate is not None
            else None
        )
        rows.append(
            {
                "stage": stage,
                "source": source,
                "temperature": temperature,
                "top_p": top_p,
                "feature": feature,
                "window_tokens": window_tokens,
                "generations": int(counter["generations"]),
                "windows": int(counter["windows"]),
                "prior_windows": prior_windows,
                "no_prior_windows": no_prior_windows,
                "hit_windows_after_prior": int(counter["hit_windows_after_prior"]),
                "hit_windows_without_prior": int(counter["hit_windows_without_prior"]),
                "p_hit_after_prior": p_after_prior,
                "p_hit_without_prior": p_without_prior,
                "risk_diff_after_prior": p_after_prior - p_without_prior,
                "risk_ratio_after_prior": _safe_ratio(p_after_prior, p_without_prior),
                "generation_opportunities": generation_opportunities,
                "observed_initiations": observed_initiations,
                "observed_rate": observed_rate,
                "observed_per_1k_opportunities": 1000.0 * observed_rate,
                "expected_rate": expected_rate,
                "expected_per_1k_opportunities": (
                    1000.0 * expected_rate if expected_rate is not None else None
                ),
                "excess_rate": excess_rate,
                "excess_per_1k_opportunities": (
                    1000.0 * excess_rate if excess_rate is not None else None
                ),
                "observed_over_expected": observed_over_expected,
                "realized_af": realized_af,
                "teacher_forced_opportunities": propensity.get("teacher_forced_opportunities"),
                "teacher_forced_reference_initiations": propensity.get(
                    "teacher_forced_reference_initiations"
                ),
                "reference_rate": reference_rate,
                "mean_prob_mass": expected_rate,
                "amplification_factor": propensity.get("amplification_factor"),
                "normalized_amplification_factor": propensity.get(
                    "normalized_amplification_factor"
                ),
                "normalization_feature": propensity.get("normalization_feature"),
                "feature_hits": int(counter["feature_hits"]),
                "repeated_hits": int(counter["repeated_hits"]),
                "generations_with_feature": int(counter["generations_with_feature"]),
                "repeat_generations": int(counter["repeat_generations"]),
                "share_repeat_generations": _safe_rate(
                    int(counter["repeat_generations"]), int(counter["generations"])
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: Any) -> str:
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{float(value):.4f}"


def _write_markdown_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    primary_rows = [row for row in rows if row["feature"] == "slop_lexicon"]
    lines = [
        "# Phase 2 Compounding Analysis",
        "",
        "Expected-vs-observed decomposition plus empirical direct test over cached generations.",
        "",
        "## Slop Lexicon",
        "",
        "| Stage | Temp | Obs/1k Opp | Exp/1k Opp | Excess/1k | Realized AF | P(hit after prior) | P(hit no prior) | Repeat Gens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary_rows:
        lines.append(
            "| {stage} | {temperature:.3f} | {observed} | {expected} | {excess} | {realized_af} | {p_after} | {p_without} | {repeat} |".format(
                stage=row["stage"],
                temperature=float(row["temperature"]),
                observed=_format_float(row["observed_per_1k_opportunities"]),
                expected=(
                    _format_float(row["expected_per_1k_opportunities"])
                    if row["expected_per_1k_opportunities"] is not None
                    else ""
                ),
                excess=(
                    _format_float(row["excess_per_1k_opportunities"])
                    if row["excess_per_1k_opportunities"] is not None
                    else ""
                ),
                realized_af=(
                    _format_float(row["realized_af"]) if row["realized_af"] is not None else ""
                ),
                p_after=_format_float(row["p_hit_after_prior"]),
                p_without=_format_float(row["p_hit_without_prior"]),
                repeat=row["repeat_generations"],
            )
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Observed rates use the Phase 2 opportunity spec where available; features without an opportunity spec have zero opportunity-denominator values.",
            "- Expected rates come from the joined teacher-forced mean probability mass when `--propensity-grid` is provided.",
            "- Conditional prior/no-prior rates are an empirical window-level direct test over generated text, not a teacher-forced probability query.",
            "- Sparse prior-window counts should be treated as directional until larger generation shards are run.",
            "- The conditional denominator is fixed token windows, so those rates are comparable within this report but not identical to feature-specific opportunity rates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analyze_phase2_compounding(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = _parse_generation_specs(args.generation_cache)
    selected_features = _selected_features(args.feature)
    propensity_grid = _load_propensity_grid(args.propensity_grid)
    counts: dict[tuple[str, str, float, float, str], Counter[str]] = defaultdict(_empty_counter)

    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage2", "phase2", "compounding", *args.wandb_tag],
        config={
            "generation_caches": {spec.stage: str(spec.path) for spec in specs},
            "propensity_grid": str(args.propensity_grid) if args.propensity_grid else None,
            "features": selected_features,
            "window_tokens": args.window_tokens,
        },
    )
    try:
        for spec in specs:
            for row in _load_generation_rows(spec.path):
                text = str(row.get("generation", ""))
                generation_counts = _analyze_generation(
                    text=text,
                    selected_features=selected_features,
                    window_tokens=args.window_tokens,
                )
                for feature, feature_counter in generation_counts.items():
                    counts[_row_key(spec.stage, row, feature)].update(feature_counter)
        rows = _summarize_counts(
            counts,
            window_tokens=args.window_tokens,
            propensity_grid=propensity_grid,
        )
        _write_csv(args.output, rows)
        _write_markdown_summary(args.summary_output, rows)

        slop_rows = [row for row in rows if row["feature"] == "slop_lexicon"]
        max_diff_row = max(slop_rows, key=lambda row: row["risk_diff_after_prior"], default=None)
        max_excess_row = max(
            (row for row in slop_rows if row["excess_rate"] is not None),
            key=lambda row: row["excess_rate"],
            default=None,
        )
        payload: dict[str, Any] = {
            "phase2_compounding/rows": len(rows),
            "phase2_compounding/stages": len(specs),
            "phase2_compounding/features": len(selected_features),
            "phase2_compounding/window_tokens": args.window_tokens,
            "phase2_compounding/output": str(args.output),
            "phase2_compounding/summary_output": str(args.summary_output),
        }
        if max_diff_row is not None:
            payload.update(
                {
                    "phase2_compounding/max_slop_risk_diff": max_diff_row["risk_diff_after_prior"],
                    "phase2_compounding/max_slop_risk_diff_stage": max_diff_row["stage"],
                    "phase2_compounding/max_slop_risk_diff_temperature": max_diff_row[
                        "temperature"
                    ],
                }
            )
        if max_excess_row is not None:
            payload.update(
                {
                    "phase2_compounding/max_slop_excess_per_1k": max_excess_row[
                        "excess_per_1k_opportunities"
                    ],
                    "phase2_compounding/max_slop_excess_stage": max_excess_row["stage"],
                    "phase2_compounding/max_slop_excess_temperature": max_excess_row[
                        "temperature"
                    ],
                }
            )
        run.log(payload)
        log_summary_table(run, "phase2_compounding", rows)
        return rows
    finally:
        run.finish()


def main() -> None:
    args = build_parser().parse_args()
    rows = run_analyze_phase2_compounding(args)
    print(f"Wrote Phase 2 compounding analysis with {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
