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
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Optional SVG plot of realized AF temperature dependence.",
    )
    parser.add_argument("--primary-feature", default="slop_lexicon")
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


def _validate_propensity_coverage(
    *,
    specs: list[GenerationSpec],
    propensity_grid: dict[tuple[str, str], dict[str, Any]],
    primary_feature: str,
) -> None:
    if not propensity_grid:
        return
    missing = [
        spec.stage for spec in specs if (spec.stage, primary_feature) not in propensity_grid
    ]
    if missing:
        available_stages = sorted(
            {stage for stage, feature in propensity_grid if feature == primary_feature}
        )
        raise ValueError(
            "propensity grid has no primary-feature rows for generation stage(s) "
            f"{', '.join(missing)}; available stages for {primary_feature!r}: "
            f"{', '.join(available_stages) or 'none'}"
        )


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


def _svg_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _stage_sort_key(stage: str) -> tuple[int, str]:
    order = {"base": 0, "sft": 1, "dpo": 2, "final": 3}
    return (order.get(stage, len(order)), stage)


def _write_realized_af_plot(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    primary_feature: str,
) -> None:
    plot_rows = [
        row
        for row in rows
        if row["feature"] == primary_feature and row.get("realized_af") is not None
    ]
    if not plot_rows:
        raise ValueError(f"no realized AF rows to plot for feature {primary_feature!r}")

    path.parent.mkdir(parents=True, exist_ok=True)
    width = 820
    height = 500
    left = 80
    right = 230
    top = 55
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    temperatures = sorted({float(row["temperature"]) for row in plot_rows})
    max_af = max(float(row["realized_af"]) for row in plot_rows)
    y_max = max(1.0, math.ceil(max_af * 10.0) / 10.0)
    if len(temperatures) == 1:
        temp_min = temperatures[0] - 0.5
        temp_max = temperatures[0] + 0.5
    else:
        temp_min = min(temperatures)
        temp_max = max(temperatures)

    def x_for(temperature: float) -> float:
        return left + (temperature - temp_min) / (temp_max - temp_min) * plot_width

    def y_for(value: float) -> float:
        return top + (1.0 - value / y_max) * plot_height

    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in plot_rows:
        by_stage[str(row["stage"])].append(row)
    palette = {
        "base": "#4c78a8",
        "sft": "#f58518",
        "dpo": "#54a24b",
        "final": "#b279a2",
    }
    fallback_colors = ["#e45756", "#72b7b2", "#ff9da6", "#9d755d"]
    fallback_index = 0

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">Realized AF Temperature Dependence: {_svg_escape(primary_feature)}</title>",
        "<desc id=\"desc\">Line chart of realized amplification factor by temperature and model stage.</desc>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="28" font-family="sans-serif" font-size="18" font-weight="700">Realized AF vs. temperature</text>',
        f'<text x="{left}" y="48" font-family="sans-serif" font-size="12" fill="#555">Feature: {_svg_escape(primary_feature)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333" stroke-width="1"/>',
    ]

    y_ticks = 5
    for tick in range(y_ticks + 1):
        value = y_max * tick / y_ticks
        y = y_for(value)
        lines.extend(
            [
                f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e5e5e5" stroke-width="1"/>',
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#555">{value:.1f}</text>',
            ]
        )
    for temperature in temperatures:
        x = x_for(temperature)
        lines.extend(
            [
                f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" y2="{top + plot_height + 5}" stroke="#333" stroke-width="1"/>',
                f'<text x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#555">{temperature:g}</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#333">temperature</text>',
            f'<text transform="translate(22 {top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#333">realized AF</text>',
        ]
    )

    legend_y = top
    for stage in sorted(by_stage, key=_stage_sort_key):
        stage_rows = sorted(by_stage[stage], key=lambda row: float(row["temperature"]))
        color = palette.get(stage)
        if color is None:
            color = fallback_colors[fallback_index % len(fallback_colors)]
            fallback_index += 1
        points = [
            (x_for(float(row["temperature"])), y_for(float(row["realized_af"])))
            for row in stage_rows
        ]
        point_attr = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        lines.append(
            f'<polyline points="{point_attr}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        for (x, y), row in zip(points, stage_rows, strict=True):
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"><title>{_svg_escape(stage)} t={float(row["temperature"]):g}: AF={float(row["realized_af"]):.3f}</title></circle>'
            )
        lines.extend(
            [
                f'<line x1="{left + plot_width + 35}" y1="{legend_y}" x2="{left + plot_width + 58}" y2="{legend_y}" stroke="{color}" stroke-width="2.5"/>',
                f'<text x="{left + plot_width + 66}" y="{legend_y + 4}" font-family="sans-serif" font-size="12" fill="#333">{_svg_escape(stage)}</text>',
            ]
        )
        legend_y += 24
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_plot_artifact(run: Any, path: Path) -> None:
    if not hasattr(run, "log_artifact"):
        return
    import wandb

    artifact = wandb.Artifact("phase2_compounding_realized_af_plot", type="phase2_plot")
    artifact.add_file(str(path))
    run.log_artifact(artifact)


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
    _validate_propensity_coverage(
        specs=specs,
        propensity_grid=propensity_grid,
        primary_feature=args.primary_feature,
    )
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
            "primary_feature": args.primary_feature,
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
        if args.plot_output is not None:
            _write_realized_af_plot(
                path=args.plot_output,
                rows=rows,
                primary_feature=args.primary_feature,
            )
            _log_plot_artifact(run, args.plot_output)

        primary_rows = [row for row in rows if row["feature"] == args.primary_feature]
        slop_rows = [row for row in rows if row["feature"] == "slop_lexicon"]
        max_diff_row = max(slop_rows, key=lambda row: row["risk_diff_after_prior"], default=None)
        max_excess_row = max(
            (row for row in slop_rows if row["excess_rate"] is not None),
            key=lambda row: row["excess_rate"],
            default=None,
        )
        max_realized_af_row = max(
            (row for row in primary_rows if row["realized_af"] is not None),
            key=lambda row: row["realized_af"],
            default=None,
        )
        payload: dict[str, Any] = {
            "phase2_compounding/rows": len(rows),
            "phase2_compounding/stages": len(specs),
            "phase2_compounding/features": len(selected_features),
            "phase2_compounding/primary_feature": args.primary_feature,
            "phase2_compounding/window_tokens": args.window_tokens,
            "phase2_compounding/output": str(args.output),
            "phase2_compounding/summary_output": str(args.summary_output),
        }
        if args.plot_output is not None:
            payload["phase2_compounding/plot_output"] = str(args.plot_output)
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
        if max_realized_af_row is not None:
            payload.update(
                {
                    "phase2_compounding/max_primary_realized_af": max_realized_af_row[
                        "realized_af"
                    ],
                    "phase2_compounding/max_primary_realized_af_stage": max_realized_af_row[
                        "stage"
                    ],
                    "phase2_compounding/max_primary_realized_af_temperature": max_realized_af_row[
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
