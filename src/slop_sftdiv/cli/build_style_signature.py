from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


STAGE_ORDER = ("base", "sft", "dpo", "final")
STAGE_LABELS = {
    "base": "Base",
    "sft": "SFT",
    "dpo": "DPO",
    "final": "Final/RLVR",
}
EPSILON = 1e-9


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a generated-output style signature from Tier-1, compounding, and pybiber register artifacts."
    )
    parser.add_argument("--tier1-generation-grid", type=Path, required=True)
    parser.add_argument("--compounding", type=Path, required=True)
    parser.add_argument("--register-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase2-analysis")
    parser.add_argument("--wandb-job-type", default="style-signature")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _signed_log2_ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    if value + EPSILON <= 0.0 or baseline + EPSILON <= 0.0:
        return None
    return math.log2((value + EPSILON) / (baseline + EPSILON))


def _stage_order(stage: str) -> int:
    return STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)


def _feature_order(feature_group: str, feature: str) -> tuple[int, str]:
    group_order = {"tier1": 0, "compounding": 1, "pybiber": 2}
    return group_order.get(feature_group, 99), feature


def _tier1_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv(path):
        stage = str(row["stage"])
        value = _float_or_none(row.get("per_1k_tokens"))
        rows.append(
            {
                "feature_group": "tier1",
                "metric": "free_run_per_1k_tokens",
                "feature": str(row["feature"]),
                "stage": stage,
                "stage_label": STAGE_LABELS.get(stage, stage),
                "value": value,
                "value_ci_low": _float_or_none(row.get("per_1k_tokens_ci_low")),
                "value_ci_high": _float_or_none(row.get("per_1k_tokens_ci_high")),
                "sft_target_baseline": None,
                "dpo_chosen_baseline": None,
                "pretrain_baseline": None,
            }
        )
    return rows


def _compounding_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv(path):
        stage = str(row["stage"])
        for metric, source_column in (
            ("prior_risk_ratio", "risk_ratio_after_prior"),
            ("prior_risk_diff", "risk_diff_after_prior"),
            ("observed_per_1k_opportunities", "observed_per_1k_opportunities"),
        ):
            value = _float_or_none(row.get(source_column))
            rows.append(
                {
                    "feature_group": "compounding",
                    "metric": metric,
                    "feature": str(row["feature"]),
                    "stage": stage,
                    "stage_label": STAGE_LABELS.get(stage, stage),
                    "value": value,
                    "value_ci_low": _float_or_none(row.get(f"{source_column}_ci_low")),
                    "value_ci_high": _float_or_none(row.get(f"{source_column}_ci_high")),
                    "sft_target_baseline": None,
                    "dpo_chosen_baseline": None,
                    "pretrain_baseline": None,
                }
            )
    return rows


def _register_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv(path):
        stage = str(row["stage"])
        value = _float_or_none(row.get("generation_per_1k_tokens"))
        rows.append(
            {
                "feature_group": "pybiber",
                "metric": "generation_per_1k_tokens",
                "feature": str(row["feature"]),
                "stage": stage,
                "stage_label": STAGE_LABELS.get(stage, stage),
                "value": value,
                "value_ci_low": _float_or_none(row.get("generation_per_1k_tokens_ci_low")),
                "value_ci_high": _float_or_none(row.get("generation_per_1k_tokens_ci_high")),
                "sft_target_baseline": _float_or_none(row.get("sft_target_per_1k_tokens")),
                "dpo_chosen_baseline": _float_or_none(row.get("dpo_chosen_per_1k_tokens")),
                "pretrain_baseline": _float_or_none(row.get("pretrain_per_1k_tokens")),
            }
        )
    return rows


def _with_stage_references(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    value_by_key = {
        (row["feature_group"], row["metric"], row["feature"], row["stage"]): row["value"]
        for row in rows
    }
    out = []
    for row in rows:
        key = (row["feature_group"], row["metric"], row["feature"])
        base = value_by_key.get((*key, "base"))
        dpo = value_by_key.get((*key, "dpo"))
        final = value_by_key.get((*key, "final"))
        sft = value_by_key.get((*key, "sft"))
        value = row["value"]
        enriched = dict(row)
        enriched.update(
            {
                "base_stage_value": base,
                "sft_stage_value": sft,
                "dpo_stage_value": dpo,
                "final_stage_value": final,
                "delta_vs_base": value - base if value is not None and base is not None else None,
                "delta_vs_dpo": value - dpo if value is not None and dpo is not None else None,
                "delta_final_vs_dpo": (
                    final - dpo if row["stage"] == "final" and final is not None and dpo is not None else None
                ),
                "log2_ratio_vs_base": _signed_log2_ratio(value, base),
                "log2_ratio_vs_sft_target": _signed_log2_ratio(
                    value, row["sft_target_baseline"]
                ),
                "log2_ratio_vs_dpo_chosen": _signed_log2_ratio(
                    value, row["dpo_chosen_baseline"]
                ),
                "log2_ratio_vs_pretrain": _signed_log2_ratio(value, row["pretrain_baseline"]),
            }
        )
        out.append(enriched)
    return out


def build_signature(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [
        *_tier1_rows(args.tier1_generation_grid),
        *_compounding_rows(args.compounding),
        *_register_rows(args.register_comparison),
    ]
    rows = _with_stage_references(rows)
    return sorted(
        rows,
        key=lambda row: (
            _feature_order(str(row["feature_group"]), str(row["feature"])),
            str(row["metric"]),
            _stage_order(str(row["stage"])),
        ),
    )


def _signature_vectors(rows: list[dict[str, Any]]) -> dict[str, dict[tuple[str, str, str], float]]:
    vectors: dict[str, dict[tuple[str, str, str], float]] = defaultdict(dict)
    for row in rows:
        value = row.get("value")
        if value is None:
            continue
        feature_key = (str(row["feature_group"]), str(row["metric"]), str(row["feature"]))
        vectors[str(row["stage"])][feature_key] = float(value)
    return vectors


def _distance_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    vectors = _signature_vectors(rows)
    keys = sorted({key for vector in vectors.values() for key in vector})
    distance_rows = []
    for left in STAGE_ORDER:
        for right in STAGE_ORDER:
            if left not in vectors or right not in vectors:
                continue
            squared = 0.0
            cosine_num = 0.0
            left_norm = 0.0
            right_norm = 0.0
            shared = 0
            for key in keys:
                if key not in vectors[left] or key not in vectors[right]:
                    continue
                a = vectors[left][key]
                b = vectors[right][key]
                squared += (a - b) ** 2
                cosine_num += a * b
                left_norm += a * a
                right_norm += b * b
                shared += 1
            cosine = (
                cosine_num / math.sqrt(left_norm * right_norm)
                if left_norm > 0.0 and right_norm > 0.0
                else None
            )
            distance_rows.append(
                {
                    "left_stage": left,
                    "right_stage": right,
                    "shared_features": shared,
                    "euclidean_distance": math.sqrt(squared),
                    "cosine_similarity": cosine,
                    "cosine_distance": 1.0 - cosine if cosine is not None else None,
                }
            )
    return distance_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "feature_group",
        "metric",
        "feature",
        "stage",
        "stage_label",
        "value",
        "value_ci_low",
        "value_ci_high",
        "sft_target_baseline",
        "dpo_chosen_baseline",
        "pretrain_baseline",
        "base_stage_value",
        "delta_vs_base",
        "delta_vs_dpo",
        "delta_final_vs_dpo",
        "log2_ratio_vs_base",
        "log2_ratio_vs_sft_target",
        "log2_ratio_vs_dpo_chosen",
        "log2_ratio_vs_pretrain",
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
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{float(value):.3f}"


def _top_rows(
    rows: list[dict[str, Any]],
    key: str,
    *,
    stage: str | None = None,
    feature_group: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get(key) is not None
        and (stage is None or row["stage"] == stage)
        and (feature_group is None or row["feature_group"] == feature_group)
    ]
    return sorted(selected, key=lambda row: abs(float(row[key])), reverse=True)[:limit]


def _write_summary(path: Path, rows: list[dict[str, Any]], distance_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 2 Final Output Style Signature",
        "",
        "This signature joins Tier-1 slop emission, empirical compounding metrics, and pybiber register rates over the final `t=1.0` generated outputs.",
        "",
        f"Rows: {len(rows)}",
        f"Stage-distance rows: {len(distance_rows)}",
        "",
        "## Final vs Base: Largest Shifts",
        "",
        "| Group | Metric | Feature | Final | Base | Delta | log2 Ratio |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in _top_rows(rows, "delta_vs_base", stage="final", limit=12):
        lines.append(
            "| {group} | {metric} | {feature} | {value} | {base} | {delta} | {ratio} |".format(
                group=row["feature_group"],
                metric=row["metric"],
                feature=row["feature"],
                value=_fmt(row["value"]),
                base=_fmt(row["base_stage_value"]),
                delta=_fmt(row["delta_vs_base"]),
                ratio=_fmt(row["log2_ratio_vs_base"]),
            )
        )
    lines.extend(
        [
            "",
            "## Final vs DPO: Largest Shifts",
            "",
            "| Group | Metric | Feature | Final | DPO | Delta |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in _top_rows(rows, "delta_final_vs_dpo", stage="final", limit=12):
        lines.append(
            "| {group} | {metric} | {feature} | {value} | {dpo} | {delta} |".format(
                group=row["feature_group"],
                metric=row["metric"],
                feature=row["feature"],
                value=_fmt(row["value"]),
                dpo=_fmt(row["dpo_stage_value"]),
                delta=_fmt(row["delta_final_vs_dpo"]),
            )
        )
    lines.extend(
        [
            "",
            "## Final Register vs SFT Targets",
            "",
            "| Feature | Final /1k | SFT Target /1k | log2 Ratio |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in _top_rows(
        rows,
        "log2_ratio_vs_sft_target",
        stage="final",
        feature_group="pybiber",
        limit=10,
    ):
        lines.append(
            "| {feature} | {value} | {baseline} | {ratio} |".format(
                feature=row["feature"],
                value=_fmt(row["value"]),
                baseline=_fmt(row["sft_target_baseline"]),
                ratio=_fmt(row["log2_ratio_vs_sft_target"]),
            )
        )
    lines.extend(
        [
            "",
            "## Stage Distances",
            "",
            "| Left | Right | Shared Features | Euclidean | Cosine Distance |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in distance_rows:
        lines.append(
            "| {left} | {right} | {shared} | {euclidean} | {cosine} |".format(
                left=row["left_stage"],
                right=row["right_stage"],
                shared=row["shared_features"],
                euclidean=_fmt(row["euclidean_distance"]),
                cosine=_fmt(row["cosine_distance"]),
            )
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Distances are on raw metric scales, so high-rate pybiber and compounding metrics dominate; use feature-level deltas for interpretation.",
            "- Pybiber values are generated-output register measurements, not opportunity-normalized propensity measurements.",
            "- Compounding values are empirical generated-text window tests; only `slop_lexicon` has the full teacher-forced expected-vs-observed join.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = build_signature(args)
    distance_rows = _distance_rows(rows)
    _write_csv(args.output, rows)
    _write_csv(args.distance_output, distance_rows)
    _write_summary(args.summary_output, rows, distance_rows)
    run_obj = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage2", "phase2", "style-signature", *args.wandb_tag],
        config={
            "tier1_generation_grid": str(args.tier1_generation_grid),
            "compounding": str(args.compounding),
            "register_comparison": str(args.register_comparison),
            "output": str(args.output),
            "distance_output": str(args.distance_output),
            "summary_output": str(args.summary_output),
        },
    )
    try:
        run_obj.log(
            {
                "style_signature/rows": len(rows),
                "style_signature/distance_rows": len(distance_rows),
                "style_signature/features": len({row["feature"] for row in rows}),
                "style_signature/stages": len({row["stage"] for row in rows}),
            }
        )
        log_summary_table(run_obj, "style_signature", rows)
        log_summary_table(run_obj, "style_signature_distances", distance_rows)
    finally:
        run_obj.finish()
    return rows


def main() -> None:
    args = build_parser().parse_args()
    rows = run(args)
    print(f"Wrote {len(rows)} style-signature rows to {args.output}")


if __name__ == "__main__":
    main()
