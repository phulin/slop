from __future__ import annotations

import argparse
import csv
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify features in an amplification-spectrum table for Phase 3."
    )
    parser.add_argument("--spectrum", type=Path, required=True)
    parser.add_argument("--preference-analysis", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--high-sft-rate-min", type=float, default=0.25)
    parser.add_argument("--af-approx-margin", type=float, default=0.25)
    parser.add_argument("--af-amplified-min", type=float, default=1.25)
    parser.add_argument("--preference-af-jump-min", type=float, default=0.25)
    parser.add_argument("--preference-af-relative-jump-min", type=float, default=0.1)
    parser.add_argument("--chosen-rejected-delta-min", type=float, default=0.0)
    parser.add_argument("--modest-af-max", type=float, default=1.25)
    parser.add_argument("--compounding-excess-min", type=float, default=0.1)
    parser.add_argument("--compounding-realized-af-min", type=float, default=1.1)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase3-analysis")
    parser.add_argument("--wandb-job-type", default="classify-amplification-spectrum")
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


def _bool_or_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_p(value: Any) -> str:
    if value in (None, ""):
        return ""
    number = float(value)
    if number != 0.0 and abs(number) < 0.001:
        return f"{number:.3e}"
    return f"{number:.3f}"


def _bool_text(value: bool) -> str:
    return "yes" if value else "no"


def _stage_value(rows: dict[str, dict[str, Any]], stage: str, column: str) -> float | None:
    return _float_or_none(rows.get(stage, {}).get(column))


def _af_value(rows: dict[str, dict[str, Any]], stage: str) -> tuple[float | None, str]:
    normalized = _stage_value(rows, stage, "teacher_forced_normalized_af")
    if normalized is not None and normalized > 0.0:
        return normalized, "normalized"
    raw = _stage_value(rows, stage, "teacher_forced_af")
    return raw, "raw" if raw is not None else ""


def _af_ci(rows: dict[str, dict[str, Any]], stage: str) -> tuple[float | None, float | None]:
    af, source = _af_value(rows, stage)
    if af is None:
        return None, None
    if source == "normalized":
        return (
            _stage_value(rows, stage, "teacher_forced_normalized_af_ci_low"),
            _stage_value(rows, stage, "teacher_forced_normalized_af_ci_high"),
        )
    return (
        _stage_value(rows, stage, "teacher_forced_af_ci_low"),
        _stage_value(rows, stage, "teacher_forced_af_ci_high"),
    )


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _preference_analysis(paths: list[Path]) -> dict[str, dict[str, Any]]:
    best_by_feature: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _read_csv(path):
            feature = str(row.get("feature", ""))
            if not feature:
                continue
            current = best_by_feature.get(feature)
            current_pairs = int(float(current.get("pairs", 0))) if current else -1
            row_pairs = int(float(row.get("pairs", 0) or 0))
            if current is None or row_pairs > current_pairs:
                best_by_feature[feature] = {**row, "preference_analysis_path": str(path)}
    return best_by_feature


def _max(values: list[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    return max(numeric) if numeric else None


def _all_available_approx_one(values: list[float | None], margin: float) -> bool:
    numeric = [value for value in values if value is not None]
    return bool(numeric) and all(1.0 - margin <= value <= 1.0 + margin for value in numeric)


def _ci_excludes_one(rows: dict[str, dict[str, Any]], stage: str) -> str:
    low, high = _af_ci(rows, stage)
    if low is None or high is None:
        return "missing"
    if low > 1.0:
        return "above_1"
    if high < 1.0:
        return "below_1"
    return "overlaps_1"


def _primary_class(
    *,
    inherited: bool,
    sft_amplified: bool,
    preference_amplified: bool,
    compounding_dominant: bool,
    measured_teacher_forced: bool,
) -> str:
    if preference_amplified:
        return "preference-amplified"
    if sft_amplified:
        return "sft-amplified"
    if compounding_dominant:
        return "compounding-dominant"
    if inherited:
        return "inherited"
    if measured_teacher_forced:
        return "measured-no-phase3-class"
    return "observed-output-only"


def _classify_feature(
    feature: str,
    rows: dict[str, dict[str, Any]],
    preference_row: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    first_row = next(iter(rows.values()))
    sft_rate = _float_or_none(first_row.get("sft_target_per_1k_tokens"))
    chosen_rate = _float_or_none(first_row.get("dpo_chosen_per_1k_tokens"))
    rejected_rate = _float_or_none(first_row.get("dpo_rejected_per_1k_tokens"))
    pretrain_rate = _float_or_none(first_row.get("pretrain_per_1k_tokens"))
    chosen_minus_rejected = (
        chosen_rate - rejected_rate if chosen_rate is not None and rejected_rate is not None else None
    )
    chosen_rejected_ratio = _ratio(chosen_rate, rejected_rate)

    afs = {stage: _af_value(rows, stage)[0] for stage in STAGE_ORDER}
    af_source = next((source for stage in STAGE_ORDER for source in [_af_value(rows, stage)[1]] if source), "")
    pre_preference_max_af = _max([afs["base"], afs["sft"]])
    post_preference_max_af = _max([afs["dpo"], afs["final"]])
    preference_af_jump = (
        post_preference_max_af - pre_preference_max_af
        if post_preference_max_af is not None and pre_preference_max_af is not None
        else None
    )
    preference_af_relative_jump = _ratio(preference_af_jump, pre_preference_max_af)
    dpo_minus_sft_af = (
        afs["dpo"] - afs["sft"] if afs["dpo"] is not None and afs["sft"] is not None else None
    )
    final_minus_dpo_af = (
        afs["final"] - afs["dpo"]
        if afs["final"] is not None and afs["dpo"] is not None
        else None
    )

    free_run = {stage: _stage_value(rows, stage, "free_run_per_1k_tokens") for stage in STAGE_ORDER}
    free_run_max_stage = max(
        (stage for stage in STAGE_ORDER if free_run[stage] is not None),
        key=lambda stage: float(free_run[stage] or 0.0),
        default="",
    )
    dpo_minus_sft_free_run = (
        free_run["dpo"] - free_run["sft"]
        if free_run["dpo"] is not None and free_run["sft"] is not None
        else None
    )
    final_minus_dpo_free_run = (
        free_run["final"] - free_run["dpo"]
        if free_run["final"] is not None and free_run["dpo"] is not None
        else None
    )

    compounding_excess = {
        stage: _stage_value(rows, stage, "compounding_excess_per_1k_opportunities")
        for stage in STAGE_ORDER
    }
    compounding_realized_af = {
        stage: _stage_value(rows, stage, "compounding_realized_af") for stage in STAGE_ORDER
    }
    max_compounding_excess = _max(list(compounding_excess.values()))
    max_realized_af = _max(list(compounding_realized_af.values()))

    af_values = list(afs.values())
    max_af = _max(af_values)
    measured_teacher_forced = max_af is not None
    high_sft_data = sft_rate is not None and sft_rate >= args.high_sft_rate_min
    af_approx_one = _all_available_approx_one(af_values, args.af_approx_margin)
    inherited = high_sft_data and af_approx_one
    sft_amplified = afs["sft"] is not None and afs["sft"] >= args.af_amplified_min
    preference_amplified = (
        post_preference_max_af is not None
        and post_preference_max_af >= args.af_amplified_min
        and preference_af_jump is not None
        and preference_af_jump >= args.preference_af_jump_min
        and preference_af_relative_jump is not None
        and preference_af_relative_jump >= args.preference_af_relative_jump_min
    )
    preference_sign_test_p = (
        _float_or_none(preference_row.get("sign_test_p")) if preference_row else None
    )
    preference_bh_q = _float_or_none(preference_row.get("bh_q")) if preference_row else None
    preference_fdr_significant = (
        _bool_or_false(preference_row.get("fdr_significant")) if preference_row else False
    )
    preference_direction = str(preference_row.get("direction", "")) if preference_row else ""
    preference_mean_delta = (
        _float_or_none(preference_row.get("mean_delta")) if preference_row else None
    )
    preference_pairs = (
        int(float(preference_row.get("pairs", 0))) if preference_row and preference_row.get("pairs") else None
    )
    if preference_row:
        preference_data_complicit = (
            preference_fdr_significant and preference_direction == "chosen_gt_rejected"
        )
        preference_evidence = "paired_sign_test_bh_fdr"
        fdr_status = "preference_pair_fdr_available_af_fdr_missing"
    else:
        preference_data_complicit = (
            chosen_minus_rejected is not None
            and chosen_minus_rejected > args.chosen_rejected_delta_min
        )
        preference_evidence = "aggregate_data_rate_delta"
        fdr_status = "not_computed_missing_p_values"
    if preference_amplified:
        preference_cause = "signal-driven" if preference_data_complicit else "dynamics-driven"
    else:
        preference_cause = "not-preference-amplified"
    compounding_dominant = (
        max_af is not None
        and max_af <= args.modest_af_max
        and max_compounding_excess is not None
        and max_compounding_excess >= args.compounding_excess_min
        and max_realized_af is not None
        and max_realized_af >= args.compounding_realized_af_min
    )
    primary_class = _primary_class(
        inherited=inherited,
        sft_amplified=sft_amplified,
        preference_amplified=preference_amplified,
        compounding_dominant=compounding_dominant,
        measured_teacher_forced=measured_teacher_forced,
    )

    notes = []
    coverage_notes = sorted({str(row.get("coverage_note", "")) for row in rows.values() if row.get("coverage_note")})
    if any("sparse held-out references" in note for note in coverage_notes):
        notes.append("sparse held-out references")
    if any("teacher-forced proxy" in note for note in coverage_notes):
        notes.append("teacher-forced proxy")
    if not measured_teacher_forced:
        notes.append("teacher-forced missing")
    if preference_amplified and not preference_data_complicit:
        notes.append("preference-stage AF rise without chosen>rejected data-rate support")
    if max_compounding_excess is None:
        notes.append("observed-vs-expected compounding unavailable")

    out = {
        "feature": feature,
        "primary_class": primary_class,
        "af_source": af_source,
        "high_sft_data": high_sft_data,
        "af_approx_one": af_approx_one,
        "sft_amplified": sft_amplified,
        "preference_amplified": preference_amplified,
        "preference_data_complicit": preference_data_complicit,
        "preference_evidence": preference_evidence,
        "preference_cause": preference_cause,
        "compounding_dominant": compounding_dominant,
        "fdr_status": fdr_status,
        "preference_analysis_path": (
            str(preference_row.get("preference_analysis_path")) if preference_row else None
        ),
        "preference_pairs": preference_pairs,
        "preference_mean_delta": preference_mean_delta,
        "preference_sign_test_p": preference_sign_test_p,
        "preference_bh_q": preference_bh_q,
        "preference_fdr_significant": preference_fdr_significant,
        "preference_direction": preference_direction,
        "pretrain_per_1k_tokens": pretrain_rate,
        "sft_target_per_1k_tokens": sft_rate,
        "dpo_chosen_per_1k_tokens": chosen_rate,
        "dpo_rejected_per_1k_tokens": rejected_rate,
        "chosen_minus_rejected_per_1k": chosen_minus_rejected,
        "chosen_rejected_ratio": chosen_rejected_ratio,
        "base_af": afs["base"],
        "sft_af": afs["sft"],
        "dpo_af": afs["dpo"],
        "final_af": afs["final"],
        "base_af_ci_relation_to_1": _ci_excludes_one(rows, "base"),
        "sft_af_ci_relation_to_1": _ci_excludes_one(rows, "sft"),
        "dpo_af_ci_relation_to_1": _ci_excludes_one(rows, "dpo"),
        "final_af_ci_relation_to_1": _ci_excludes_one(rows, "final"),
        "pre_preference_max_af": pre_preference_max_af,
        "post_preference_max_af": post_preference_max_af,
        "preference_af_jump": preference_af_jump,
        "preference_af_relative_jump": preference_af_relative_jump,
        "dpo_minus_sft_af": dpo_minus_sft_af,
        "final_minus_dpo_af": final_minus_dpo_af,
        "base_free_run_per_1k_tokens": free_run["base"],
        "sft_free_run_per_1k_tokens": free_run["sft"],
        "dpo_free_run_per_1k_tokens": free_run["dpo"],
        "final_free_run_per_1k_tokens": free_run["final"],
        "free_run_max_stage": free_run_max_stage,
        "dpo_minus_sft_free_run_per_1k": dpo_minus_sft_free_run,
        "final_minus_dpo_free_run_per_1k": final_minus_dpo_free_run,
        "max_compounding_excess_per_1k_opportunities": max_compounding_excess,
        "max_compounding_realized_af": max_realized_af,
        "notes": "; ".join(notes),
    }
    return out


def classify_spectrum(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows_by_feature: dict[str, dict[str, dict[str, Any]]] = {}
    for row in _read_csv(args.spectrum):
        rows_by_feature.setdefault(str(row["feature"]), {})[str(row["stage"])] = row
    preference_by_feature = _preference_analysis(args.preference_analysis)
    return [
        _classify_feature(feature, rows_by_stage, preference_by_feature.get(feature), args)
        for feature, rows_by_stage in sorted(rows_by_feature.items())
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "feature",
        "primary_class",
        "preference_cause",
        "preference_evidence",
        "af_source",
        "fdr_status",
        "preference_pairs",
        "preference_mean_delta",
        "preference_sign_test_p",
        "preference_bh_q",
        "preference_fdr_significant",
        "preference_direction",
        "sft_target_per_1k_tokens",
        "dpo_chosen_per_1k_tokens",
        "dpo_rejected_per_1k_tokens",
        "chosen_minus_rejected_per_1k",
        "base_af",
        "sft_af",
        "dpo_af",
        "final_af",
        "preference_af_jump",
        "preference_af_relative_jump",
        "base_free_run_per_1k_tokens",
        "sft_free_run_per_1k_tokens",
        "dpo_free_run_per_1k_tokens",
        "final_free_run_per_1k_tokens",
        "free_run_max_stage",
        "max_compounding_excess_per_1k_opportunities",
        "max_compounding_realized_af",
        "notes",
    ]
    columns = [column for column in preferred if column in columns] + [
        column for column in columns if column not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    class_counts: dict[str, int] = {}
    for row in rows:
        class_counts[row["primary_class"]] = class_counts.get(row["primary_class"], 0) + 1

    lines = [
        "# Phase 3 Bounded Amplification-Spectrum Classification",
        "",
        "This report classifies features from an assembled amplification-spectrum table.",
        "It is a bounded OLMo 3/Dolci Phase 3 layer over the retained Phase 2 artifacts,",
        "not the full EXPERIMENTS.md two-ladder production grid.",
        "",
        f"Input spectrum: `{args.spectrum}`",
        f"Rows: {len(rows)} feature classifications",
        "",
        "## Classification Counts",
        "",
        "| Class | Count |",
        "|---|---:|",
    ]
    for class_name, count in sorted(class_counts.items()):
        lines.append(f"| {class_name} | {count} |")

    lines.extend(
        [
            "",
            "## Feature Classifications",
            "",
        "| Feature | Class | Cause | Preference evidence | q | SFT Data /1k | Chosen - Rejected /1k | Base AF | SFT AF | DPO AF | Final AF | Free-run max | Max compounding excess | Notes |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {feature} | {klass} | {cause} | {evidence} | {q} | {sft_rate} | {chosen_delta} | {base_af} | {sft_af} | {dpo_af} | {final_af} | {free_stage} | {excess} | {notes} |".format(
                feature=row["feature"],
                klass=row["primary_class"],
                cause=row["preference_cause"],
                evidence=row["preference_evidence"],
                q=_fmt_p(row.get("preference_bh_q")),
                sft_rate=_fmt(row.get("sft_target_per_1k_tokens")),
                chosen_delta=_fmt(row.get("chosen_minus_rejected_per_1k")),
                base_af=_fmt(row.get("base_af")),
                sft_af=_fmt(row.get("sft_af")),
                dpo_af=_fmt(row.get("dpo_af")),
                final_af=_fmt(row.get("final_af")),
                free_stage=STAGE_LABELS.get(str(row.get("free_run_max_stage")), ""),
                excess=_fmt(row.get("max_compounding_excess_per_1k_opportunities")),
                notes=row.get("notes", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Rule Settings",
            "",
            f"- High SFT-data rate: `>= {args.high_sft_rate_min}` per 1k tokens.",
            f"- AF approximately 1: within `+/- {args.af_approx_margin}`.",
            f"- Amplified AF: `>= {args.af_amplified_min}`.",
            f"- Preference-stage AF jump: `>= {args.preference_af_jump_min}` from max(base, SFT) to max(DPO, final).",
            f"- Preference-stage relative AF jump: `>= {args.preference_af_relative_jump_min}`.",
            f"- Preference-data complicity: chosen - rejected `> {args.chosen_rejected_delta_min}` per 1k tokens.",
            f"- Compounding-dominant: max AF `<= {args.modest_af_max}`, max compounding excess `>= {args.compounding_excess_min}`, and max realized AF `>= {args.compounding_realized_af_min}`.",
            "",
            "## Phase 3 Completion Caveats",
            "",
            "- When `--preference-analysis` is supplied, signal-driven preference complicity uses paired sign-test BH-FDR from Result A.",
            "- AF-stage FDR is still unavailable because the retained spectrum artifacts do not contain per-feature AF p-values.",
            "- SmolLM3 replication and cross-ladder AF rank correlation are not present in this bounded OLMo-only table.",
            "- Classifications are rule-based labels over measured artifacts; feature-level notes should be read alongside the Phase 2 denominator and coverage caveats.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = classify_spectrum(args)
    _write_csv(args.output, rows)
    _write_summary(args.summary_output, rows, args)
    run_obj = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage3", "phase3", "amplification-spectrum-classification", *args.wandb_tag],
        config={
            "spectrum": str(args.spectrum),
            "preference_analysis": [str(path) for path in args.preference_analysis],
            "output": str(args.output),
            "summary_output": str(args.summary_output),
            "high_sft_rate_min": args.high_sft_rate_min,
            "af_approx_margin": args.af_approx_margin,
            "af_amplified_min": args.af_amplified_min,
            "preference_af_jump_min": args.preference_af_jump_min,
            "preference_af_relative_jump_min": args.preference_af_relative_jump_min,
            "chosen_rejected_delta_min": args.chosen_rejected_delta_min,
            "modest_af_max": args.modest_af_max,
            "compounding_excess_min": args.compounding_excess_min,
            "compounding_realized_af_min": args.compounding_realized_af_min,
        },
    )
    try:
        class_counts: dict[str, int] = {}
        for row in rows:
            class_counts[row["primary_class"]] = class_counts.get(row["primary_class"], 0) + 1
        run_obj.log(
            {
                "phase3_classification/features": len(rows),
                **{
                    f"phase3_classification/class_count/{class_name}": count
                    for class_name, count in class_counts.items()
                },
            }
        )
        log_summary_table(run_obj, "phase3_feature_classification", rows)
    finally:
        run_obj.finish()
    return rows


def main() -> None:
    args = build_parser().parse_args()
    rows = run(args)
    print(f"Wrote {len(rows)} Phase 3 feature classifications to {args.output}")


if __name__ == "__main__":
    main()
