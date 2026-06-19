from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path.cwd()
DEFAULT_MANUSCRIPT = Path("docs/experiments/paper_manuscript_draft.md")
DEFAULT_OUTPUT_CSV = Path("artifacts/paper/manuscript/paper_numeric_claims_audit.csv")
DEFAULT_OUTPUT_MD = Path("docs/experiments/paper_numeric_claims_audit.md")


@dataclass(frozen=True)
class NumericCheck:
    check_id: str
    source_path: Path
    expected_text: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit key manuscript numeric claims.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def run_audit_paper_numeric_claims(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = args.root.resolve()
    manuscript_path = _resolve_path(root, args.manuscript)
    manuscript_text = manuscript_path.read_text(encoding="utf-8") if manuscript_path.exists() else ""
    checks = build_numeric_checks(root)
    rows = audit_numeric_claims(manuscript_text, checks)
    output_csv = _resolve_path(root, args.output_csv)
    output_md = _resolve_path(root, args.output_md)
    _write_csv(output_csv, rows)
    _write_markdown(output_md, rows, output_csv=args.output_csv)
    return rows


def build_numeric_checks(root: Path) -> list[NumericCheck]:
    checks: list[NumericCheck] = []
    pybiber_intervals = Path("artifacts/stage1/census/phase1_pybiber_register_intervals.csv")
    pybiber_means = Path("artifacts/stage1/census/phase1_pybiber_register_means.csv")
    eqbench = Path("artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv")
    olmo_spectrum = Path(
        "artifacts/phase3/analysis/"
        "olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv"
    )
    olmo_classification = Path(
        "artifacts/phase3/analysis/"
        "olmo3_phase3_reduced_sglang_feature_classification_t07_long1024.csv"
    )
    smol_spectrum = Path(
        "artifacts/phase3/analysis/"
        "smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_"
        "baselines_data_rates_slop_neutral_rule3_production.csv"
    )
    correlations = Path(
        "artifacts/phase3/analysis/"
        "olmo3_vs_smollm3_no_think_512prompt_production_baselines_data_rates_tf_"
        "generation_compounding_slop_neutral_rule3_correlations.csv"
    )
    phase4_stage = Path(
        "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "tier3_teacher_forced_exact_512/olmo3_phase4_tier3_512_exact_sequence_stage_grid.csv"
    )
    phase4_summary = Path(
        "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "ig_10000doc/phase4_detector_ig_summary.json"
    )

    intervals = _read_csv(root / pybiber_intervals)
    means = _read_csv(root / pybiber_means)
    checks.append(
        NumericCheck(
            "pybiber_narrative_drop",
            pybiber_intervals,
            (
                f"Past tense falls from {_interval(intervals, 'dolma_pretrain', 'f_01_past_tense')} "
                f"to {_interval(intervals, 'sft_target', 'f_01_past_tense')}; "
                f"first-person pronouns fall from "
                f"{_interval(intervals, 'dolma_pretrain', 'f_06_first_person_pronouns')} "
                f"to {_interval(intervals, 'sft_target', 'f_06_first_person_pronouns')}; "
                f"third-person pronouns fall from "
                f"{_interval(intervals, 'dolma_pretrain', 'f_08_third_person_pronouns')} "
                f"to {_interval(intervals, 'sft_target', 'f_08_third_person_pronouns')}; "
                f"and clausal coordination falls from "
                f"`{_mean(means, 'dolma_pretrain', 'f_65_clausal_coordination')}` "
                f"to `{_mean(means, 'sft_target', 'f_65_clausal_coordination')}`"
            ),
        )
    )
    checks.append(
        NumericCheck(
            "pybiber_expository_shift",
            pybiber_means,
            (
                f"Nominalizations rise from `{_mean(means, 'dolma_pretrain', 'f_14_nominalizations')}` "
                f"to `{_mean(means, 'sft_target', 'f_14_nominalizations')}`, "
                f"attributive adjectives rise from `{_mean(means, 'dolma_pretrain', 'f_40_adj_attr')}` "
                f"to `{_mean(means, 'sft_target', 'f_40_adj_attr')}`, "
                f"present tense rises from `{_mean(means, 'dolma_pretrain', 'f_03_present_tense')}` "
                f"to `{_mean(means, 'sft_target', 'f_03_present_tense')}`, and "
                f"prepositions rise from `{_mean(means, 'dolma_pretrain', 'f_39_prepositions')}` "
                f"to `{_mean(means, 'sft_target', 'f_39_prepositions')}`"
            ),
        )
    )
    checks.append(
        NumericCheck(
            "pybiber_not_hedging",
            pybiber_means,
            (
                f"Hedges fall from `{_mean(means, 'dolma_pretrain', 'f_47_hedges')}` "
                f"in Dolma to `{_mean(means, 'sft_target', 'f_47_hedges')}` in SFT; "
                f"amplifiers fall from `{_mean(means, 'dolma_pretrain', 'f_48_amplifiers')}` "
                f"to `{_mean(means, 'sft_target', 'f_48_amplifiers')}`; and emphatics fall "
                f"from `{_mean(means, 'dolma_pretrain', 'f_49_emphatics')}` "
                f"to `{_mean(means, 'sft_target', 'f_49_emphatics')}`"
            ),
        )
    )
    checks.append(
        NumericCheck(
            "pybiber_dpo_contrast",
            pybiber_means,
            (
                f"chosen responses have more broad nouns "
                f"(`{_mean(means, 'dpo_chosen', 'f_16_other_nouns')}` vs. "
                f"`{_mean(means, 'dpo_rejected', 'f_16_other_nouns')}`), "
                f"attributive adjectives (`{_mean(means, 'dpo_chosen', 'f_40_adj_attr')}` "
                f"vs. `{_mean(means, 'dpo_rejected', 'f_40_adj_attr')}`), "
                f"adverbs (`{_mean(means, 'dpo_chosen', 'f_42_adverbs')}` vs. "
                f"`{_mean(means, 'dpo_rejected', 'f_42_adverbs')}`), and "
                f"prepositions (`{_mean(means, 'dpo_chosen', 'f_39_prepositions')}` "
                f"vs. `{_mean(means, 'dpo_rejected', 'f_39_prepositions')}`)"
            ),
        )
    )

    eq_rows = _read_csv(root / eqbench)
    checks.append(
        NumericCheck(
            "eqbench_olmo_stage_scores",
            eqbench,
            (
                f"the score rises from base `{_eq(eq_rows, 'olmo3', 'base')}` "
                f"[95% bootstrap CI `{_eq_low(eq_rows, 'olmo3', 'base')}`, "
                f"`{_eq_high(eq_rows, 'olmo3', 'base')}`] to SFT "
                f"`{_eq(eq_rows, 'olmo3', 'sft')}` [`{_eq_low(eq_rows, 'olmo3', 'sft')}`, "
                f"`{_eq_high(eq_rows, 'olmo3', 'sft')}`], then eases through DPO "
                f"`{_eq(eq_rows, 'olmo3', 'dpo')}` [`{_eq_low(eq_rows, 'olmo3', 'dpo')}`, "
                f"`{_eq_high(eq_rows, 'olmo3', 'dpo')}`] and final "
                f"`{_eq(eq_rows, 'olmo3', 'final')}` [`{_eq_low(eq_rows, 'olmo3', 'final')}`, "
                f"`{_eq_high(eq_rows, 'olmo3', 'final')}`]"
            ),
        )
    )
    checks.append(
        NumericCheck(
            "eqbench_smollm3_stage_scores",
            eqbench,
            (
                f"Its score rises from base `{_eq(eq_rows, 'smollm3_no_think', 'base')}` "
                f"[`{_eq_low(eq_rows, 'smollm3_no_think', 'base')}`, "
                f"`{_eq_high(eq_rows, 'smollm3_no_think', 'base')}`] to SFT "
                f"`{_eq(eq_rows, 'smollm3_no_think', 'sft')}` "
                f"[`{_eq_low(eq_rows, 'smollm3_no_think', 'sft')}`, "
                f"`{_eq_high(eq_rows, 'smollm3_no_think', 'sft')}`], then remains high "
                f"at DPO/APO `{_eq(eq_rows, 'smollm3_no_think', 'dpo')}` "
                f"[`{_eq_low(eq_rows, 'smollm3_no_think', 'dpo')}`, "
                f"`{_eq_high(eq_rows, 'smollm3_no_think', 'dpo')}`] and final "
                f"`{_eq(eq_rows, 'smollm3_no_think', 'final')}` "
                f"[`{_eq_low(eq_rows, 'smollm3_no_think', 'final')}`, "
                f"`{_eq_high(eq_rows, 'smollm3_no_think', 'final')}`]"
            ),
        )
    )

    olmo_rows = _read_csv(root / olmo_spectrum)
    olmo_class_rows = _read_csv(root / olmo_classification)
    checks.append(
        NumericCheck(
            "olmo_slop_lexicon_af",
            olmo_spectrum,
            (
                f"rises from base `{_stage_value(olmo_rows, 'slop_lexicon', 'base', 'teacher_forced_normalized_af')}` "
                f"and SFT `{_stage_value(olmo_rows, 'slop_lexicon', 'sft', 'teacher_forced_normalized_af')}` "
                f"to DPO `{_stage_value(olmo_rows, 'slop_lexicon', 'dpo', 'teacher_forced_normalized_af')}`, "
                f"then falls at final/RLVR to "
                f"`{_stage_value(olmo_rows, 'slop_lexicon', 'final', 'teacher_forced_normalized_af')}`"
            ),
        )
    )
    checks.append(
        NumericCheck(
            "olmo_slop_lexicon_preference_data",
            olmo_classification,
            (
                "the aggregate chosen-minus-rejected direction is negative "
                f"and the BH-adjusted q-value is "
                f"`{_class_value(olmo_class_rows, 'slop_lexicon', 'preference_bh_q')}`"
            ),
        )
    )
    checks.append(
        NumericCheck(
            "olmo_slop_lexicon_free_run",
            olmo_spectrum,
            (
                f"base `{_stage_value(olmo_rows, 'slop_lexicon', 'base', 'free_run_per_1k_tokens')}`, "
                f"SFT `{_stage_value(olmo_rows, 'slop_lexicon', 'sft', 'free_run_per_1k_tokens')}`, "
                f"DPO `{_stage_value(olmo_rows, 'slop_lexicon', 'dpo', 'free_run_per_1k_tokens')}`, "
                f"and final/RLVR `{_stage_value(olmo_rows, 'slop_lexicon', 'final', 'free_run_per_1k_tokens')}` "
                "hits per 1,000 generated tokens"
            ),
        )
    )
    checks.append(
        NumericCheck(
            "olmo_slop_lexicon_compounding",
            olmo_spectrum,
            (
                f"base `{_stage_value(olmo_rows, 'slop_lexicon', 'base', 'compounding_excess_per_1k_opportunities')}`, "
                f"SFT `{_stage_value(olmo_rows, 'slop_lexicon', 'sft', 'compounding_excess_per_1k_opportunities')}`, "
                f"DPO `{_stage_value(olmo_rows, 'slop_lexicon', 'dpo', 'compounding_excess_per_1k_opportunities')}`, "
                f"and final/RLVR `{_stage_value(olmo_rows, 'slop_lexicon', 'final', 'compounding_excess_per_1k_opportunities')}` "
                "excess hits per 1,000 opportunities"
            ),
        )
    )

    smol_rows = _read_csv(root / smol_spectrum)
    corr_rows = _read_csv(root / correlations)
    checks.append(
        NumericCheck(
            "smollm3_slop_lexicon_free_run",
            smol_spectrum,
            (
                f"rise from base `{_stage_value(smol_rows, 'slop_lexicon', 'base', 'free_run_per_1k_tokens')}` "
                f"to SFT `{_stage_value(smol_rows, 'slop_lexicon', 'sft', 'free_run_per_1k_tokens')}`, "
                f"DPO/APO `{_stage_value(smol_rows, 'slop_lexicon', 'dpo', 'free_run_per_1k_tokens')}`, "
                f"and final `{_stage_value(smol_rows, 'slop_lexicon', 'final', 'free_run_per_1k_tokens')}`"
            ),
        )
    )
    checks.append(
        NumericCheck(
            "cross_ladder_correlation",
            correlations,
            (
                f"The AF rank correlation is positive: Spearman "
                f"`{_correlation(corr_rows, 'spearman_af')}` and Pearson "
                f"`{_correlation(corr_rows, 'pearson_af')}`"
            ),
        )
    )

    with (root / phase4_summary).open(encoding="utf-8") as handle:
        ig_summary = json.load(handle)
    phase4_rows = _read_csv(root / phase4_stage)
    checks.append(
        NumericCheck(
            "phase4_attribution_counts",
            phase4_summary,
            (
                f"attributed over {ig_summary['documents_attributed']:,} generated documents, "
                f"producing {ig_summary['doc_span_rows']:,} document-span rows and "
                f"{ig_summary['unique_spans']:,} unique spans"
            ),
        )
    )
    checks.append(
        NumericCheck(
            "phase4_tier3_selected_af",
            phase4_stage,
            (
                f"Process framing has the most stable denominator-supported signal: "
                f"{_ref_inits(phase4_rows, 'phase4_ig_process_framing')} reference initiations, "
                f"raw AF above one at every stage, and raw AF rising from base "
                f"`{_stage_value(phase4_rows, 'phase4_ig_process_framing', 'base', 'amplification_factor')}` "
                f"to final `{_stage_value(phase4_rows, 'phase4_ig_process_framing', 'final', 'amplification_factor')}`"
            ),
        )
    )
    checks.append(
        NumericCheck(
            "phase4_tier3_sparse_af",
            phase4_stage,
            (
                f"Follow-up offers are the largest Tier-3 signal but remain sparse: "
                f"{_ref_inits(phase4_rows, 'phase4_ig_followup_offer')} reference initiations and raw AF "
                f"jumping from base `{_stage_value(phase4_rows, 'phase4_ig_followup_offer', 'base', 'amplification_factor')}` "
                "to about `39` after SFT"
            ),
        )
    )
    return checks


def audit_numeric_claims(text: str, checks: list[NumericCheck]) -> list[dict[str, str]]:
    normalized = _normalize(text)
    rows = []
    for check in checks:
        expected = _normalize(check.expected_text)
        present = expected in normalized
        rows.append(
            {
                "check_id": check.check_id,
                "status": "ready" if present else "review",
                "source_path": check.source_path.as_posix(),
                "expected_text": check.expected_text,
                "missing_text": "" if present else check.expected_text,
            }
        )
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _one(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(str(row.get(field, "")) == str(value) for field, value in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def _fmt(value: str | float) -> str:
    return f"{float(value):.3f}"


def _mean(rows: list[dict[str, str]], sample: str, feature: str) -> str:
    return _fmt(_one(rows, sample=sample, feature=feature)["token_weighted_mean"])


def _interval(rows: list[dict[str, str]], sample: str, feature: str) -> str:
    row = _one(rows, statistic="mean", sample=sample, feature=feature)
    return f"`{_fmt(row['point_estimate'])}` [`{_fmt(row['ci_low'])}`, `{_fmt(row['ci_high'])}`]"


def _eq(rows: list[dict[str, str]], ladder: str, stage: str) -> str:
    return _fmt(_one(rows, ladder=ladder, stage=stage)["eqbench_slop_score"])


def _eq_low(rows: list[dict[str, str]], ladder: str, stage: str) -> str:
    return _fmt(_one(rows, ladder=ladder, stage=stage)["eqbench_slop_score_ci_low"])


def _eq_high(rows: list[dict[str, str]], ladder: str, stage: str) -> str:
    return _fmt(_one(rows, ladder=ladder, stage=stage)["eqbench_slop_score_ci_high"])


def _stage_value(rows: list[dict[str, str]], feature: str, stage: str, column: str) -> str:
    return _fmt(_one(rows, feature=feature, stage=stage)[column])


def _class_value(rows: list[dict[str, str]], feature: str, column: str) -> str:
    return _fmt(_one(rows, feature=feature)[column])


def _correlation(rows: list[dict[str, str]], column: str) -> str:
    return _fmt(_one(rows, stage="all", scope="all")[column])


def _ref_inits(rows: list[dict[str, str]], feature: str) -> str:
    return str(int(float(_one(rows, feature=feature, stage="base")["reference_initiations"])))


def _normalize(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized.replace("- ", "-")


def _resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["check_id", "status", "source_path", "expected_text", "missing_text"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readiness = "ready" if all(row["status"] == "ready" for row in rows) else "review"
    lines = [
        "# Paper Numeric Claims Audit",
        "",
        f"Machine-readable audit: `{output_csv}`",
        "",
        f"Readiness status: `{readiness}`.",
        (
            "This audit checks that headline manuscript numbers match their source "
            "CSV/JSON artifacts after manuscript rounding."
        ),
        "",
        "| Check | Status | Source | Missing |",
        "|---|---|---|---|",
    ]
    for row in rows:
        missing = "none" if not row["missing_text"] else row["missing_text"]
        lines.append(
            f"| {row['check_id']} | {row['status']} | `{row['source_path']}` | {missing} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_audit_paper_numeric_claims(args)
    print(f"Wrote {len(rows)} numeric-claims audit rows to {args.output_csv}")
    print(f"Wrote numeric-claims audit summary to {args.output_md}")


if __name__ == "__main__":
    main()
