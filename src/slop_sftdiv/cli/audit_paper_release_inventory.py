from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path.cwd()
DEFAULT_OUTPUT_CSV = Path("artifacts/paper/submission/paper_release_inventory.csv")
DEFAULT_OUTPUT_MD = Path("docs/experiments/paper_release_inventory.md")


@dataclass(frozen=True)
class ReleaseItem:
    category: str
    path: Path
    release_role: str
    caveat: str


RELEASE_ITEMS: tuple[ReleaseItem, ...] = (
    ReleaseItem(
        "feature_definitions",
        Path("src/slop_sftdiv/features/tier1_matchers.py"),
        "Tier-1 regex/span matchers used for corpus and generation scoring.",
        "Rule-of-three remains demoted for independent publication-grade claims.",
    ),
    ReleaseItem(
        "feature_definitions",
        Path("src/slop_sftdiv/features/eqbench_slop.py"),
        "Portable EQ-Bench Slop Score implementation and aggregate scoring helpers.",
        "Benchmark bridge only; not the causal propensity estimand.",
    ),
    ReleaseItem(
        "feature_definitions",
        Path("src/slop_sftdiv/features/pybiber_full.py"),
        "Full pybiber feature extraction wrapper for corpus-side register analysis.",
        "Corpus-side Phase 1 use only; generated-output full pybiber is not claimed.",
    ),
    ReleaseItem(
        "opportunity_definitions",
        Path("src/slop_sftdiv/propensity.py"),
        "Opportunity definitions and initiator sets for teacher-forced propensity.",
        "Some Tier-3 opportunities remain provisional until human validation.",
    ),
    ReleaseItem(
        "opportunity_definitions",
        Path("artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_detector_tier3_matcher_specs.json"),
        "Detector-derived Tier-3 matcher specs and opportunity definitions.",
        "Candidate features require human perceptibility labels and precision validation.",
    ),
    ReleaseItem(
        "harness_code",
        Path("src/slop_sftdiv/cli/teacher_forced_propensity.py"),
        "Teacher-forced propensity harness.",
        "Production conclusions use bounded retained reference panels.",
    ),
    ReleaseItem(
        "harness_code",
        Path("src/slop_sftdiv/cli/free_running_emission.py"),
        "Free-running generation/scoring harness.",
        "Reduced SGLang grids are the production estimand.",
    ),
    ReleaseItem(
        "harness_code",
        Path("src/slop_sftdiv/generation_text.py"),
        "Generation text extraction and normalization helpers.",
        "Used to keep generated-output scoring consistent across backends.",
    ),
    ReleaseItem(
        "harness_code",
        Path("src/slop_sftdiv/cli/import_phase4_blind_labels.py"),
        "Phase 4 blinded-label import utility.",
        "Use only after annotators return blinded sheets with matching private maps.",
    ),
    ReleaseItem(
        "harness_code",
        Path("src/slop_sftdiv/cli/summarize_phase4_label_agreement.py"),
        "Phase 4 dual-annotator agreement summarizer.",
        "Agreement rows require adjudication before final human-perceptibility claims.",
    ),
    ReleaseItem(
        "harness_code",
        Path("src/slop_sftdiv/cli/apply_phase4_label_adjudication.py"),
        "Phase 4 adjudication applier for human labels.",
        "Only adjudicated canonical labels should feed final human-perceptibility summaries.",
    ),
    ReleaseItem(
        "harness_code",
        Path("src/slop_sftdiv/cli/check_paper_package.py"),
        "Paper package consistency checker.",
        "Guardrail only; passing checks do not resolve venue or human-label blockers.",
    ),
    ReleaseItem(
        "harness_code",
        Path("src/slop_sftdiv/cli/materialize_paper_submission_bundle.py"),
        "Venue-neutral draft-bundle materializer.",
        "Staging helper only; final venue submission still requires template adaptation.",
    ),
    ReleaseItem(
        "harness_code",
        Path("src/slop_sftdiv/cli/audit_paper_reproducibility_manifest.py"),
        "Paper reproducibility manifest writer.",
        "Checksum inventory only; does not certify external archival completeness.",
    ),
    ReleaseItem(
        "runbook",
        Path("docs/experiments/paper_reproduction_runbook.md"),
        "Paper-facing refresh command order.",
        "Refreshes cached paper layer; does not rerun large production jobs.",
    ),
    ReleaseItem(
        "runbook",
        Path("docs/experiments/paper_venue_sizing_inventory.md"),
        "Venue-facing figure/table sizing and export inventory.",
        "Inventory only; final dimensions still depend on target venue.",
    ),
    ReleaseItem(
        "runbook",
        Path("docs/experiments/phase3_production_runbook.md"),
        "Reduced SGLang generation and amplification-spectrum production runbook.",
        "Original exhaustive OLMo grid remains optional addendum.",
    ),
    ReleaseItem(
        "runbook",
        Path("docs/experiments/phase4_production_runbook.md"),
        "Detector discovery and Tier-3 follow-up production runbook.",
        "Human labels remain external to the current cached run.",
    ),
    ReleaseItem(
        "cached_statistics",
        Path("artifacts/stage1/census/feature_rates_by_corpus.parquet"),
        "Phase 1 corpus feature rates.",
        "Bounded English-filtered retained samples, not full-corpus coverage.",
    ),
    ReleaseItem(
        "cached_statistics",
        Path("artifacts/stage1/census/preference_pair_deltas.parquet"),
        "Phase 1 Dolci DPO chosen/rejected pair deltas.",
        "Preference-data contrast is not a pure human-preference estimate.",
    ),
    ReleaseItem(
        "cached_statistics",
        Path("artifacts/stage1/census/phase1_pybiber_register_intervals.csv"),
        "Selected full-pybiber bootstrap intervals.",
        "Selected paper-facing features, not all 67 features.",
    ),
    ReleaseItem(
        "cached_statistics",
        Path("artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv"),
        "EQ-Bench Slop Score stage intervals.",
        "Aggregate diagnostic with output-bootstrap intervals.",
    ),
    ReleaseItem(
        "cached_statistics",
        Path("artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv"),
        "Primary reduced OLMo amplification spectrum.",
        "Bounded SGLang production scope.",
    ),
    ReleaseItem(
        "cached_statistics",
        Path(
            "artifacts/phase3/analysis/"
            "smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_"
            "baselines_data_rates_slop_neutral_rule3_production.csv"
        ),
        "Primary SmolLM3 no-think amplification spectrum.",
        "Cross-ladder comparison has small shared non-missing AF support.",
    ),
    ReleaseItem(
        "cached_statistics",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/"
            "olmo3_phase4_tier3_512_exact_sequence_stage_grid.csv"
        ),
        "Phase 4 Tier-3 exact sequence-mass teacher-forced stage grid.",
        "Tier-3 features remain candidate-only pending human labels.",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit paper open-release inventory.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def run_audit_paper_release_inventory(args: argparse.Namespace) -> list[dict[str, str]]:
    root = args.root.resolve()
    rows = build_release_inventory_rows(root, RELEASE_ITEMS)
    output_csv = _resolve_path(root, args.output_csv)
    output_md = _resolve_path(root, args.output_md)
    _write_csv(output_csv, rows)
    _write_markdown(output_md, rows, output_csv=args.output_csv)
    return rows


def build_release_inventory_rows(root: Path, items: tuple[ReleaseItem, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in items:
        path = root / item.path
        rows.append(
            {
                "category": item.category,
                "path": str(item.path),
                "status": "ready" if path.exists() else "missing",
                "size_bytes": str(path.stat().st_size) if path.exists() else "",
                "release_role": item.release_role,
                "caveat": item.caveat,
            }
        )
    return rows


def _resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["category", "path", "status", "size_bytes", "release_role", "caveat"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, str]], *, output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readiness = "ready" if all(row["status"] == "ready" for row in rows) else "review"
    category_counts: dict[str, int] = {}
    for row in rows:
        category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
    lines = [
        "# Paper Release Inventory",
        "",
        f"Machine-readable inventory: `{output_csv}`",
        "",
        f"Readiness status: `{readiness}`.",
        (
            "This inventory maps the EXPERIMENTS.md open-release deliverable to "
            "concrete repository paths: feature definitions, opportunity definitions, "
            "harness code, runbooks, and cached paper statistics. It is an inventory, "
            "not a license review or final archival deposit."
        ),
        "",
        "| Category | Items |",
        "|---|---:|",
    ]
    for category, count in sorted(category_counts.items()):
        lines.append(f"| {category} | {count} |")
    lines.extend(
        [
            "",
            "| Category | Path | Status | Role | Caveat |",
            "|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['category']} | `{row['path']}` | {row['status']} | "
            f"{row['release_role']} | {row['caveat']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_audit_paper_release_inventory(args)
    ready = sum(row["status"] == "ready" for row in rows)
    missing = sum(row["status"] != "ready" for row in rows)
    print(f"Wrote release inventory with {ready} ready rows and {missing} missing")


if __name__ == "__main__":
    main()
