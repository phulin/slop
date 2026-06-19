from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path.cwd()
DEFAULT_OUTPUT_CSV = Path("artifacts/paper/submission/paper_reproducibility_manifest.csv")
DEFAULT_OUTPUT_MD = Path("docs/experiments/paper_reproducibility_manifest.md")


@dataclass(frozen=True)
class ManifestItem:
    category: str
    path: Path


DEFAULT_ITEMS: tuple[ManifestItem, ...] = (
    ManifestItem("paper_control", Path("EXPERIMENTS.md")),
    ManifestItem("paper_control", Path("docs/experiments/paper_package_index.md")),
    ManifestItem("paper_control", Path("docs/experiments/paper_claim_matrix.md")),
    ManifestItem("paper_control", Path("docs/experiments/paper_claim_evidence_map.md")),
    ManifestItem("paper_control", Path("docs/experiments/paper_reader_glossary.md")),
    ManifestItem("paper_control", Path("docs/experiments/paper_measurement_synthesis.md")),
    ManifestItem("paper_control", Path("docs/experiments/paper_readiness_audit.md")),
    ManifestItem("paper_control", Path("docs/experiments/paper_scaffold.md")),
    ManifestItem("paper_control", Path("docs/experiments/paper_submission_exit_audit.md")),
    ManifestItem("paper_control", Path("docs/experiments/paper_venue_decision_checklist.md")),
    ManifestItem("paper_control", Path("docs/experiments/paper_venue_adaptation_runbook.md")),
    ManifestItem("paper_control", Path("docs/experiments/paper_venue_sizing_inventory.md")),
    ManifestItem("paper_control", Path("docs/experiments/paper_reproduction_runbook.md")),
    ManifestItem("paper_control", Path("docs/experiments/paper_reviewer_faq.md")),
    ManifestItem("paper_control", Path("docs/experiments/paper_release_inventory.md")),
    ManifestItem("paper_control", Path("docs/experiments/paper_source_provenance_audit.md")),
    ManifestItem("manuscript", Path("docs/experiments/paper_manuscript_draft.md")),
    ManifestItem("manuscript", Path("docs/experiments/paper_methods_draft.md")),
    ManifestItem("manuscript", Path("docs/experiments/paper_results_draft.md")),
    ManifestItem("manuscript", Path("docs/experiments/paper_intro_discussion_draft.md")),
    ManifestItem("manuscript", Path("docs/experiments/paper_caption_appendix_draft.md")),
    ManifestItem("manuscript", Path("docs/experiments/paper_camera_ready_tables.md")),
    ManifestItem("manuscript", Path("docs/experiments/paper_latex_table_drafts.tex")),
    ManifestItem("manuscript_audit", Path("docs/experiments/paper_claim_language_audit.md")),
    ManifestItem("manuscript_audit", Path("docs/experiments/paper_claim_evidence_audit.md")),
    ManifestItem("manuscript_audit", Path("docs/experiments/paper_abstract_conclusion_audit.md")),
    ManifestItem("manuscript_audit", Path("docs/experiments/paper_manuscript_structure_audit.md")),
    ManifestItem("manuscript_audit", Path("docs/experiments/paper_limitations_audit.md")),
    ManifestItem("manuscript_audit", Path("docs/experiments/paper_terminology_audit.md")),
    ManifestItem("manuscript_audit", Path("docs/experiments/paper_numeric_claims_audit.md")),
    ManifestItem("manuscript_audit", Path("docs/experiments/paper_review_hygiene_audit.md")),
    ManifestItem("source_note", Path("docs/experiments/source_card_notes.md")),
    ManifestItem("reference", Path("docs/experiments/paper_citation_plan.md")),
    ManifestItem("reference", Path("docs/experiments/paper_citation_audit.md")),
    ManifestItem("reference", Path("docs/experiments/paper_reference_sources.md")),
    ManifestItem("reference", Path("docs/experiments/paper_references.bib")),
    ManifestItem("release_code", Path("src/slop_sftdiv/features/tier1_matchers.py")),
    ManifestItem("release_code", Path("src/slop_sftdiv/features/eqbench_slop.py")),
    ManifestItem("release_code", Path("src/slop_sftdiv/features/pybiber_full.py")),
    ManifestItem("release_code", Path("src/slop_sftdiv/propensity.py")),
    ManifestItem("release_code", Path("src/slop_sftdiv/cli/teacher_forced_propensity.py")),
    ManifestItem("release_code", Path("src/slop_sftdiv/cli/free_running_emission.py")),
    ManifestItem("release_code", Path("src/slop_sftdiv/generation_text.py")),
    ManifestItem("release_code", Path("src/slop_sftdiv/cli/import_phase4_blind_labels.py")),
    ManifestItem("release_code", Path("src/slop_sftdiv/cli/summarize_phase4_label_agreement.py")),
    ManifestItem("release_code", Path("src/slop_sftdiv/cli/apply_phase4_label_adjudication.py")),
    ManifestItem("release_code", Path("src/slop_sftdiv/cli/check_paper_package.py")),
    ManifestItem("release_code", Path("src/slop_sftdiv/cli/materialize_paper_submission_bundle.py")),
    ManifestItem("release_code", Path("src/slop_sftdiv/cli/audit_paper_reproducibility_manifest.py")),
    ManifestItem("figure", Path("artifacts/paper/figures/figure1_pybiber_register_selected.svg")),
    ManifestItem("figure", Path("artifacts/paper/figures/figure2_eqbench_stage_scores.svg")),
    ManifestItem("figure", Path("artifacts/paper/figures/figure3_olmo_slop_lexicon_views.svg")),
    ManifestItem("figure", Path("artifacts/paper/figures/figure4_cross_ladder_af_scatter.svg")),
    ManifestItem("figure", Path("artifacts/paper/figures/figure5_phase4_tier3_raw_af.svg")),
    ManifestItem(
        "figure_export",
        Path("artifacts/paper/figures/submission_exports/figure1_pybiber_register_selected.pdf"),
    ),
    ManifestItem(
        "figure_export",
        Path("artifacts/paper/figures/submission_exports/figure1_pybiber_register_selected.png"),
    ),
    ManifestItem(
        "figure_export",
        Path("artifacts/paper/figures/submission_exports/figure2_eqbench_stage_scores.pdf"),
    ),
    ManifestItem(
        "figure_export",
        Path("artifacts/paper/figures/submission_exports/figure2_eqbench_stage_scores.png"),
    ),
    ManifestItem(
        "figure_export",
        Path("artifacts/paper/figures/submission_exports/figure3_olmo_slop_lexicon_views.pdf"),
    ),
    ManifestItem(
        "figure_export",
        Path("artifacts/paper/figures/submission_exports/figure3_olmo_slop_lexicon_views.png"),
    ),
    ManifestItem(
        "figure_export",
        Path("artifacts/paper/figures/submission_exports/figure4_cross_ladder_af_scatter.pdf"),
    ),
    ManifestItem(
        "figure_export",
        Path("artifacts/paper/figures/submission_exports/figure4_cross_ladder_af_scatter.png"),
    ),
    ManifestItem(
        "figure_export",
        Path("artifacts/paper/figures/submission_exports/figure5_phase4_tier3_raw_af.pdf"),
    ),
    ManifestItem(
        "figure_export",
        Path("artifacts/paper/figures/submission_exports/figure5_phase4_tier3_raw_af.png"),
    ),
    ManifestItem(
        "figure_review",
        Path("artifacts/paper/figures/visual_review_png/figure1_pybiber_register_selected.png"),
    ),
    ManifestItem(
        "figure_review",
        Path("artifacts/paper/figures/visual_review_png/figure2_eqbench_stage_scores.png"),
    ),
    ManifestItem(
        "figure_review",
        Path("artifacts/paper/figures/visual_review_png/figure3_olmo_slop_lexicon_views.png"),
    ),
    ManifestItem(
        "figure_review",
        Path("artifacts/paper/figures/visual_review_png/figure4_cross_ladder_af_scatter.png"),
    ),
    ManifestItem(
        "figure_review",
        Path("artifacts/paper/figures/visual_review_png/figure5_phase4_tier3_raw_af.png"),
    ),
    ManifestItem("paper_audit_csv", Path("artifacts/paper/claims/paper_claim_language_audit.csv")),
    ManifestItem("paper_audit_csv", Path("artifacts/paper/claims/paper_claim_evidence_audit.csv")),
    ManifestItem(
        "paper_audit_csv",
        Path("artifacts/paper/manuscript/paper_abstract_conclusion_audit.csv"),
    ),
    ManifestItem(
        "paper_audit_csv",
        Path("artifacts/paper/manuscript/paper_manuscript_structure_audit.csv"),
    ),
    ManifestItem("paper_audit_csv", Path("artifacts/paper/manuscript/paper_limitations_audit.csv")),
    ManifestItem(
        "paper_audit_csv",
        Path("artifacts/paper/manuscript/paper_terminology_audit.csv"),
    ),
    ManifestItem(
        "paper_audit_csv",
        Path("artifacts/paper/manuscript/paper_numeric_claims_audit.csv"),
    ),
    ManifestItem(
        "paper_audit_csv",
        Path("artifacts/paper/manuscript/paper_review_hygiene_audit.csv"),
    ),
    ManifestItem(
        "paper_audit_csv",
        Path("artifacts/paper/submission/paper_submission_exit_audit.csv"),
    ),
    ManifestItem(
        "paper_audit_csv",
        Path("artifacts/paper/submission/paper_venue_decision_checklist.csv"),
    ),
    ManifestItem(
        "paper_audit_csv",
        Path("artifacts/paper/submission/paper_release_inventory.csv"),
    ),
    ManifestItem(
        "paper_audit_csv",
        Path("artifacts/paper/submission/paper_source_provenance_audit.csv"),
    ),
    ManifestItem(
        "paper_audit_csv",
        Path("artifacts/paper/references/paper_citation_audit.csv"),
    ),
    ManifestItem(
        "paper_audit_csv",
        Path("artifacts/paper/figures/paper_figure_readiness_audit.csv"),
    ),
    ManifestItem(
        "paper_audit_csv",
        Path("artifacts/paper/figures/paper_figure_visual_review.csv"),
    ),
    ManifestItem(
        "paper_audit_csv",
        Path("artifacts/paper/tables/paper_table_readiness_audit.csv"),
    ),
    ManifestItem(
        "paper_audit_csv",
        Path("artifacts/paper/tables/paper_table_typography_review.csv"),
    ),
    ManifestItem(
        "submission_bundle",
        Path("artifacts/paper/submission/draft_bundle/README.md"),
    ),
    ManifestItem(
        "submission_bundle",
        Path("artifacts/paper/submission/draft_bundle/MANIFEST.csv"),
    ),
    ManifestItem(
        "phase1_evidence",
        Path("docs/experiments/phase1_pybiber_register_analysis.md"),
    ),
    ManifestItem(
        "phase1_evidence",
        Path("artifacts/stage1/census/phase1_pybiber_register_means.csv"),
    ),
    ManifestItem(
        "phase1_evidence",
        Path("artifacts/stage1/census/phase1_pybiber_register_deltas.csv"),
    ),
    ManifestItem(
        "phase1_evidence",
        Path("artifacts/stage1/census/phase1_pybiber_register_intervals.csv"),
    ),
    ManifestItem(
        "eqbench_evidence",
        Path("artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv"),
    ),
    ManifestItem(
        "eqbench_evidence",
        Path("artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.md"),
    ),
    ManifestItem(
        "phase4_evidence",
        Path("docs/experiments/phase4_human_perceptibility_protocol.md"),
    ),
    ManifestItem(
        "phase4_evidence",
        Path("docs/experiments/phase4_human_annotation_codebook.md"),
    ),
    ManifestItem(
        "phase4_evidence",
        Path("docs/experiments/phase4_human_annotation_readiness.md"),
    ),
    ManifestItem(
        "phase4_evidence",
        Path("docs/experiments/phase4_human_annotation_handoff.md"),
    ),
    ManifestItem(
        "phase4_evidence",
        Path("docs/experiments/phase4_human_labeling_execution_checklist.md"),
    ),
    ManifestItem(
        "phase4_evidence",
        Path("docs/experiments/phase4_human_perceptibility_summary.md"),
    ),
    ManifestItem(
        "phase4_evidence",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_annotation_readiness.csv"
        ),
    ),
    ManifestItem(
        "phase4_evidence",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_pilot_20_each.csv"
        ),
    ),
    ManifestItem(
        "phase4_evidence",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_pilot_20_each.csv"
        ),
    ),
    ManifestItem(
        "phase4_evidence",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_pilot_20_each_map.csv"
        ),
    ),
    ManifestItem(
        "phase4_evidence",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_full.csv"
        ),
    ),
    ManifestItem(
        "phase4_evidence",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_full_map.csv"
        ),
    ),
    ManifestItem(
        "phase4_evidence",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_summary.csv"
        ),
    ),
    ManifestItem(
        "phase4_evidence",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "human_annotation_handoff/README.md"
        ),
    ),
    ManifestItem(
        "phase4_evidence",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "human_annotation_handoff/MANIFEST.csv"
        ),
    ),
    ManifestItem(
        "phase4_evidence",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "human_annotation_handoff/annotator/"
            "README.md"
        ),
    ),
    ManifestItem(
        "phase4_evidence",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "human_annotation_handoff/coordinator/"
            "phase4_human_labeling_execution_checklist.md"
        ),
    ),
    ManifestItem(
        "phase4_evidence",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "human_annotation_handoff/annotator/"
            "phase4_human_perceptibility_blind_pilot_20_each.csv"
        ),
    ),
    ManifestItem(
        "phase4_evidence",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "human_annotation_handoff/annotator/annotator_a/"
            "phase4_human_perceptibility_blind_pilot_20_each_annotator_a.csv"
        ),
    ),
    ManifestItem(
        "phase4_evidence",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "human_annotation_handoff/annotator/annotator_b/"
            "phase4_human_perceptibility_blind_pilot_20_each_annotator_b.csv"
        ),
    ),
    ManifestItem(
        "phase4_evidence",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "human_annotation_handoff/annotator/phase4_human_perceptibility_blind_full.csv"
        ),
    ),
    ManifestItem(
        "phase4_evidence",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "human_annotation_handoff/annotator/annotator_a/"
            "phase4_human_perceptibility_blind_full_annotator_a.csv"
        ),
    ),
    ManifestItem(
        "phase4_evidence",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "human_annotation_handoff/annotator/annotator_b/"
            "phase4_human_perceptibility_blind_full_annotator_b.csv"
        ),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a reproducibility manifest for paper-facing docs and artifacts."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument(
        "--path",
        type=Path,
        action="append",
        default=None,
        help="Optional path to include instead of the default paper bundle list.",
    )
    return parser


def run_audit_paper_reproducibility_manifest(
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    root = args.root.resolve()
    items = _items_from_args(args.path)
    rows = build_manifest_rows(root, items)
    output_csv = _resolve_path(root, args.output_csv)
    output_md = _resolve_path(root, args.output_md)
    _write_csv(output_csv, rows)
    _write_markdown(output_md, rows, output_csv=args.output_csv)
    return rows


def build_manifest_rows(root: Path, items: tuple[ManifestItem, ...]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        relative = item.path.as_posix()
        resolved = _resolve_path(root, item.path)
        if resolved.exists() and resolved.is_file():
            rows.append(
                {
                    "category": item.category,
                    "path": relative,
                    "status": "ready",
                    "size_bytes": str(resolved.stat().st_size),
                    "sha256": _sha256(resolved),
                }
            )
        else:
            rows.append(
                {
                    "category": item.category,
                    "path": relative,
                    "status": "missing",
                    "size_bytes": "",
                    "sha256": "",
                }
            )
    return rows


def _items_from_args(paths: list[Path] | None) -> tuple[ManifestItem, ...]:
    if not paths:
        return DEFAULT_ITEMS
    return tuple(ManifestItem(_category_for_path(path), path) for path in paths)


def _category_for_path(path: Path) -> str:
    path_text = path.as_posix()
    if path_text.startswith("docs/experiments/"):
        return "document"
    if path_text.startswith("src/"):
        return "release_code"
    if path_text.startswith("artifacts/paper/figures/"):
        return "figure"
    if path_text.startswith("artifacts/"):
        return "artifact"
    if path_text.endswith(".bib"):
        return "reference"
    return "paper_control"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["category", "path", "status", "size_bytes", "sha256"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readiness = "ready" if all(row["status"] == "ready" for row in rows) else "review"
    categories = sorted({row["category"] for row in rows})
    lines = [
        "# Paper Reproducibility Manifest",
        "",
        f"Machine-readable manifest: `{output_csv}`",
        "",
        f"Readiness status: `{readiness}`.",
        (
            "This manifest records the paper-facing evidence bundle with file sizes "
            "and SHA-256 checksums. It is a reproducibility index, not a venue "
            "submission certificate."
        ),
        "",
        "| Category | Ready | Missing |",
        "|---|---:|---:|",
    ]
    for category in categories:
        category_rows = [row for row in rows if row["category"] == category]
        ready = sum(row["status"] == "ready" for row in category_rows)
        missing = sum(row["status"] != "ready" for row in category_rows)
        lines.append(f"| {category} | {ready} | {missing} |")
    lines.extend(
        [
            "",
            "| Category | Path | Status | Bytes | SHA-256 |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in rows:
        sha = row["sha256"] or "missing"
        size = row["size_bytes"] or "missing"
        lines.append(
            f"| {row['category']} | `{row['path']}` | {row['status']} | {size} | {sha} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_audit_paper_reproducibility_manifest(args)
    print(f"Wrote {len(rows)} reproducibility manifest rows to {args.output_csv}")
    print(f"Wrote reproducibility manifest summary to {args.output_md}")


if __name__ == "__main__":
    main()
