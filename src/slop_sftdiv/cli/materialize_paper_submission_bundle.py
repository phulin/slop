from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path.cwd()
DEFAULT_OUTPUT_DIR = Path("artifacts/paper/submission/draft_bundle")


@dataclass(frozen=True)
class BundleItem:
    category: str
    source: Path
    destination: Path


FIGURE_STEMS = (
    "figure1_pybiber_register_selected",
    "figure2_eqbench_stage_scores",
    "figure3_olmo_slop_lexicon_views",
    "figure4_cross_ladder_af_scatter",
    "figure5_phase4_tier3_raw_af",
)


DEFAULT_ITEMS: tuple[BundleItem, ...] = (
    BundleItem(
        "readiness",
        Path("docs/experiments/paper_package_index.md"),
        Path("review_controls/paper_package_index.md"),
    ),
    BundleItem(
        "manuscript",
        Path("docs/experiments/paper_manuscript_draft.md"),
        Path("manuscript/paper_manuscript_draft.md"),
    ),
    BundleItem(
        "manuscript",
        Path("docs/experiments/paper_caption_appendix_draft.md"),
        Path("manuscript/paper_caption_appendix_draft.md"),
    ),
    BundleItem(
        "tables",
        Path("docs/experiments/paper_camera_ready_tables.md"),
        Path("tables/paper_camera_ready_tables.md"),
    ),
    BundleItem(
        "tables",
        Path("docs/experiments/paper_latex_table_drafts.tex"),
        Path("tables/paper_latex_table_drafts.tex"),
    ),
    BundleItem(
        "references",
        Path("docs/experiments/paper_references.bib"),
        Path("references/paper_references.bib"),
    ),
    BundleItem(
        "claim_control",
        Path("docs/experiments/paper_claim_matrix.md"),
        Path("review_controls/paper_claim_matrix.md"),
    ),
    BundleItem(
        "claim_control",
        Path("docs/experiments/paper_claim_evidence_map.md"),
        Path("review_controls/paper_claim_evidence_map.md"),
    ),
    BundleItem(
        "claim_control",
        Path("docs/experiments/paper_reader_glossary.md"),
        Path("review_controls/paper_reader_glossary.md"),
    ),
    BundleItem(
        "claim_control",
        Path("docs/experiments/paper_measurement_synthesis.md"),
        Path("review_controls/paper_measurement_synthesis.md"),
    ),
    BundleItem(
        "readiness",
        Path("docs/experiments/paper_submission_exit_audit.md"),
        Path("review_controls/paper_submission_exit_audit.md"),
    ),
    BundleItem(
        "readiness",
        Path("docs/experiments/paper_review_hygiene_audit.md"),
        Path("review_controls/paper_review_hygiene_audit.md"),
    ),
    BundleItem(
        "readiness",
        Path("docs/experiments/paper_release_inventory.md"),
        Path("review_controls/paper_release_inventory.md"),
    ),
    BundleItem(
        "readiness",
        Path("docs/experiments/paper_source_provenance_audit.md"),
        Path("review_controls/paper_source_provenance_audit.md"),
    ),
    BundleItem(
        "readiness",
        Path("docs/experiments/paper_venue_adaptation_runbook.md"),
        Path("review_controls/paper_venue_adaptation_runbook.md"),
    ),
    BundleItem(
        "readiness",
        Path("docs/experiments/paper_venue_sizing_inventory.md"),
        Path("review_controls/paper_venue_sizing_inventory.md"),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a venue-neutral paper submission draft bundle."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def run_materialize_paper_submission_bundle(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = args.root.resolve()
    output_dir = _resolve_output(root, args.output_dir)
    items = list(DEFAULT_ITEMS) + _figure_items()
    rows = materialize_bundle(root, output_dir, items)
    _write_manifest(output_dir / "MANIFEST.csv", rows)
    _write_readme(output_dir / "README.md", rows)
    return rows


def materialize_bundle(
    root: Path,
    output_dir: Path,
    items: list[BundleItem],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        source_path = root / item.source
        destination_path = output_dir / item.destination
        status = "ready" if source_path.exists() else "missing"
        size_bytes = ""
        sha256 = ""
        if status == "ready":
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            size_bytes = str(destination_path.stat().st_size)
            sha256 = _sha256(destination_path)
        rows.append(
            {
                "category": item.category,
                "source_path": str(item.source),
                "bundle_path": str(item.destination),
                "status": status,
                "size_bytes": size_bytes,
                "sha256": sha256,
            }
        )
    return rows


def _figure_items() -> list[BundleItem]:
    items: list[BundleItem] = []
    for stem in FIGURE_STEMS:
        items.extend(
            [
                BundleItem(
                    "figure_svg",
                    Path(f"artifacts/paper/figures/{stem}.svg"),
                    Path(f"figures/svg/{stem}.svg"),
                ),
                BundleItem(
                    "figure_pdf",
                    Path(f"artifacts/paper/figures/submission_exports/{stem}.pdf"),
                    Path(f"figures/pdf/{stem}.pdf"),
                ),
                BundleItem(
                    "figure_png",
                    Path(f"artifacts/paper/figures/submission_exports/{stem}.png"),
                    Path(f"figures/png/{stem}.png"),
                ),
            ]
        )
    return items


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["category", "source_path", "bundle_path", "status", "size_bytes", "sha256"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ready = sum(row["status"] == "ready" for row in rows)
    missing = sum(row["status"] != "ready" for row in rows)
    lines = [
        "# Paper Submission Draft Bundle",
        "",
        "This is a venue-neutral draft bundle assembled from the current paper package.",
        "It is not a submission certificate and does not resolve target-venue sizing,",
        "template, or Phase 4 human-labeling blockers.",
        "",
        f"Ready files: `{ready}`.",
        f"Missing files: `{missing}`.",
        "",
        "Contents:",
        "",
        "- `manuscript/`: current manuscript and caption/appendix draft.",
        "- `tables/`: Markdown and generic LaTeX table drafts.",
        "- `references/`: BibTeX file.",
        "- `figures/`: SVG sources plus PDF and PNG export candidates.",
        "- `review_controls/`: package index plus claim/readiness controls that bound paper wording.",
        "- `MANIFEST.csv`: bundle paths, source paths, sizes, and SHA-256 checksums.",
        "",
        "Before submission, follow `paper_venue_adaptation_runbook.md`, resolve the",
        "venue-decision checklist, and rerun `uv run slop-check-paper-package --root /home/user/slop`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_output(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> None:
    args = build_parser().parse_args()
    rows = run_materialize_paper_submission_bundle(args)
    ready = sum(row["status"] == "ready" for row in rows)
    missing = sum(row["status"] != "ready" for row in rows)
    print(f"Wrote paper submission draft bundle with {ready} ready files and {missing} missing")


if __name__ == "__main__":
    main()
