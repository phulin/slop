from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path.cwd()
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/phase4/modernbert_detector_combined_v2_clean/human_annotation_handoff"
)
DEFAULT_SUMMARY = Path("docs/experiments/phase4_human_annotation_handoff.md")

FORBIDDEN_ANNOTATOR_COLUMNS = {
    "annotation_id",
    "feature",
    "candidate_label",
    "cluster",
    "source",
    "record_id",
}


@dataclass(frozen=True)
class HandoffItem:
    category: str
    source: Path
    destination: Path
    audience: str


DEFAULT_ITEMS: tuple[HandoffItem, ...] = (
    HandoffItem(
        "annotator_guidance",
        Path("docs/experiments/phase4_human_annotation_codebook.md"),
        Path("annotator/phase4_human_annotation_codebook.md"),
        "annotator",
    ),
    HandoffItem(
        "annotator_sheet",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_pilot_20_each.csv"
        ),
        Path("annotator/phase4_human_perceptibility_blind_pilot_20_each.csv"),
        "annotator",
    ),
    HandoffItem(
        "annotator_sheet",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_full.csv"
        ),
        Path("annotator/phase4_human_perceptibility_blind_full.csv"),
        "annotator",
    ),
    HandoffItem(
        "annotator_assignment",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_pilot_20_each.csv"
        ),
        Path(
            "annotator/annotator_a/"
            "phase4_human_perceptibility_blind_pilot_20_each_annotator_a.csv"
        ),
        "annotator",
    ),
    HandoffItem(
        "annotator_assignment",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_pilot_20_each.csv"
        ),
        Path(
            "annotator/annotator_b/"
            "phase4_human_perceptibility_blind_pilot_20_each_annotator_b.csv"
        ),
        "annotator",
    ),
    HandoffItem(
        "annotator_assignment",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_full.csv"
        ),
        Path("annotator/annotator_a/phase4_human_perceptibility_blind_full_annotator_a.csv"),
        "annotator",
    ),
    HandoffItem(
        "annotator_assignment",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_full.csv"
        ),
        Path("annotator/annotator_b/phase4_human_perceptibility_blind_full_annotator_b.csv"),
        "annotator",
    ),
    HandoffItem(
        "coordinator_guidance",
        Path("docs/experiments/phase4_human_perceptibility_protocol.md"),
        Path("coordinator/phase4_human_perceptibility_protocol.md"),
        "coordinator",
    ),
    HandoffItem(
        "coordinator_guidance",
        Path("docs/experiments/phase4_human_annotation_readiness.md"),
        Path("coordinator/phase4_human_annotation_readiness.md"),
        "coordinator",
    ),
    HandoffItem(
        "coordinator_guidance",
        Path("docs/experiments/phase4_human_labeling_execution_checklist.md"),
        Path("coordinator/phase4_human_labeling_execution_checklist.md"),
        "coordinator",
    ),
    HandoffItem(
        "coordinator_map",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_pilot_20_each_map.csv"
        ),
        Path("coordinator/private_maps/phase4_human_perceptibility_blind_pilot_20_each_map.csv"),
        "coordinator",
    ),
    HandoffItem(
        "coordinator_map",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_full_map.csv"
        ),
        Path("coordinator/private_maps/phase4_human_perceptibility_blind_full_map.csv"),
        "coordinator",
    ),
    HandoffItem(
        "coordinator_source",
        Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_annotation_package.jsonl"
        ),
        Path("coordinator/private_maps/phase4_human_perceptibility_annotation_package.jsonl"),
        "coordinator",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a Phase 4 human-annotation handoff bundle."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    return parser


def run_materialize_phase4_human_handoff(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = args.root.resolve()
    output_dir = _resolve_path(root, args.output_dir)
    rows = materialize_handoff(root, output_dir, DEFAULT_ITEMS)
    rows.append(_write_annotator_readme(output_dir / "annotator/README.md"))
    _write_manifest(output_dir / "MANIFEST.csv", rows)
    _write_readme(output_dir / "README.md", rows)
    _write_summary(_resolve_path(root, args.summary_output), rows, output_dir=args.output_dir)
    return rows


def materialize_handoff(
    root: Path,
    output_dir: Path,
    items: tuple[HandoffItem, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        source_path = root / item.source
        destination_path = output_dir / item.destination
        status = "ready" if source_path.exists() else "missing"
        redaction_status = ""
        size_bytes = ""
        sha256 = ""
        if status == "ready":
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            size_bytes = str(destination_path.stat().st_size)
            sha256 = _sha256(destination_path)
            redaction_status = _redaction_status(destination_path, audience=item.audience)
        rows.append(
            {
                "category": item.category,
                "audience": item.audience,
                "source_path": str(item.source),
                "bundle_path": str(item.destination),
                "status": status,
                "redaction_status": redaction_status,
                "size_bytes": size_bytes,
                "sha256": sha256,
            }
        )
    return rows


def _redaction_status(path: Path, *, audience: str) -> str:
    if audience != "annotator" or path.suffix.lower() != ".csv":
        return "not_applicable"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
    leaked = sorted(FORBIDDEN_ANNOTATOR_COLUMNS & fieldnames)
    if leaked:
        return "leaks_columns:" + ",".join(leaked)
    return "redacted"


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "category",
        "audience",
        "source_path",
        "bundle_path",
        "status",
        "redaction_status",
        "size_bytes",
        "sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_annotator_readme(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 4 Annotator Quickstart",
        "",
        "You are labeling whether a highlighted span contributes to a recognizable",
        "AI slop-style effect in the local snippet. Judge style, not factual",
        "correctness.",
        "",
        "Use one assigned CSV:",
        "",
        "- `annotator_a/phase4_human_perceptibility_blind_pilot_20_each_annotator_a.csv` or",
        "  `annotator_b/phase4_human_perceptibility_blind_pilot_20_each_annotator_b.csv` for the pilot.",
        "- `annotator_a/phase4_human_perceptibility_blind_full_annotator_a.csv` or",
        "  `annotator_b/phase4_human_perceptibility_blind_full_annotator_b.csv` for the full package.",
        "",
        "Edit only `human_perceived_slop`, `shaib_taxonomy_label`, and `notes`.",
        "Do not edit `blind_id`, `matched_text`, or `snippet`.",
        "Protected-field edits are rejected during import.",
        "",
        "`human_perceived_slop` values:",
        "",
        "- `yes`: locally perceptible style excess, formula, generic assistant prose, or gratuitous closure.",
        "- `no`: natural, task-required, domain-conventional, or not stylistically salient.",
        "- `context_dependent`: possibly slop-like but the snippet is not enough to decide.",
        "- `unclear`: cannot interpret the snippet or matched text.",
        "",
        "Read `phase4_human_annotation_codebook.md` before labeling.",
        "Return the filled assigned CSV only; do not use or request coordinator maps.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "category": "annotator_guidance",
        "audience": "annotator",
        "source_path": "generated",
        "bundle_path": "annotator/README.md",
        "status": "ready",
        "redaction_status": "not_applicable",
        "size_bytes": str(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _write_readme(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ready = sum(row["status"] == "ready" for row in rows)
    missing = sum(row["status"] != "ready" for row in rows)
    annotator_redacted = sum(
        row["audience"] == "annotator" and row["redaction_status"] == "redacted"
        for row in rows
    )
    lines = [
        "# Phase 4 Human Annotation Handoff",
        "",
        "This bundle separates annotator-facing blinded files from coordinator-only",
        "maps and source identifiers. Share only `annotator/` with independent",
        "annotators. Keep `coordinator/private_maps/` private until after labels",
        "are collected, agreement is summarized, and adjudication is ready.",
        "",
        f"Ready files: `{ready}`.",
        f"Missing files: `{missing}`.",
        f"Redacted annotator CSVs: `{annotator_redacted}`.",
        "",
        "Contents:",
        "",
        "- `annotator/`: quickstart, codebook, generic blinded CSVs, and annotator-specific copies.",
        "- `coordinator/`: protocol, readiness summary, labeling checklist, private maps, and source package.",
        "- `MANIFEST.csv`: source paths, bundle paths, status, redaction status, sizes, and SHA-256 checksums.",
        "",
        "Recommended flow: send `annotator/annotator_a/` to annotator A and",
        "`annotator/annotator_b/` to annotator B, collect independent labels,",
        "then use the coordinator maps with the import/agreement/adjudication",
        "commands in the protocol.",
        "",
        "Coordinator workflow after labels return: run",
        "`uv run slop-summarize-phase4-label-agreement`, fill",
        "`adjudicated_human_perceived_slop`,",
        "`adjudicated_shaib_taxonomy_label`, and `adjudication_notes`, run",
        "`uv run slop-apply-phase4-label-adjudication`, then run",
        "`uv run slop-summarize-phase4-human-labels`.",
        "",
        "Annotator fill rules: edit only `human_perceived_slop`,",
        "`shaib_taxonomy_label`, and `notes`. Do not edit `blind_id`,",
        "`matched_text`, or `snippet`; those fields are required for import,",
        "agreement, adjudication, and unblinding. Protected-field edits are",
        "rejected during import.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(path: Path, rows: list[dict[str, Any]], *, output_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readiness = (
        "ready"
        if all(row["status"] == "ready" for row in rows)
        and all(
            row["redaction_status"] in {"redacted", "not_applicable"}
            for row in rows
        )
        else "review"
    )
    lines = [
        "# Phase 4 Human Annotation Handoff",
        "",
        f"Handoff directory: `{output_dir}`",
        "",
        f"Readiness status: `{readiness}`.",
        (
            "This handoff materializes a safer labeling package: annotator-facing "
            "files contain only blinded IDs, matched text, snippets, label columns, "
            "and notes; coordinator-only files preserve maps and source identifiers."
        ),
        "",
        "| Audience | Category | Bundle path | Status | Redaction status |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['audience']} | {row['category']} | `{row['bundle_path']}` | "
            f"{row['status']} | {row['redaction_status'] or 'not_run'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = build_parser().parse_args()
    rows = run_materialize_phase4_human_handoff(args)
    ready = sum(row["status"] == "ready" for row in rows)
    missing = sum(row["status"] != "ready" for row in rows)
    redacted = sum(
        row["audience"] == "annotator" and row["redaction_status"] == "redacted"
        for row in rows
    )
    print(
        "Wrote Phase 4 human annotation handoff with "
        f"{ready} ready files, {missing} missing, and {redacted} redacted annotator CSVs"
    )


if __name__ == "__main__":
    main()
