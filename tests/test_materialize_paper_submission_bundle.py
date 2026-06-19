from __future__ import annotations

import csv
from argparse import Namespace

from slop_sftdiv.cli.materialize_paper_submission_bundle import (
    FIGURE_STEMS,
    run_materialize_paper_submission_bundle,
)


def _write(path, text: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")


def test_materialize_paper_submission_bundle_copies_expected_files(tmp_path):
    for relative in [
        "docs/experiments/paper_package_index.md",
        "docs/experiments/paper_manuscript_draft.md",
        "docs/experiments/paper_caption_appendix_draft.md",
        "docs/experiments/paper_camera_ready_tables.md",
        "docs/experiments/paper_latex_table_drafts.tex",
        "docs/experiments/paper_references.bib",
        "docs/experiments/paper_claim_matrix.md",
        "docs/experiments/paper_claim_evidence_map.md",
        "docs/experiments/paper_reader_glossary.md",
        "docs/experiments/paper_measurement_synthesis.md",
        "docs/experiments/paper_submission_exit_audit.md",
        "docs/experiments/paper_review_hygiene_audit.md",
        "docs/experiments/paper_release_inventory.md",
        "docs/experiments/paper_source_provenance_audit.md",
        "docs/experiments/paper_venue_adaptation_runbook.md",
        "docs/experiments/paper_venue_sizing_inventory.md",
    ]:
        _write(tmp_path / relative, f"{relative}\n")
    for stem in FIGURE_STEMS:
        _write(tmp_path / f"artifacts/paper/figures/{stem}.svg", f"<svg>{stem}</svg>\n")
        _write(tmp_path / f"artifacts/paper/figures/submission_exports/{stem}.pdf", b"%PDF\n")
        _write(
            tmp_path / f"artifacts/paper/figures/submission_exports/{stem}.png",
            b"\x89PNG\r\n\x1a\n",
        )

    output_dir = tmp_path / "bundle"
    rows = run_materialize_paper_submission_bundle(
        Namespace(root=tmp_path, output_dir=output_dir)
    )

    assert len(rows) == 31
    assert {row["status"] for row in rows} == {"ready"}
    assert (output_dir / "manuscript/paper_manuscript_draft.md").exists()
    assert (output_dir / "tables/paper_latex_table_drafts.tex").exists()
    assert (output_dir / "references/paper_references.bib").exists()
    assert (output_dir / "review_controls/paper_package_index.md").exists()
    assert (output_dir / "review_controls/paper_reader_glossary.md").exists()
    assert (output_dir / "review_controls/paper_measurement_synthesis.md").exists()
    assert (output_dir / "review_controls/paper_venue_sizing_inventory.md").exists()
    assert (output_dir / "figures/svg/figure1_pybiber_register_selected.svg").exists()
    assert (output_dir / "figures/pdf/figure1_pybiber_register_selected.pdf").exists()
    assert (output_dir / "figures/png/figure1_pybiber_register_selected.png").exists()
    assert "venue-neutral draft bundle" in (output_dir / "README.md").read_text(
        encoding="utf-8"
    )
    with (output_dir / "MANIFEST.csv").open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert len(manifest) == 31
    assert all(len(row["sha256"]) == 64 for row in manifest)


def test_materialize_paper_submission_bundle_marks_missing_sources(tmp_path):
    output_dir = tmp_path / "bundle"

    rows = run_materialize_paper_submission_bundle(
        Namespace(root=tmp_path, output_dir=output_dir)
    )

    assert {row["status"] for row in rows} == {"missing"}
    assert "Missing files: `31`." in (output_dir / "README.md").read_text(
        encoding="utf-8"
    )
