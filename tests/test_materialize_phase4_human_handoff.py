from __future__ import annotations

import csv
from argparse import Namespace

from slop_sftdiv.cli.materialize_phase4_human_handoff import (
    run_materialize_phase4_human_handoff,
)


def _write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_phase4_human_handoff_separates_annotator_and_coordinator_files(tmp_path):
    base = tmp_path / "artifacts/phase4/modernbert_detector_combined_v2_clean"
    _write(tmp_path / "docs/experiments/phase4_human_annotation_codebook.md", "Codebook\n")
    _write(
        tmp_path / "docs/experiments/phase4_human_perceptibility_protocol.md",
        "Protocol\n",
    )
    _write(
        tmp_path / "docs/experiments/phase4_human_annotation_readiness.md",
        "Readiness\n",
    )
    _write(
        tmp_path / "docs/experiments/phase4_human_labeling_execution_checklist.md",
        "Checklist\n",
    )
    blind_csv = (
        "blind_id,matched_text,snippet,human_perceived_slop,shaib_taxonomy_label,notes\n"
        "blind_0001,Sure,\"Sure, here is an answer\",,,\n"
    )
    _write(base / "phase4_human_perceptibility_blind_pilot_20_each.csv", blind_csv)
    _write(base / "phase4_human_perceptibility_blind_full.csv", blind_csv)
    _write(
        base / "phase4_human_perceptibility_blind_pilot_20_each_map.csv",
        "blind_id,annotation_id,feature,candidate_label,cluster,source,record_id,matched_text\n"
        "blind_0001,row_1,feature,label,cluster,source,record,Sure\n",
    )
    _write(
        base / "phase4_human_perceptibility_blind_full_map.csv",
        "blind_id,annotation_id,feature,candidate_label,cluster,source,record_id,matched_text\n"
        "blind_0001,row_1,feature,label,cluster,source,record,Sure\n",
    )
    _write(base / "phase4_human_perceptibility_annotation_package.jsonl", "{}\n")
    output_dir = tmp_path / "handoff"
    summary = tmp_path / "handoff.md"

    rows = run_materialize_phase4_human_handoff(
        Namespace(root=tmp_path, output_dir=output_dir, summary_output=summary)
    )

    assert len(rows) == 14
    assert {row["status"] for row in rows} == {"ready"}
    assert (output_dir / "annotator/README.md").exists()
    assert (
        output_dir / "annotator/phase4_human_perceptibility_blind_full.csv"
    ).exists()
    assert (
        output_dir
        / "annotator/annotator_a/phase4_human_perceptibility_blind_full_annotator_a.csv"
    ).exists()
    assert (
        output_dir
        / "annotator/annotator_b/phase4_human_perceptibility_blind_full_annotator_b.csv"
    ).exists()
    assert (
        output_dir
        / "coordinator/private_maps/phase4_human_perceptibility_blind_full_map.csv"
    ).exists()
    assert (
        output_dir / "coordinator/phase4_human_labeling_execution_checklist.md"
    ).exists()
    with (output_dir / "MANIFEST.csv").open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    annotator_rows = [row for row in manifest if row["audience"] == "annotator"]
    assert [
        row["redaction_status"]
        for row in annotator_rows
        if row["bundle_path"].endswith(".csv")
    ] == ["redacted", "redacted", "redacted", "redacted", "redacted", "redacted"]
    assert "Readiness status: `ready`." in summary.read_text(encoding="utf-8")
    assert "Share only `annotator/`" in (output_dir / "README.md").read_text(
        encoding="utf-8"
    )
    annotator_readme = (output_dir / "annotator/README.md").read_text(encoding="utf-8")
    assert "Edit only `human_perceived_slop`, `shaib_taxonomy_label`, and `notes`" in annotator_readme
    assert "Protected-field edits are rejected during import." in annotator_readme
    assert "Return the filled assigned CSV only" in annotator_readme
    assert "Annotator fill rules: edit only `human_perceived_slop`" in (
        output_dir / "README.md"
    ).read_text(encoding="utf-8")
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "`uv run slop-summarize-phase4-label-agreement`" in readme
    assert "`adjudicated_human_perceived_slop`" in readme
    assert "`uv run slop-apply-phase4-label-adjudication`" in readme
    assert "`uv run slop-summarize-phase4-human-labels`" in readme
    assert "Protected-field edits are" in readme
    assert "labeling checklist" in readme


def test_phase4_human_handoff_flags_annotator_identifier_leak(tmp_path):
    base = tmp_path / "artifacts/phase4/modernbert_detector_combined_v2_clean"
    _write(tmp_path / "docs/experiments/phase4_human_annotation_codebook.md", "Codebook\n")
    _write(
        tmp_path / "docs/experiments/phase4_human_perceptibility_protocol.md",
        "Protocol\n",
    )
    _write(
        tmp_path / "docs/experiments/phase4_human_annotation_readiness.md",
        "Readiness\n",
    )
    _write(
        tmp_path / "docs/experiments/phase4_human_labeling_execution_checklist.md",
        "Checklist\n",
    )
    leaked_csv = (
        "blind_id,feature,matched_text,snippet,human_perceived_slop,shaib_taxonomy_label,notes\n"
        "blind_0001,feature,Sure,\"Sure, here is an answer\",,,\n"
    )
    _write(base / "phase4_human_perceptibility_blind_pilot_20_each.csv", leaked_csv)
    _write(base / "phase4_human_perceptibility_blind_full.csv", leaked_csv)
    _write(base / "phase4_human_perceptibility_blind_pilot_20_each_map.csv", "map\n")
    _write(base / "phase4_human_perceptibility_blind_full_map.csv", "map\n")
    _write(base / "phase4_human_perceptibility_annotation_package.jsonl", "{}\n")
    output_dir = tmp_path / "handoff"
    summary = tmp_path / "handoff.md"

    rows = run_materialize_phase4_human_handoff(
        Namespace(root=tmp_path, output_dir=output_dir, summary_output=summary)
    )

    assert any(row["redaction_status"] == "leaks_columns:feature" for row in rows)
    assert "Readiness status: `review`." in summary.read_text(encoding="utf-8")
