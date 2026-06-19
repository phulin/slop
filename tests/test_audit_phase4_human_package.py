from __future__ import annotations

import csv
import json

from slop_sftdiv.cli.audit_phase4_human_package import (
    build_parser,
    run_audit_phase4_human_package,
)


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _row(feature: str, index: int, *, annotation_id: str | None = None) -> dict[str, str]:
    return {
        "annotation_id": annotation_id or f"{feature}::{index:03d}",
        "feature": feature,
        "cluster": f"{feature}_cluster",
        "candidate_label": f"{feature} label",
        "source": "olmo3_base_t1",
        "record_id": f"record-{index}",
        "matched_text": "Sure",
        "snippet": "Sure, here is a concise answer.",
        "human_perceived_slop": "",
        "shaib_taxonomy_label": "",
        "notes": "",
    }


def test_audit_phase4_human_package_marks_balanced_blank_package_ready(tmp_path):
    package = tmp_path / "package.jsonl"
    _write_jsonl(
        package,
        [
            _row("feature_a", 0),
            _row("feature_a", 1),
            _row("feature_b", 0),
            _row("feature_b", 1),
        ],
    )
    output_csv = tmp_path / "readiness.csv"
    output_md = tmp_path / "readiness.md"
    pilot_csv = tmp_path / "pilot.csv"
    blind_pilot_csv = tmp_path / "blind_pilot.csv"
    blind_map_csv = tmp_path / "blind_map.csv"
    blind_full_csv = tmp_path / "blind_full.csv"
    blind_full_map_csv = tmp_path / "blind_full_map.csv"
    args = build_parser().parse_args(
        [
            "--input",
            str(package),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
            "--pilot-output",
            str(pilot_csv),
            "--blind-pilot-output",
            str(blind_pilot_csv),
            "--blind-pilot-map-output",
            str(blind_map_csv),
            "--blind-full-output",
            str(blind_full_csv),
            "--blind-full-map-output",
            str(blind_full_map_csv),
            "--expected-features",
            "2",
            "--expected-rows-per-feature",
            "2",
            "--pilot-per-feature",
            "1",
        ]
    )

    rows = run_audit_phase4_human_package(args)

    assert [row["feature"] for row in rows] == ["feature_a", "feature_b"]
    assert all(row["schema_status"] == "ok" for row in rows)
    assert all(row["row_count_status"] == "ok" for row in rows)
    assert "Readiness status: `ready_for_labeling`." in output_md.read_text(
        encoding="utf-8"
    )
    with pilot_csv.open(encoding="utf-8", newline="") as handle:
        pilot_rows = list(csv.DictReader(handle))
    assert [row["annotation_id"] for row in pilot_rows] == [
        "feature_a::000",
        "feature_b::000",
    ]
    with blind_pilot_csv.open(encoding="utf-8", newline="") as handle:
        blind_rows = list(csv.DictReader(handle))
    assert list(blind_rows[0]) == [
        "blind_id",
        "matched_text",
        "snippet",
        "human_perceived_slop",
        "shaib_taxonomy_label",
        "notes",
    ]
    assert {row["blind_id"] for row in blind_rows} == {"blind_0001", "blind_0002"}
    assert "feature" not in blind_rows[0]
    with blind_map_csv.open(encoding="utf-8", newline="") as handle:
        map_rows = list(csv.DictReader(handle))
    assert list(map_rows[0]) == [
        "blind_id",
        "annotation_id",
        "feature",
        "candidate_label",
        "cluster",
        "source",
        "record_id",
        "matched_text",
        "snippet",
    ]
    assert {row["annotation_id"] for row in map_rows} == {
        "feature_a::000",
        "feature_b::000",
    }
    with blind_full_csv.open(encoding="utf-8", newline="") as handle:
        blind_full_rows = list(csv.DictReader(handle))
    assert len(blind_full_rows) == 4
    assert {row["blind_id"] for row in blind_full_rows} == {
        "blind_0001",
        "blind_0002",
        "blind_0003",
        "blind_0004",
    }
    assert "feature" not in blind_full_rows[0]
    with blind_full_map_csv.open(encoding="utf-8", newline="") as handle:
        blind_full_map_rows = list(csv.DictReader(handle))
    assert {row["annotation_id"] for row in blind_full_map_rows} == {
        "feature_a::000",
        "feature_a::001",
        "feature_b::000",
        "feature_b::001",
    }
    assert "Blind pilot labeling sheet" in output_md.read_text(encoding="utf-8")
    assert "Blind full-package labeling sheet" in output_md.read_text(encoding="utf-8")


def test_audit_phase4_human_package_reports_review_for_shape_problems(tmp_path):
    package = tmp_path / "package.jsonl"
    missing_field = _row("feature_a", 1)
    missing_field.pop("snippet")
    _write_jsonl(
        package,
        [
            _row("feature_a", 0, annotation_id="dup"),
            missing_field | {"annotation_id": "dup"},
        ],
    )
    output_csv = tmp_path / "readiness.csv"
    output_md = tmp_path / "readiness.md"
    pilot_csv = tmp_path / "pilot.csv"
    blind_pilot_csv = tmp_path / "blind_pilot.csv"
    blind_map_csv = tmp_path / "blind_map.csv"
    blind_full_csv = tmp_path / "blind_full.csv"
    blind_full_map_csv = tmp_path / "blind_full_map.csv"
    args = build_parser().parse_args(
        [
            "--input",
            str(package),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
            "--pilot-output",
            str(pilot_csv),
            "--blind-pilot-output",
            str(blind_pilot_csv),
            "--blind-pilot-map-output",
            str(blind_map_csv),
            "--blind-full-output",
            str(blind_full_csv),
            "--blind-full-map-output",
            str(blind_full_map_csv),
            "--expected-features",
            "2",
            "--expected-rows-per-feature",
            "2",
        ]
    )

    rows = run_audit_phase4_human_package(args)

    assert rows[0]["schema_status"] == "missing_fields"
    text = output_md.read_text(encoding="utf-8")
    assert "Readiness status: `review`." in text
    assert "missing fields snippet" in text
    assert "duplicate annotation_id values: dup" in text
    assert "feature count 1 != expected 2" in text
