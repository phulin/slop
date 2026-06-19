from __future__ import annotations

import csv
import json

import pytest

from slop_sftdiv.cli.import_phase4_blind_labels import (
    build_parser,
    import_blind_rows,
    run_import_phase4_blind_labels,
)


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_import_phase4_blind_labels_writes_canonical_jsonl(tmp_path):
    blind_labels = tmp_path / "blind.csv"
    blind_map = tmp_path / "map.csv"
    output = tmp_path / "labeled.jsonl"
    _write_csv(
        blind_labels,
        [
            {
                "blind_id": "blind_0001",
                "matched_text": "Also",
                "snippet": "Also, this adds another step.",
                "human_perceived_slop": "Yes",
                "shaib_taxonomy_label": "Structure",
                "notes": "formulaic transition",
            }
        ],
    )
    _write_csv(
        blind_map,
        [
            {
                "blind_id": "blind_0001",
                "annotation_id": "phase4_ig_additive_transition::000",
                "feature": "phase4_ig_additive_transition",
                "candidate_label": "Additive/explanatory transition",
                "cluster": "additive_transition",
                "source": "olmo3_base_t1",
                "record_id": "record-1",
                "matched_text": "Also",
                "snippet": "Also, this adds another step.",
            }
        ],
    )
    args = build_parser().parse_args(
        [
            "--blind-labels",
            str(blind_labels),
            "--blind-map",
            str(blind_map),
            "--output",
            str(output),
            "--strict",
        ]
    )

    rows = run_import_phase4_blind_labels(args)

    assert rows[0]["feature"] == "phase4_ig_additive_transition"
    assert rows[0]["human_perceived_slop"] == "yes"
    assert rows[0]["shaib_taxonomy_label"] == "structure"
    persisted = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert persisted[0]["annotation_id"] == "phase4_ig_additive_transition::000"
    assert persisted[0]["blind_id"] == "blind_0001"


def test_import_phase4_blind_labels_strict_rejects_missing_label():
    with pytest.raises(ValueError, match="missing human_perceived_slop"):
        import_blind_rows(
            [
                {
                    "blind_id": "blind_0001",
                    "matched_text": "Sure",
                    "snippet": "Sure, here is an answer.",
                    "human_perceived_slop": "",
                    "shaib_taxonomy_label": "",
                    "notes": "",
                }
            ],
            [
                {
                    "blind_id": "blind_0001",
                    "annotation_id": "feature::000",
                    "feature": "feature",
                    "candidate_label": "Feature",
                    "cluster": "cluster",
                    "source": "source",
                    "record_id": "record",
                    "matched_text": "Sure",
                    "snippet": "Sure, here is an answer.",
                }
            ],
            strict=True,
        )


def test_import_phase4_blind_labels_rejects_protected_field_edits():
    with pytest.raises(ValueError, match="protected field 'snippet' was edited"):
        import_blind_rows(
            [
                {
                    "blind_id": "blind_0001",
                    "matched_text": "Sure",
                    "snippet": "Edited snippet",
                    "human_perceived_slop": "no",
                    "shaib_taxonomy_label": "none",
                    "notes": "",
                }
            ],
            [
                {
                    "blind_id": "blind_0001",
                    "annotation_id": "feature::000",
                    "feature": "feature",
                    "candidate_label": "Feature",
                    "cluster": "cluster",
                    "source": "source",
                    "record_id": "record",
                    "matched_text": "Sure",
                    "snippet": "Sure, here is an answer.",
                }
            ],
            strict=False,
        )


def test_import_phase4_blind_labels_requires_protected_map_fields():
    with pytest.raises(ValueError, match="missing required fields snippet"):
        import_blind_rows(
            [
                {
                    "blind_id": "blind_0001",
                    "matched_text": "Sure",
                    "snippet": "Sure, here is an answer.",
                    "human_perceived_slop": "no",
                    "shaib_taxonomy_label": "none",
                    "notes": "",
                }
            ],
            [
                {
                    "blind_id": "blind_0001",
                    "annotation_id": "feature::000",
                    "feature": "feature",
                    "candidate_label": "Feature",
                    "cluster": "cluster",
                    "source": "source",
                    "record_id": "record",
                    "matched_text": "Sure",
                }
            ],
            strict=False,
        )


def test_import_phase4_blind_labels_rejects_unmapped_blind_id():
    with pytest.raises(ValueError, match="no map row"):
        import_blind_rows(
            [
                {
                    "blind_id": "blind_9999",
                    "matched_text": "Sure",
                    "snippet": "Sure, here is an answer.",
                    "human_perceived_slop": "no",
                    "shaib_taxonomy_label": "none",
                    "notes": "",
                }
            ],
            [],
            strict=False,
        )
