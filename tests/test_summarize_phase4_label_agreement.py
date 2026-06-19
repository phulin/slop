from __future__ import annotations

import csv

import pytest

from slop_sftdiv.cli.summarize_phase4_label_agreement import (
    build_parser,
    run_summarize_phase4_label_agreement,
    summarize_agreement,
)


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _blind_row(blind_id: str, perception: str, taxonomy: str = "none") -> dict[str, str]:
    return {
        "blind_id": blind_id,
        "matched_text": "Sure",
        "snippet": "Sure, here is an answer.",
        "human_perceived_slop": perception,
        "shaib_taxonomy_label": taxonomy,
        "notes": "",
    }


def _map_row(blind_id: str, feature: str = "phase4_ig_followup_offer") -> dict[str, str]:
    return {
        "blind_id": blind_id,
        "annotation_id": f"{feature}::{blind_id[-4:]}",
        "feature": feature,
        "candidate_label": "Follow-up assistance offer",
        "cluster": "followup_offer",
        "source": "olmo3_base_t1",
        "record_id": f"record-{blind_id}",
        "matched_text": "Sure",
        "snippet": "Sure, here is an answer.",
    }


def test_summarize_phase4_label_agreement_writes_summary_and_disagreements(tmp_path):
    annotator_a = tmp_path / "a.csv"
    annotator_b = tmp_path / "b.csv"
    blind_map = tmp_path / "map.csv"
    output = tmp_path / "agreement.csv"
    disagreements = tmp_path / "disagreements.csv"
    summary = tmp_path / "agreement.md"
    _write_csv(
        annotator_a,
        [
            _blind_row("blind_0001", "yes", "tone"),
            _blind_row("blind_0002", "no", "none"),
            _blind_row("blind_0003", "context_dependent", "coherence"),
        ],
    )
    _write_csv(
        annotator_b,
        [
            _blind_row("blind_0001", "yes", "tone"),
            _blind_row("blind_0002", "yes", "tone"),
            _blind_row("blind_0003", "context_dependent", "coherence"),
        ],
    )
    _write_csv(blind_map, [_map_row("blind_0001"), _map_row("blind_0002"), _map_row("blind_0003")])
    args = build_parser().parse_args(
        [
            "--annotator-a",
            str(annotator_a),
            "--annotator-b",
            str(annotator_b),
            "--blind-map",
            str(blind_map),
            "--output",
            str(output),
            "--disagreements-output",
            str(disagreements),
            "--summary-output",
            str(summary),
            "--strict",
        ]
    )

    rows = run_summarize_phase4_label_agreement(args)

    all_row = next(row for row in rows if row["feature"] == "ALL")
    assert all_row["both_labeled"] == 3
    assert all_row["perception_agreement_count"] == 2
    assert all_row["perception_agreement_rate"] == "0.667"
    assert all_row["yes_no_conflicts"] == 1
    with disagreements.open(encoding="utf-8", newline="") as handle:
        disagreement_rows = list(csv.DictReader(handle))
    assert [row["blind_id"] for row in disagreement_rows] == ["blind_0002"]
    assert "Perception disagreement rows: `1`." in summary.read_text(encoding="utf-8")


def test_summarize_phase4_label_agreement_rejects_mismatched_ids():
    with pytest.raises(ValueError, match="different blind_id sets"):
        summarize_agreement(
            [
                {
                    "blind_id": "blind_0001",
                    "feature": "feature",
                    "candidate_label": "Feature",
                    "human_perceived_slop": "yes",
                    "shaib_taxonomy_label": "tone",
                }
            ],
            [
                {
                    "blind_id": "blind_9999",
                    "feature": "feature",
                    "candidate_label": "Feature",
                    "human_perceived_slop": "yes",
                    "shaib_taxonomy_label": "tone",
                }
            ],
        )
