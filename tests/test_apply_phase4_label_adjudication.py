from __future__ import annotations

import csv
import json

import pytest

from slop_sftdiv.cli.apply_phase4_label_adjudication import (
    apply_adjudications,
    build_parser,
    run_apply_phase4_label_adjudication,
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


def _map_row(blind_id: str) -> dict[str, str]:
    return {
        "blind_id": blind_id,
        "annotation_id": f"phase4_ig_followup_offer::{blind_id[-4:]}",
        "feature": "phase4_ig_followup_offer",
        "candidate_label": "Follow-up assistance offer",
        "cluster": "followup_offer",
        "source": "olmo3_base_t1",
        "record_id": f"record-{blind_id}",
        "matched_text": "Sure",
        "snippet": "Sure, here is an answer.",
    }


def test_apply_phase4_label_adjudication_merges_consensus_and_adjudicated_rows(tmp_path):
    annotator_a = tmp_path / "a.csv"
    annotator_b = tmp_path / "b.csv"
    blind_map = tmp_path / "map.csv"
    adjudications = tmp_path / "adjudications.csv"
    output = tmp_path / "adjudicated.jsonl"
    _write_csv(
        annotator_a,
        [_blind_row("blind_0001", "yes", "tone"), _blind_row("blind_0002", "no", "none")],
    )
    _write_csv(
        annotator_b,
        [_blind_row("blind_0001", "yes", "tone"), _blind_row("blind_0002", "yes", "tone")],
    )
    _write_csv(blind_map, [_map_row("blind_0001"), _map_row("blind_0002")])
    _write_csv(
        adjudications,
        [
            {
                "blind_id": "blind_0002",
                "adjudicated_human_perceived_slop": "context_dependent",
                "adjudicated_shaib_taxonomy_label": "tone",
                "adjudication_notes": "needs prompt context",
            }
        ],
    )
    args = build_parser().parse_args(
        [
            "--annotator-a",
            str(annotator_a),
            "--annotator-b",
            str(annotator_b),
            "--blind-map",
            str(blind_map),
            "--adjudications",
            str(adjudications),
            "--output",
            str(output),
            "--strict",
        ]
    )

    rows = run_apply_phase4_label_adjudication(args)

    assert rows[0]["adjudication_status"] == "consensus"
    assert rows[0]["human_perceived_slop"] == "yes"
    assert rows[1]["adjudication_status"] == "adjudicated"
    assert rows[1]["human_perceived_slop"] == "context_dependent"
    assert rows[1]["notes"] == "needs prompt context"
    persisted = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert persisted[1]["annotator_a_human_perceived_slop"] == "no"
    assert persisted[1]["annotator_b_human_perceived_slop"] == "yes"


def test_apply_phase4_label_adjudication_strict_requires_disagreement_adjudication():
    with pytest.raises(ValueError, match="missing adjudication row"):
        apply_adjudications(
            [
                {
                    "blind_id": "blind_0001",
                    "human_perceived_slop": "yes",
                    "shaib_taxonomy_label": "tone",
                }
            ],
            [
                {
                    "blind_id": "blind_0001",
                    "human_perceived_slop": "no",
                    "shaib_taxonomy_label": "none",
                }
            ],
            [],
            strict=True,
        )


def test_apply_phase4_label_adjudication_rejects_unknown_adjudication_ids():
    with pytest.raises(ValueError, match="unknown blind_id values: blind_9999"):
        apply_adjudications(
            [
                {
                    "blind_id": "blind_0001",
                    "human_perceived_slop": "yes",
                    "shaib_taxonomy_label": "tone",
                }
            ],
            [
                {
                    "blind_id": "blind_0001",
                    "human_perceived_slop": "no",
                    "shaib_taxonomy_label": "none",
                }
            ],
            [
                {
                    "blind_id": "blind_9999",
                    "adjudicated_human_perceived_slop": "yes",
                    "adjudicated_shaib_taxonomy_label": "tone",
                    "adjudication_notes": "typo id",
                }
            ],
            strict=True,
        )


def test_apply_phase4_label_adjudication_rejects_stale_consensus_adjudications():
    with pytest.raises(ValueError, match="consensus blind_id values: blind_0001"):
        apply_adjudications(
            [
                {
                    "blind_id": "blind_0001",
                    "human_perceived_slop": "yes",
                    "shaib_taxonomy_label": "tone",
                }
            ],
            [
                {
                    "blind_id": "blind_0001",
                    "human_perceived_slop": "yes",
                    "shaib_taxonomy_label": "tone",
                }
            ],
            [
                {
                    "blind_id": "blind_0001",
                    "adjudicated_human_perceived_slop": "yes",
                    "adjudicated_shaib_taxonomy_label": "tone",
                    "adjudication_notes": "stale disagreement row",
                }
            ],
            strict=True,
        )
