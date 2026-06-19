from __future__ import annotations

import csv
import json

import pytest

from slop_sftdiv.cli.summarize_phase4_human_labels import (
    build_parser,
    run_summarize_phase4_human_labels,
)


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_summarize_phase4_human_labels_handles_blank_package(tmp_path):
    package = tmp_path / "labels.jsonl"
    _write_jsonl(
        package,
        [
            {
                "feature": "phase4_ig_followup_offer",
                "candidate_label": "Follow-up offer/friendly closure",
                "cluster": "followup_offer",
                "human_perceived_slop": "",
                "shaib_taxonomy_label": "",
            },
            {
                "feature": "phase4_ig_followup_offer",
                "candidate_label": "Follow-up offer/friendly closure",
                "cluster": "followup_offer",
                "human_perceived_slop": "",
                "shaib_taxonomy_label": "",
            },
        ],
    )
    output = tmp_path / "summary.csv"
    summary = tmp_path / "summary.md"
    args = build_parser().parse_args(
        ["--input", str(package), "--output", str(output), "--summary-output", str(summary)]
    )

    rows = run_summarize_phase4_human_labels(args)

    assert rows[0]["labeled"] == 0
    assert rows[0]["blank"] == 2
    assert rows[0]["venn_region"] == "unlabeled"
    assert "unlabeled" in summary.read_text(encoding="utf-8")


def test_summarize_phase4_human_labels_promotes_human_perceived_feature(tmp_path):
    package = tmp_path / "labels.jsonl"
    _write_jsonl(
        package,
        [
            {
                "feature": "phase4_ig_process_framing",
                "candidate_label": "Explicit assistant process framing",
                "cluster": "assistant_process_framing",
                "human_perceived_slop": "yes",
                "shaib_taxonomy_label": "structure",
            },
            {
                "feature": "phase4_ig_process_framing",
                "candidate_label": "Explicit assistant process framing",
                "cluster": "assistant_process_framing",
                "human_perceived_slop": "context_dependent",
                "shaib_taxonomy_label": "coherence",
            },
            {
                "feature": "phase4_ig_process_framing",
                "candidate_label": "Explicit assistant process framing",
                "cluster": "assistant_process_framing",
                "human_perceived_slop": "no",
                "shaib_taxonomy_label": "none",
            },
        ],
    )
    output = tmp_path / "summary.csv"
    summary = tmp_path / "summary.md"
    args = build_parser().parse_args(
        ["--input", str(package), "--output", str(output), "--summary-output", str(summary)]
    )

    rows = run_summarize_phase4_human_labels(args)

    assert rows[0]["labeled"] == 3
    assert rows[0]["human_perceived_or_context_count"] == 2
    assert rows[0]["venn_region"] == "detector_and_human_perceived"
    with output.open(encoding="utf-8", newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert persisted[0]["feature"] == "phase4_ig_process_framing"


def test_summarize_phase4_human_labels_strict_rejects_unknown_label(tmp_path):
    package = tmp_path / "labels.jsonl"
    _write_jsonl(
        package,
        [
            {
                "feature": "phase4_ig_code_boilerplate",
                "candidate_label": "Code/task boilerplate",
                "cluster": "code_boilerplate",
                "human_perceived_slop": "maybe",
                "shaib_taxonomy_label": "none",
            }
        ],
    )
    args = build_parser().parse_args(["--input", str(package), "--strict"])

    with pytest.raises(ValueError, match="unknown human_perceived_slop"):
        run_summarize_phase4_human_labels(args)

