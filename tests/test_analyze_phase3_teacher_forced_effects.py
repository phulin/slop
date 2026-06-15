import csv

import pytest

from slop_sftdiv.cli.analyze_phase3_teacher_forced_effects import (
    analyze_teacher_forced_effects,
    build_parser,
    run_analyze_phase3_teacher_forced_effects,
)


FIELDNAMES = [
    "source",
    "record_id",
    "role",
    "feature",
    "opportunity_kind",
    "char_offset",
    "prefix_tokens",
    "reference_initiates",
    "matched_subtype",
    "prob_mass",
    "initiator_sequences",
]


def _row(record_id, feature, char_offset, prob_mass, reference_initiates=0):
    return {
        "source": "pkg",
        "record_id": record_id,
        "role": "target_response",
        "feature": feature,
        "opportunity_kind": "token_start",
        "char_offset": str(char_offset),
        "prefix_tokens": "4",
        "reference_initiates": str(reference_initiates),
        "matched_subtype": "",
        "prob_mass": str(prob_mass),
        "initiator_sequences": "2",
    }


def _write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_analyze_teacher_forced_effects_pairs_opportunities_and_reports_af(tmp_path):
    left_path = tmp_path / "left.csv"
    right_path = tmp_path / "right.csv"
    _write_rows(
        left_path,
        [
            _row("a", "slop_lexicon", 0, 0.1, 1),
            _row("b", "slop_lexicon", 0, 0.1, 0),
            _row("c", "slop_lexicon", 0, 0.4, 0),
            _row("a", "stock_openers", 0, 0.2, 0),
        ],
    )
    _write_rows(
        right_path,
        [
            _row("a", "slop_lexicon", 0, 0.2, 1),
            _row("b", "slop_lexicon", 0, 0.3, 0),
            _row("c", "slop_lexicon", 0, 0.1, 0),
            _row("unpaired", "slop_lexicon", 0, 0.9, 0),
            _row("a", "stock_openers", 0, 0.1, 0),
        ],
    )
    args = build_parser().parse_args(
        [
            "--opportunity-cache",
            f"sft={left_path}",
            "--opportunity-cache",
            f"dpo={right_path}",
            "--comparison",
            "sft=dpo",
            "--feature",
            "slop_lexicon",
            "--feature",
            "stock_openers",
            "--output",
            str(tmp_path / "out.csv"),
            "--summary-output",
            str(tmp_path / "out.md"),
        ]
    )

    rows = analyze_teacher_forced_effects(args)
    by_feature = {row["feature"]: row for row in rows}

    slop = by_feature["slop_lexicon"]
    assert slop["paired_opportunities"] == 3
    assert slop["positive_pairs"] == 2
    assert slop["negative_pairs"] == 1
    assert slop["reference_initiations"] == 1
    assert slop["reference_rate"] == pytest.approx(1 / 3)
    assert slop["left_af"] == pytest.approx(0.6)
    assert slop["right_af"] == pytest.approx(0.6)
    assert slop["af_delta"] == pytest.approx(0.0)
    assert slop["sign_test_p"] == pytest.approx(1.0)
    assert by_feature["stock_openers"]["direction"] == "left_gt_right"


def test_run_analyze_teacher_forced_effects_writes_outputs_and_logs(tmp_path, monkeypatch):
    base_path = tmp_path / "base.csv"
    dpo_path = tmp_path / "dpo.csv"
    output_path = tmp_path / "effects.csv"
    summary_path = tmp_path / "effects.md"
    _write_rows(base_path, [_row("a", "slop_lexicon", 0, 0.1, 1)])
    _write_rows(dpo_path, [_row("a", "slop_lexicon", 0, 0.2, 1)])
    payloads = []
    tables = {}

    class FakeRun:
        def log(self, payload):
            payloads.append(payload)

        def finish(self):
            pass

    monkeypatch.setattr(
        "slop_sftdiv.cli.analyze_phase3_teacher_forced_effects.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.analyze_phase3_teacher_forced_effects.log_summary_table",
        lambda _run, table_name, rows: tables.setdefault(table_name, rows),
    )
    args = build_parser().parse_args(
        [
            "--opportunity-cache",
            f"base={base_path}",
            "--opportunity-cache",
            f"dpo={dpo_path}",
            "--comparison",
            "base=dpo",
            "--feature",
            "slop_lexicon",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    rows = run_analyze_phase3_teacher_forced_effects(args)

    assert output_path.exists()
    assert summary_path.exists()
    assert len(rows) == 1
    assert payloads[-1]["teacher_forced_effects/rows"] == 1
    assert tables["phase3_teacher_forced_stage_effects"] == rows
