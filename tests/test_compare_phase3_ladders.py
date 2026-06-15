import csv

import pytest

from slop_sftdiv.cli.compare_phase3_ladders import build_parser, run


def _write_csv(path, rows):
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _row(feature, stage, af, *, normalized="", free_run=0.0):
    return {
        "feature": feature,
        "stage": stage,
        "stage_label": stage,
        "teacher_forced_af": str(af),
        "teacher_forced_normalized_af": str(normalized),
        "free_run_per_1k_tokens": str(free_run),
        "sft_target_per_1k_tokens": "1.0",
        "dpo_chosen_per_1k_tokens": "2.0",
        "dpo_rejected_per_1k_tokens": "1.5",
        "coverage_note": "covered",
    }


def test_compare_phase3_ladders_aligns_shared_rows_and_correlates(tmp_path, monkeypatch):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    aligned_output = tmp_path / "aligned.csv"
    correlation_output = tmp_path / "corr.csv"
    summary_output = tmp_path / "summary.md"
    _write_csv(
        left,
        [
            _row("a", "base", 1.0, normalized=1.2, free_run=0.1),
            _row("b", "base", 2.0, free_run=0.2),
            _row("c", "base", 3.0, free_run=0.3),
            _row("a", "dpo", 1.0, free_run=0.4),
            _row("left_only", "base", 10.0),
        ],
    )
    _write_csv(
        right,
        [
            _row("a", "base", 10.0, normalized=12.0, free_run=1.1),
            _row("b", "base", 20.0, free_run=1.2),
            _row("c", "base", 30.0, free_run=1.3),
            _row("a", "dpo", 2.0, free_run=1.4),
            _row("right_only", "base", 40.0),
        ],
    )

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(
        "slop_sftdiv.cli.compare_phase3_ladders.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    logged_tables = {}
    monkeypatch.setattr(
        "slop_sftdiv.cli.compare_phase3_ladders.log_summary_table",
        lambda _run, name, rows: logged_tables.setdefault(name, rows),
    )

    args = build_parser().parse_args(
        [
            "--left-spectrum",
            str(left),
            "--right-spectrum",
            str(right),
            "--left-label",
            "olmo",
            "--right-label",
            "smol",
            "--aligned-output",
            str(aligned_output),
            "--correlation-output",
            str(correlation_output),
            "--summary-output",
            str(summary_output),
            "--wandb-mode",
            "disabled",
        ]
    )
    aligned, correlations = run(args)

    by_key = {(row["feature"], row["stage"]): row for row in aligned}
    assert set(by_key) == {("a", "base"), ("b", "base"), ("c", "base"), ("a", "dpo")}
    assert by_key[("a", "base")]["left_af"] == 1.2
    assert by_key[("a", "base")]["right_af"] == 12.0
    assert by_key[("a", "base")]["left_af_source"] == "normalized"
    assert by_key[("b", "base")]["left_af_source"] == "raw"
    assert by_key[("a", "base")]["free_run_delta_right_minus_left"] == pytest.approx(1.0)
    overall = next(row for row in correlations if row["scope"] == "all")
    base = next(row for row in correlations if row["scope"] == "stage" and row["stage"] == "base")
    dpo = next(row for row in correlations if row["scope"] == "stage" and row["stage"] == "dpo")
    assert overall["shared_af_values"] == 4
    assert overall["spearman_af"] == pytest.approx(1.0)
    assert base["shared_af_values"] == 3
    assert base["spearman_af"] == pytest.approx(1.0)
    assert dpo["shared_af_values"] == 1
    assert dpo["spearman_af"] is None
    assert aligned_output.exists()
    assert correlation_output.exists()
    assert "Phase 3 Cross-Ladder Amplification Comparison" in summary_output.read_text()
    assert "phase3_cross_ladder_aligned" in logged_tables
    assert "phase3_cross_ladder_correlations" in logged_tables
