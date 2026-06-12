import csv

import pytest

from slop_sftdiv.cli.score_labels import build_parser, run_score_labels


def _write_labels(path, rows):
    fieldnames = ["label", "feature", "sample_key"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_score_labels_writes_precision_report_and_logs_wandb(tmp_path, monkeypatch):
    input_path = tmp_path / "labels.csv"
    output_path = tmp_path / "report.csv"
    _write_labels(
        input_path,
        [
            {"label": "exact", "feature": "slop_lexicon", "sample_key": "a"},
            {"label": "exact", "feature": "slop_lexicon", "sample_key": "b"},
            {"label": "false_positive", "feature": "slop_lexicon", "sample_key": "c"},
            {"label": "ambiguous", "feature": "rule_of_three_approx", "sample_key": "d"},
        ],
    )
    metric_payloads = []
    table_rows = {}

    class FakeRun:
        def log(self, payload):
            metric_payloads.append(payload)

        def finish(self):
            pass

    monkeypatch.setattr("slop_sftdiv.cli.score_labels.init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr(
        "slop_sftdiv.cli.score_labels.log_summary_table",
        lambda _run, table_name, rows: table_rows.setdefault(table_name, rows),
    )
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--target-precision",
            "0.8",
            "--min-labeled",
            "3",
            "--wandb-mode",
            "disabled",
        ]
    )

    scored = run_score_labels(args)

    assert output_path.exists()
    by_feature = {row["feature"]: row for row in scored}
    assert by_feature["slop_lexicon"]["precision"] == pytest.approx(2 / 3)
    assert by_feature["slop_lexicon"]["status"] == "demote"
    assert by_feature["rule_of_three_approx"]["ambiguous_rate"] == 1.0
    assert by_feature["rule_of_three_approx"]["sample_size_pass"] is False
    assert metric_payloads[-1]["matcher/features_scored"] == 2
    assert metric_payloads[-1]["matcher/features_demoted"] == 2
    assert table_rows["precision_report"] == scored


def test_score_labels_rejects_missing_or_invalid_labels(tmp_path):
    input_path = tmp_path / "labels.csv"
    output_path = tmp_path / "report.csv"
    _write_labels(
        input_path,
        [
            {"label": "exact", "feature": "slop_lexicon", "sample_key": "a"},
            {"label": "", "feature": "slop_lexicon", "sample_key": "b"},
            {"label": "maybe", "feature": "slop_lexicon", "sample_key": "c"},
        ],
    )
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    with pytest.raises(ValueError, match="invalid or missing labels"):
        run_score_labels(args)


def test_score_labels_can_ignore_incomplete_rows(tmp_path):
    input_path = tmp_path / "labels.csv"
    output_path = tmp_path / "report.csv"
    _write_labels(
        input_path,
        [
            {"label": "exact", "feature": "slop_lexicon", "sample_key": "a"},
            {"label": "", "feature": "slop_lexicon", "sample_key": "b"},
        ],
    )
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--allow-incomplete",
            "--wandb-mode",
            "disabled",
        ]
    )

    scored = run_score_labels(args)

    assert scored[0]["labeled"] == 1
    assert scored[0]["exact"] == 1
