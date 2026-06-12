import csv

import pytest

from slop_sftdiv.cli.analyze_pairs import analyze_rows, build_parser, run_analyze_pairs


FIELDNAMES = [
    "source",
    "subset",
    "pair_id",
    "preference_type",
    "chosen_model",
    "rejected_model",
    "feature",
    "chosen_count",
    "rejected_count",
    "chosen_tokens",
    "rejected_tokens",
    "chosen_rate_per_1k_tokens",
    "rejected_rate_per_1k_tokens",
    "chosen_minus_rejected",
]


def _row(pair_id, feature, delta, chosen_tokens=100, rejected_tokens=80, subset="train"):
    chosen_rate = max(delta, 0)
    rejected_rate = max(-delta, 0)
    return {
        "source": "prefs",
        "subset": subset,
        "pair_id": pair_id,
        "preference_type": "",
        "chosen_model": "",
        "rejected_model": "",
        "feature": feature,
        "chosen_count": "0",
        "rejected_count": "0",
        "chosen_tokens": str(chosen_tokens),
        "rejected_tokens": str(rejected_tokens),
        "chosen_rate_per_1k_tokens": str(chosen_rate),
        "rejected_rate_per_1k_tokens": str(rejected_rate),
        "chosen_minus_rejected": str(delta),
    }


def _write_pair_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_analyze_rows_computes_sign_test_and_bh_q_values():
    rows = [
        *[_row(f"p{i}", "slop_lexicon", 1.0) for i in range(5)],
        _row("p6", "slop_lexicon", -1.0),
        _row("p1", "stock_openers", 0.0),
        _row("p2", "stock_openers", 0.0),
    ]

    report = analyze_rows(rows, alpha=0.2)
    by_feature = {row["feature"]: row for row in report}

    assert by_feature["slop_lexicon"]["positive_pairs"] == 5
    assert by_feature["slop_lexicon"]["subset"] == "train"
    assert by_feature["slop_lexicon"]["negative_pairs"] == 1
    assert by_feature["slop_lexicon"]["sign_test_p"] == pytest.approx(0.21875)
    assert by_feature["slop_lexicon"]["direction"] == "chosen_gt_rejected"
    assert by_feature["slop_lexicon"]["logit_feature_coef"] > 0
    assert by_feature["slop_lexicon"]["logit_feature_odds_ratio"] > 1
    assert by_feature["stock_openers"]["zero_pairs"] == 2
    assert by_feature["stock_openers"]["sign_test_p"] == 1.0
    assert by_feature["stock_openers"]["direction"] == "tie"


def test_run_analyze_pairs_writes_report_and_logs_wandb(tmp_path, monkeypatch):
    input_path = tmp_path / "pairs.csv"
    output_path = tmp_path / "analysis.csv"
    _write_pair_rows(
        input_path,
        [
            _row("p1", "slop_lexicon", 1.0),
            _row("p2", "slop_lexicon", 2.0),
            _row("p3", "stock_openers", -1.0),
        ],
    )
    metric_payloads = []
    table_rows = {}

    class FakeRun:
        def log(self, payload):
            metric_payloads.append(payload)

        def finish(self):
            pass

    monkeypatch.setattr("slop_sftdiv.cli.analyze_pairs.init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr(
        "slop_sftdiv.cli.analyze_pairs.log_summary_table",
        lambda _run, table_name, rows: table_rows.setdefault(table_name, rows),
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

    rows = run_analyze_pairs(args)

    assert output_path.exists()
    assert {row["feature"] for row in rows} == {"slop_lexicon", "stock_openers"}
    assert metric_payloads[-1]["preference/features"] == 2
    assert metric_payloads[-1]["preference/pair_delta_rows"] == 3
    assert table_rows["preference_feature_analysis"] == rows
    assert "logit_feature_coef" in rows[0]


def test_analyze_rows_keeps_source_subset_groups_separate():
    rows = [
        _row("p1", "slop_lexicon", 1.0, subset="a"),
        _row("p2", "slop_lexicon", -1.0, subset="b"),
    ]

    report = analyze_rows(rows, alpha=0.05)

    assert len(report) == 2
    assert {row["subset"] for row in report} == {"a", "b"}


def test_analyze_rows_keeps_preference_metadata_groups_separate():
    rows = [
        {
            **_row("p1", "slop_lexicon", 1.0),
            "preference_type": "delta_learning",
            "chosen_model": "strong",
            "rejected_model": "weak",
        },
        {
            **_row("p2", "slop_lexicon", -1.0),
            "preference_type": "llm_judged",
            "chosen_model": "judge_chosen",
            "rejected_model": "judge_rejected",
        },
    ]

    report = analyze_rows(rows, alpha=0.05)

    assert len(report) == 2
    assert {row["preference_type"] for row in report} == {"delta_learning", "llm_judged"}


def test_analyze_rows_logit_controls_for_length_direction():
    rows = [
        _row("p1", "slop_lexicon", 2.0, chosen_tokens=120, rejected_tokens=80),
        _row("p2", "slop_lexicon", 2.0, chosen_tokens=80, rejected_tokens=120),
        _row("p3", "slop_lexicon", 2.0, chosen_tokens=90, rejected_tokens=110),
        _row("p4", "slop_lexicon", -1.0, chosen_tokens=110, rejected_tokens=90),
    ]

    [report] = analyze_rows(rows, alpha=0.05)

    assert report["logit_iterations"] > 0
    assert report["logit_feature_coef"] > 0
    assert abs(report["logit_length_coef"]) < report["logit_feature_coef"]


def test_analyze_rows_sign_test_handles_large_samples():
    rows = [
        *[_row(f"p{i}", "slop_lexicon", 1.0) for i in range(6000)],
        *[_row(f"n{i}", "slop_lexicon", -1.0) for i in range(4000)],
    ]

    [report] = analyze_rows(rows, alpha=0.05)

    assert 0.0 <= report["sign_test_p"] <= 1.0
    assert report["sign_test_p"] < 1e-80
