from __future__ import annotations

import csv
import json

from slop_sftdiv.cli.measure_phase2_denominators import (
    build_parser,
    run_measure_phase2_denominators,
)


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_measure_phase2_denominators_reports_feature_support(tmp_path, monkeypatch):
    input_path = tmp_path / "phase2.jsonl"
    output_path = tmp_path / "support.csv"
    summary_path = tmp_path / "support_summary.json"
    _write_jsonl(
        input_path,
        [
            {
                "phase2_prompt_id": "a",
                "text": "It's not about speed; it's about robust tests. I hope this helps.",
            },
            {
                "phase2_prompt_id": "b",
                "text": "Great question. Use a fixture.",
            },
        ],
    )
    logged_payloads = []
    logged_tables = {}

    class FakeRun:
        def log(self, payload):
            logged_payloads.append(payload)

        def finish(self):
            pass

    monkeypatch.setattr(
        "slop_sftdiv.cli.measure_phase2_denominators.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.measure_phase2_denominators.log_summary_table",
        lambda _run, table_name, rows: logged_tables.setdefault(table_name, rows),
    )
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--sample-size",
            "2",
            "--feature",
            "contrastive_negation",
            "--feature",
            "slop_lexicon",
            "--feature",
            "stock_openers_closers",
            "--feature",
            "rule_of_three_approx",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    summary = run_measure_phase2_denominators(args)

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["feature"]: row for row in csv.DictReader(handle)}
    assert int(rows["contrastive_negation"]["reference_initiations"]) == 1
    assert int(rows["slop_lexicon"]["reference_initiations"]) == 1
    assert int(rows["stock_openers_closers"]["reference_initiations"]) == 2
    assert int(rows["stock_openers_closers"]["documents_with_reference"]) == 2
    assert "rule_of_three_approx" not in rows
    assert summary["requested_features"] == 4
    assert summary["features"] == 3
    assert summary["omitted_features"] == 1
    assert summary["omitted_feature_ids"] == ("rule_of_three_approx",)
    assert summary["measurement_rows"] == 2
    assert summary["feature_measurement_rows"] == 6
    assert summary["features_with_references"] == 3
    assert json.loads(summary_path.read_text(encoding="utf-8"))["reference_initiations"] == 4
    assert logged_payloads[-1]["phase2_denominators/features_with_references"] == 3
    logged_table = json.dumps(logged_tables["phase2_denominator_support"])
    assert "Great question" not in logged_table
    assert "robust tests" not in logged_table
