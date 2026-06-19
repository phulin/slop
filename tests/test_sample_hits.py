import csv
import json

from slop_sftdiv.cli.sample_hits import build_parser, run_sample_hits


def test_sample_hits_writes_label_template_and_logs_counts(tmp_path, monkeypatch):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "hits.csv"
    input_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "id": "a",
                    "text": "Great question. It is not just fast but reliable. Delve deeper.",
                },
                {"id": "b", "text": "- Robust tests\nI hope this helps."},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    metric_payloads = []
    table_rows = {}

    class FakeRun:
        def log(self, payload):
            metric_payloads.append(payload)

        def finish(self):
            pass

    monkeypatch.setattr("slop_sftdiv.cli.sample_hits.init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr(
        "slop_sftdiv.cli.sample_hits.log_summary_table",
        lambda _run, table_name, rows: table_rows.setdefault(table_name, rows),
    )

    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--sample-size",
            "2",
            "--hits-per-feature",
            "2",
            "--output",
            str(output_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    rows = run_sample_hits(args)

    assert output_path.exists()
    with output_path.open(encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert written == rows
    assert {row["feature"] for row in rows} >= {
        "stock_openers",
        "contrastive_negation",
        "slop_lexicon",
        "stock_closers",
    }
    assert "list_header_bold_lead_in" not in {row["feature"] for row in rows}
    assert all(row["label"] == "" for row in rows)
    assert all(row["context"] for row in rows)
    assert metric_payloads[-1]["precision/docs"] == 2
    assert metric_payloads[-1]["precision/sampled_hits"] == len(rows)
    assert table_rows["precision_feature_counts"]


def test_sample_hits_can_filter_features(tmp_path, monkeypatch):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "hits.csv"
    input_path.write_text(
        json.dumps(
            {
                "id": "a",
                "text": "Great question. It is not just fast but reliable. Delve deeper.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    monkeypatch.setattr("slop_sftdiv.cli.sample_hits.init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr(
        "slop_sftdiv.cli.sample_hits.log_summary_table",
        lambda _run, _table_name, _rows: None,
    )

    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--sample-size",
            "1",
            "--hits-per-feature",
            "5",
            "--feature",
            "contrastive_negation",
            "--feature",
            "slop_lexicon",
            "--output",
            str(output_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    rows = run_sample_hits(args)

    assert rows
    assert {row["feature"] for row in rows} <= {"contrastive_negation", "slop_lexicon"}
