import json

import pytest

from slop_sftdiv.cli.census import build_parser, run_census


def test_run_census_with_disabled_wandb_on_local_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "summary.csv"
    input_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"id": "a", "text": "Certainly, delve into this robust plan."},
                {"id": "b", "text": "It is not noisy but focused."},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--sample-size",
            "2",
            "--output",
            str(output_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    frame = run_census(args)

    assert output_path.exists()
    assert not frame.empty
    assert set(frame["feature"]) >= {
        "contrastive_negation",
        "slop_lexicon",
        "stock_openers_closers",
    }
    assert frame.loc[frame["feature"] == "slop_lexicon", "count"].sum() >= 2
    assert frame["docs"].max() == 2


def test_run_census_writes_pair_output_for_preference_records(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    input_path = tmp_path / "pairs.jsonl"
    output_path = tmp_path / "summary.csv"
    pair_output_path = tmp_path / "pairs.csv"
    input_path.write_text(
        json.dumps(
            {
                "pair_id": "pair-1",
                "chosen": "Certainly, delve into this robust plan.",
                "rejected": "Plain answer.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--sample-size",
            "1",
            "--output",
            str(output_path),
            "--pair-output",
            str(pair_output_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    frame = run_census(args)

    pair_rows = pytest.importorskip("pandas").read_csv(pair_output_path)
    assert output_path.exists()
    assert not frame.empty
    assert list(pair_rows.columns) == [
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
    slop_row = pair_rows.loc[pair_rows["feature"] == "slop_lexicon"].iloc[0]
    assert slop_row["pair_id"] == "pair-1"
    assert slop_row["chosen_count"] == 2
    assert slop_row["rejected_count"] == 0
    assert slop_row["chosen_tokens"] > 0
    assert slop_row["rejected_tokens"] > 0
    assert slop_row["chosen_minus_rejected"] == pytest.approx(
        slop_row["chosen_rate_per_1k_tokens"] - slop_row["rejected_rate_per_1k_tokens"]
    )


def test_run_census_does_not_log_raw_preference_text_to_wandb(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    input_path = tmp_path / "pairs.jsonl"
    output_path = tmp_path / "summary.csv"
    logged_tables = {}
    input_path.write_text(
        json.dumps(
            {
                "pair_id": "pair-1",
                "prompt": "Sensitive prompt",
                "chosen": "Sensitive chosen",
                "rejected": "Sensitive rejected",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def capture_table(_run, table_name, rows):
        logged_tables[table_name] = rows

    monkeypatch.setattr("slop_sftdiv.cli.census.log_summary_table", capture_table)
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--sample-size",
            "1",
            "--output",
            str(output_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    run_census(args)

    sampled_records = logged_tables["sampled_records"]
    assert sampled_records
    payload = json.dumps(sampled_records)
    assert "Sensitive prompt" not in payload
    assert "Sensitive chosen" not in payload
    assert "Sensitive rejected" not in payload


def test_run_census_groups_by_promoted_source_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    input_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "summary.csv"
    input_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"id": "a", "text": "Delve into tests.", "source": "web_cc"},
                {"id": "b", "text": "Great question.", "source": "forums_qa"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--sample-size",
            "2",
            "--output",
            str(output_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    frame = run_census(args)

    assert set(frame["subset"]) == {"web_cc", "forums_qa"}


def test_run_census_logs_throughput_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "summary.csv"
    input_path.write_text(
        json.dumps({"id": "a", "text": "Great question. Delve into tests."}) + "\n",
        encoding="utf-8",
    )
    table_rows = {}
    metric_payloads = []
    init_kwargs = {}

    class FakeRun:
        def log(self, payload):
            metric_payloads.append(payload)

        def finish(self):
            pass

    def capture_init(**kwargs):
        init_kwargs.update(kwargs)
        return FakeRun()

    monkeypatch.setattr("slop_sftdiv.cli.census.init_wandb", capture_init)
    monkeypatch.setattr(
        "slop_sftdiv.cli.census.log_summary_table",
        lambda _run, table_name, rows: table_rows.setdefault(table_name, rows),
    )
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--sample-size",
            "1",
            "--output",
            str(output_path),
            "--wandb-mode",
            "disabled",
            "--wandb-group",
            "corpus-sampling",
            "--wandb-job-type",
            "throughput",
            "--wandb-tag",
            "tiny",
            "--wandb-tag",
            "dry_run",
        ]
    )

    run_census(args)

    metrics = metric_payloads[-1]
    assert metrics["corpus/docs"] == 1
    assert metrics["corpus/measurement_rows"] == 1
    assert metrics["corpus/tokens"] > 0
    assert metrics["corpus/wall_seconds"] > 0
    assert metrics["corpus/tokens_per_sec"] > 0
    assert metrics["corpus/peak_rss_mb"] > 0
    assert init_kwargs["group"] == "corpus-sampling"
    assert init_kwargs["job_type"] == "throughput"
    assert "tiny" in init_kwargs["tags"]
    assert "dry_run" in init_kwargs["tags"]
    assert table_rows["source_metrics"][0]["docs"] == 1
    assert table_rows["source_metrics"][0]["tokens"] == metrics["corpus/tokens"]
