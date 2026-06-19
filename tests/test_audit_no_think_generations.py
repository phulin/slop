import csv
import json

from slop_sftdiv.cli.audit_no_think_generations import build_parser, run


def test_audit_no_think_generations_allows_plain_think_words(tmp_path):
    generation_path = tmp_path / "generations.jsonl"
    output_path = tmp_path / "audit.csv"
    summary_path = tmp_path / "audit.md"
    generation_path.write_text(
        json.dumps({"record_id": "ok", "generation": "I think this is a normal answer."})
        + "\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--generation-cache",
            str(generation_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ]
    )

    rows = run(args)

    assert rows[0].passed is True
    assert rows[0].records_with_marker == 0
    assert rows[0].records_without_generation == 0
    with output_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["passed"] == "True"
    assert "Non-empty generations" in summary_path.read_text(encoding="utf-8")


def test_audit_no_think_generations_flags_markup_and_examples(tmp_path):
    generation_path = tmp_path / "generations.jsonl"
    output_path = tmp_path / "audit.csv"
    summary_path = tmp_path / "audit.md"
    generation_path.write_text(
        json.dumps({"record_id": "bad-1", "generation": "<think>hidden</think> answer"})
        + "\n"
        + json.dumps({"record_id": "bad-2", "generation": "```thinking\nhidden\n```"})
        + "\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--generation-cache",
            str(generation_path),
            "--max-examples",
            "1",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ]
    )

    rows = run(args)

    assert rows[0].passed is False
    assert rows[0].records == 2
    assert rows[0].records_with_generation == 2
    assert rows[0].records_without_generation == 0
    assert rows[0].records_with_marker == 2
    assert rows[0].marker_hits == 3
    assert json.loads(rows[0].example_record_ids_json) == ["bad-1"]
    with output_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["records_with_marker"] == "2"


def test_audit_no_think_generations_counts_empty_generations(tmp_path):
    generation_path = tmp_path / "generations.jsonl"
    output_path = tmp_path / "audit.csv"
    summary_path = tmp_path / "audit.md"
    generation_path.write_text(
        json.dumps({"record_id": "empty", "generation": ""})
        + "\n"
        + json.dumps({"record_id": "missing"})
        + "\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--generation-cache",
            str(generation_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ]
    )

    rows = run(args)

    assert rows[0].passed is True
    assert rows[0].records == 2
    assert rows[0].records_with_generation == 0
    assert rows[0].records_without_generation == 2
    with output_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["records_without_generation"] == "2"
