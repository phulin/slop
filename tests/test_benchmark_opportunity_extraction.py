import csv
import json

from slop_sftdiv.cli.benchmark_opportunity_extraction import build_parser, run_benchmark


def test_benchmark_opportunity_extraction_writes_full_and_offsets_rows(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "benchmark.csv"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "a",
                        "text": "Great question. It is not noise, it is signal. Delve deeper.",
                    }
                ),
                json.dumps(
                    {
                        "id": "b",
                        "chosen": "For example, robust tests catch issues.",
                        "rejected": "Plain tests run.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--feature",
            "contrastive_negation",
            "--feature",
            "slop_lexicon",
            "--sample-size",
            "2",
            "--repeats",
            "1",
            "--warmup",
            "0",
            "--output",
            str(output_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    rows = run_benchmark(args)

    assert output_path.exists()
    assert {row["mode"] for row in rows} == {"full", "offsets_only"}
    full_row = next(row for row in rows if row["mode"] == "full")
    offsets_row = next(row for row in rows if row["mode"] == "offsets_only")
    assert full_row["measurement_rows"] == 3
    assert full_row["opportunities"] == offsets_row["opportunities"]
    assert full_row["reference_initiations"] >= 2
    assert offsets_row["reference_initiations"] == 0
    with output_path.open(encoding="utf-8", newline="") as handle:
        written_rows = list(csv.DictReader(handle))
    assert len(written_rows) == 2
    assert written_rows[0]["features"] == "contrastive_negation,slop_lexicon"


def test_benchmark_opportunity_extraction_can_disable_offsets_only(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "benchmark.csv"
    input_path.write_text(
        json.dumps({"text": "Great question. Delve into tests."}) + "\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--feature",
            "stock_openers",
            "--no-include-offsets-only",
            "--repeats",
            "1",
            "--warmup",
            "0",
            "--output",
            str(output_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    rows = run_benchmark(args)

    assert [row["mode"] for row in rows] == ["full"]
    assert rows[0]["opportunities"] == 1
    assert rows[0]["reference_initiations"] == 1
