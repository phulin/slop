import csv
import json

from slop_sftdiv.cli.phase2_generation_status import build_parser, run_phase2_generation_status


def test_phase2_generation_status_reports_completion(tmp_path, capsys):
    generations_path = tmp_path / "generations.jsonl"
    summary_path = tmp_path / "summary.csv"
    selection_path = tmp_path / "selection.json"
    output_path = tmp_path / "status.csv"
    generations_path.write_text("{}\n{}\n", encoding="utf-8")
    summary_path.write_text("feature,count\nslop_lexicon,0\n", encoding="utf-8")
    selection_path.write_text(
        json.dumps(
            {
                "stage": "dpo",
                "temperature": 1.0,
                "pid": "",
                "expected_generations": 2,
                "generations_output": str(generations_path),
                "summary_output": str(summary_path),
                "log_output": str(tmp_path / "launch.log"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--selection",
            str(selection_path),
            "--output",
            str(output_path),
        ]
    )

    rows = run_phase2_generation_status(args)

    assert rows[0]["completed"] is True
    assert rows[0]["existing_generations"] == 2
    assert "completed=True" in capsys.readouterr().out
    with output_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["completed"] == "True"
