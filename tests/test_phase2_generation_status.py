import csv
import json
from datetime import datetime, timedelta, timezone

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


def test_phase2_generation_status_reports_tqdm_log_progress(tmp_path, capsys):
    generations_path = tmp_path / "generations.jsonl"
    summary_path = tmp_path / "summary.csv"
    log_path = tmp_path / "launch.log"
    selection_path = tmp_path / "selection.json"
    output_path = tmp_path / "status.csv"
    log_path.write_text(
        "free-run:pkg: 16prompt [05:44, 21.56s/prompt]\r"
        "free-run:pkg: 32prompt [11:32, 21.66s/prompt]\n",
        encoding="utf-8",
    )
    selection_path.write_text(
        json.dumps(
            {
                "stage": "dpo",
                "temperature": 1.0,
                "pid": "",
                "command": "uv run slop-free-running-emission --completions-per-prompt 8",
                "expected_generations": 8192,
                "generations_output": str(generations_path),
                "summary_output": str(summary_path),
                "log_output": str(log_path),
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

    assert rows[0]["latest_log_prompts"] == 32
    assert rows[0]["latest_log_generations_estimate"] == 256
    assert rows[0]["latest_log_elapsed_seconds"] == 692
    assert rows[0]["latest_log_avg_seconds_per_prompt"] == 21.625
    assert rows[0]["latest_log_seconds_per_prompt"] == 21.66
    assert rows[0]["eta_hms"] == "5:58:07"
    assert rows[0]["eta_avg_hms"] == "5:57:32"
    output = capsys.readouterr().out
    assert "log_prompts=32" in output
    assert "eta=5:58:07" in output
    assert "eta_avg=5:57:32" in output
    with output_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["latest_log_prompts"] == "32"
    assert csv_rows[0]["latest_log_generations_estimate"] == "256"
    assert csv_rows[0]["latest_log_elapsed_seconds"] == "692"
    assert csv_rows[0]["latest_log_avg_seconds_per_prompt"] == "21.625"
    assert csv_rows[0]["latest_log_seconds_per_prompt"] == "21.66"
    assert csv_rows[0]["eta_hms"] == "5:58:07"
    assert csv_rows[0]["eta_avg_hms"] == "5:57:32"


def test_phase2_generation_status_estimates_eta_from_existing_generations(tmp_path, capsys):
    generations_path = tmp_path / "generations.jsonl"
    summary_path = tmp_path / "summary.csv"
    selection_path = tmp_path / "selection.json"
    output_path = tmp_path / "status.csv"
    generations_path.write_text("{}\n" * 100, encoding="utf-8")
    started_at = datetime.now(timezone.utc) - timedelta(hours=1)
    selection_path.write_text(
        json.dumps(
            {
                "stage": "base",
                "temperature": 0.0,
                "pid": "",
                "started_at": started_at.isoformat(),
                "expected_generations": 200,
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

    assert rows[0]["existing_generations"] == 100
    assert 0.02 < rows[0]["existing_generations_per_second"] < 0.03
    assert 3500 < rows[0]["existing_eta_seconds"] < 3700
    output = capsys.readouterr().out
    assert "eta_existing=" in output
    with output_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["existing_generations_per_second"]
    assert csv_rows[0]["existing_eta_hms"]


def test_phase2_generation_status_ignores_resumed_rows_for_throughput_until_new_rows(
    tmp_path,
):
    generations_path = tmp_path / "generations.jsonl"
    summary_path = tmp_path / "summary.csv"
    selection_path = tmp_path / "selection.json"
    generations_path.write_text("{}\n{}\n", encoding="utf-8")
    started_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    selection_path.write_text(
        json.dumps(
            {
                "stage": "base",
                "temperature": 0.0,
                "pid": "",
                "started_at": started_at.isoformat(),
                "expected_generations": 4,
                "generations_output": str(generations_path),
                "summary_output": str(summary_path),
                "log_output": str(tmp_path / "launch.log"),
                "resume_initial_generations": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(["--selection", str(selection_path)])

    rows = run_phase2_generation_status(args)

    assert rows[0]["existing_generations"] == 2
    assert rows[0]["existing_generations_per_second"] == ""
    assert rows[0]["existing_eta_hms"] == ""
