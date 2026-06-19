import csv

from slop_sftdiv.cli.summarize_phase3_generation_plans import build_parser, run


def _write_plan(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "temperature",
                "completed",
                "expected_generations",
                "existing_generations",
                "estimated_a100_hours",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_phase3_generation_plans_writes_csv_and_markdown(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    output = tmp_path / "summary.csv"
    summary = tmp_path / "summary.md"
    _write_plan(
        left,
        [
            {
                "stage": "base",
                "temperature": "0.0",
                "completed": "True",
                "expected_generations": "4",
                "existing_generations": "4",
                "estimated_a100_hours": "1.5",
            },
            {
                "stage": "sft",
                "temperature": "0.0",
                "completed": "False",
                "expected_generations": "4",
                "existing_generations": "1",
                "estimated_a100_hours": "2.5",
            },
        ],
    )
    _write_plan(
        right,
        [
            {
                "stage": "base",
                "temperature": "1.0",
                "completed": "no",
                "expected_generations": "8",
                "existing_generations": "0",
                "estimated_a100_hours": "3.0",
            }
        ],
    )

    args = build_parser().parse_args(
        [
            "--plan",
            f"left={left}",
            "--plan",
            f"right={right}",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ]
    )

    rows = run(args)

    assert rows[0]["plan_name"] == "left"
    assert rows[0]["total_shards"] == 2
    assert rows[0]["completed_shards"] == 1
    assert rows[0]["in_flight_shards"] == 0
    assert rows[0]["missing_shards"] == 1
    assert rows[0]["expected_generations"] == 8
    assert rows[0]["existing_generations"] == 5
    assert rows[0]["status_existing_generations"] == 0
    assert rows[0]["missing_estimated_a100_hours"] == 2.5
    assert rows[1]["plan_name"] == "right"
    assert rows[1]["missing_estimated_a100_hours"] == 3.0
    with output.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["completed_shards"] == "1"
    text = summary.read_text(encoding="utf-8")
    assert "`left`" in text
    assert "| `right` | 0 | 0 | 1 | 8 | 0 |  | 3.00 |" in text


def test_summarize_phase3_generation_plans_merges_live_selection_status(tmp_path):
    plan = tmp_path / "plan.csv"
    status = tmp_path / "status.csv"
    output = tmp_path / "summary.csv"
    summary = tmp_path / "summary.md"
    _write_plan(
        plan,
        [
            {
                "stage": "base",
                "temperature": "0.0",
                "completed": "False",
                "expected_generations": "4096",
                "existing_generations": "0",
                "estimated_a100_hours": "3.25",
            },
            {
                "stage": "sft",
                "temperature": "0.0",
                "completed": "False",
                "expected_generations": "4096",
                "existing_generations": "0",
                "estimated_a100_hours": "3.25",
            },
        ],
    )
    with status.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "temperature",
                "process_alive",
                "completed",
                "existing_generations",
                "eta_hms",
                "eta_avg_hms",
                "existing_eta_hms",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "stage": "base",
                "temperature": "0.0",
                "process_alive": "True",
                "completed": "False",
                "existing_generations": "512",
                "eta_hms": "0:30:00",
                "eta_avg_hms": "0:35:00",
                "existing_eta_hms": "0:45:00",
            }
        )
    args = build_parser().parse_args(
        [
            "--plan",
            f"smol={plan}",
            "--selection-status",
            f"smol={status}",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ]
    )

    rows = run(args)

    assert rows[0]["in_flight_shards"] == 1
    assert rows[0]["missing_shards"] == 1
    assert rows[0]["existing_generations"] == 512
    assert rows[0]["status_existing_generations"] == 512
    assert rows[0]["active_eta_hms"] == "0:30:00"
    assert rows[0]["missing_estimated_a100_hours"] == 3.25
    with output.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["in_flight_shards"] == "1"
    assert csv_rows[0]["active_eta_hms"] == "0:30:00"
    assert "| `smol` | 0 | 1 | 1 | 8192 | 512 | 0:30:00 | 3.25 |" in summary.read_text(
        encoding="utf-8"
    )
