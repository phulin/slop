import csv

import pytest

from slop_sftdiv.cli.materialize_phase2_generation_continuation_plan import (
    build_parser,
    materialize_phase2_generation_continuation_plan,
)


def _write_plan(path):
    rows = [
        {
            "stage": "base",
            "model": "model-base",
            "model_revision": "",
            "temperature": "0.0",
            "top_p": "0.95",
            "sample_size": "2",
            "completions_per_prompt": "2",
            "max_new_tokens": "8",
            "apply_chat_template": "False",
            "chat_template_kwargs_json": "",
            "missing_chat_template": "error",
            "expected_generations": "4",
            "expected_generated_tokens": "32",
            "estimated_seconds": "10.0",
            "estimated_a100_hours": "1.5",
            "generations_output": str(path.parent / "base.jsonl"),
            "summary_output": str(path.parent / "base_summary.csv"),
            "completed": "False",
            "existing_generations": "0",
            "command": "uv run slop-free-running-emission --generation-batch-size 64 --flag x",
        },
        {
            "stage": "sft",
            "model": "model-sft",
            "model_revision": "",
            "temperature": "1.0",
            "top_p": "0.95",
            "sample_size": "2",
            "completions_per_prompt": "2",
            "max_new_tokens": "8",
            "apply_chat_template": "False",
            "chat_template_kwargs_json": "",
            "missing_chat_template": "error",
            "expected_generations": "4",
            "expected_generated_tokens": "32",
            "estimated_seconds": "10.0",
            "estimated_a100_hours": "2.5",
            "generations_output": str(path.parent / "sft.jsonl"),
            "summary_output": str(path.parent / "sft_summary.csv"),
            "completed": "False",
            "existing_generations": "0",
            "command": "uv run slop-free-running-emission --generation-batch-size 64 --resume",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_materialize_phase2_generation_continuation_plan_rewrites_commands_and_counts(
    tmp_path,
):
    input_plan = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    summary = tmp_path / "summary.md"
    _write_plan(input_plan)
    (tmp_path / "base.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--input-plan",
            str(input_plan),
            "--output",
            str(output),
            "--summary-output",
            str(summary),
            "--generation-batch-size",
            "32",
            "--resume",
        ]
    )

    rows = materialize_phase2_generation_continuation_plan(args)

    assert rows[0]["existing_generations"] == 2
    assert rows[0]["completed"] is False
    assert "--generation-batch-size 32" in rows[0]["command"]
    assert rows[0]["command"].endswith("--resume")
    assert rows[1]["command"].count("--resume") == 1
    with output.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["existing_generations"] == "2"
    text = summary.read_text(encoding="utf-8")
    assert "uses generation batch size 32" in text
    assert "| base | 0 | 4 | 2 | 1.50 | no |" in text


def test_materialize_phase2_generation_continuation_plan_validates_batch_size(tmp_path):
    input_plan = tmp_path / "input.csv"
    _write_plan(input_plan)
    args = build_parser().parse_args(
        [
            "--input-plan",
            str(input_plan),
            "--output",
            str(tmp_path / "output.csv"),
            "--summary-output",
            str(tmp_path / "summary.md"),
            "--generation-batch-size",
            "0",
        ]
    )

    with pytest.raises(ValueError, match="positive"):
        materialize_phase2_generation_continuation_plan(args)
