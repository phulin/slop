import csv
import json
import subprocess

import pytest

from slop_sftdiv.cli.launch_phase2_generation_shard import (
    build_parser,
    run_launch_phase2_generation_shard,
)


def _write_plan(path):
    rows = [
        {
            "stage": "base",
            "model": "model-base",
            "temperature": "0.0",
            "estimated_a100_hours": "31.0",
            "expected_generations": "40000",
            "generations_output": "base.jsonl",
            "summary_output": "base_summary.csv",
            "completed": "False",
            "command": "uv run echo base",
        },
        {
            "stage": "dpo",
            "model": "model-dpo",
            "temperature": "1.0",
            "estimated_a100_hours": "3.0",
            "expected_generations": "4096",
            "generations_output": "dpo.jsonl",
            "summary_output": "dpo_summary.csv",
            "completed": "False",
            "command": "uv run echo dpo",
        },
        {
            "stage": "sft",
            "model": "model-sft",
            "temperature": "1.0",
            "estimated_a100_hours": "3.0",
            "expected_generations": "4096",
            "generations_output": "sft.jsonl",
            "summary_output": "sft_summary.csv",
            "completed": "True",
            "command": "uv run echo sft",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_launch_phase2_generation_shard_selects_without_execute(tmp_path, capsys):
    plan_path = tmp_path / "plan.csv"
    selection_path = tmp_path / "selection.json"
    _write_plan(plan_path)
    args = build_parser().parse_args(
        [
            "--plan",
            str(plan_path),
            "--stage",
            "dpo",
            "--temperature",
            "1.0",
            "--selection-output",
            str(selection_path),
        ]
    )

    payload = run_launch_phase2_generation_shard(args)

    assert payload["stage"] == "dpo"
    assert payload["executed"] is False
    assert "uv run echo dpo" in capsys.readouterr().out
    assert json.loads(selection_path.read_text(encoding="utf-8"))["stage"] == "dpo"


def test_launch_phase2_generation_shard_refuses_expensive_shard(tmp_path):
    plan_path = tmp_path / "plan.csv"
    _write_plan(plan_path)
    args = build_parser().parse_args(
        [
            "--plan",
            str(plan_path),
            "--stage",
            "base",
            "--max-estimated-a100-hours",
            "8",
        ]
    )

    with pytest.raises(ValueError, match="above limit"):
        run_launch_phase2_generation_shard(args)


def test_launch_phase2_generation_shard_execute_is_explicit_and_mocked(tmp_path, monkeypatch):
    plan_path = tmp_path / "plan.csv"
    _write_plan(plan_path)
    calls = []

    def fake_run(command, *, check):
        calls.append((command, check))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("slop_sftdiv.cli.launch_phase2_generation_shard.subprocess.run", fake_run)
    args = build_parser().parse_args(
        [
            "--plan",
            str(plan_path),
            "--stage",
            "dpo",
            "--execute",
        ]
    )

    payload = run_launch_phase2_generation_shard(args)

    assert payload["executed"] is True
    assert calls == [(["uv", "run", "echo", "dpo"], True)]


def test_launch_phase2_generation_shard_detaches_and_writes_pid(tmp_path, monkeypatch):
    plan_path = tmp_path / "plan.csv"
    selection_path = tmp_path / "selection.json"
    log_path = tmp_path / "selection.log"
    _write_plan(plan_path)
    popen_calls = []

    class FakeProcess:
        pid = 12345

    def fake_popen(command, *, stdout, stderr, start_new_session):
        popen_calls.append((command, stdout.name, stderr, start_new_session))
        return FakeProcess()

    monkeypatch.setattr("slop_sftdiv.cli.launch_phase2_generation_shard.subprocess.Popen", fake_popen)
    args = build_parser().parse_args(
        [
            "--plan",
            str(plan_path),
            "--stage",
            "dpo",
            "--execute",
            "--detach",
            "--selection-output",
            str(selection_path),
            "--log-output",
            str(log_path),
        ]
    )

    payload = run_launch_phase2_generation_shard(args)

    assert payload["detached"] is True
    assert payload["pid"] == 12345
    assert json.loads(selection_path.read_text(encoding="utf-8"))["pid"] == 12345
    assert popen_calls == [(["uv", "run", "echo", "dpo"], str(log_path), subprocess.STDOUT, True)]
