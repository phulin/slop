import csv
import json
import subprocess

import pytest

from slop_sftdiv.cli.continue_phase2_generation_plan import (
    build_parser,
    continue_phase2_generation_plan,
)


def _write_plan(path, rows=None):
    rows = rows or [
        {
            "stage": "base",
            "temperature": "0.0",
            "estimated_a100_hours": "3.0",
            "expected_generations": "4",
            "generations_output": str(path.parent / "base.jsonl"),
            "summary_output": str(path.parent / "base_summary.csv"),
            "completed": "False",
            "command": "uv run echo base",
        },
        {
            "stage": "sft",
            "temperature": "1.0",
            "estimated_a100_hours": "9.0",
            "expected_generations": "4",
            "generations_output": str(path.parent / "sft.jsonl"),
            "summary_output": str(path.parent / "sft_summary.csv"),
            "completed": "False",
            "command": "uv run echo sft",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _args(*items):
    return build_parser().parse_args(list(items))


def test_continue_phase2_generation_plan_reports_active_selection(tmp_path, monkeypatch):
    plan = tmp_path / "plan.csv"
    selection_dir = tmp_path / "selections"
    selection_dir.mkdir()
    _write_plan(plan)
    selection = selection_dir / "run_base_t0.json"
    selection.write_text(
        json.dumps(
            {
                "stage": "base",
                "temperature": 0.0,
                "pid": 123,
                "generations_output": str(tmp_path / "active.jsonl"),
                "summary_output": str(tmp_path / "active_summary.csv"),
                "expected_generations": 4,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "active.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    monkeypatch.setattr("slop_sftdiv.cli.continue_phase2_generation_plan._process_alive", lambda pid: True)

    decision = continue_phase2_generation_plan(
        _args("--plan", str(plan), "--selection-dir", str(selection_dir), "--selection-prefix", "run")
    )

    assert decision["action"] == "active_shard_running"
    assert decision["active"][0]["existing_generations"] == 1


def test_continue_phase2_generation_plan_ignores_decision_json(tmp_path, monkeypatch):
    plan = tmp_path / "plan.csv"
    selection_dir = tmp_path / "selections"
    selection_dir.mkdir()
    _write_plan(plan)
    (selection_dir / "run_continuation_decision.json").write_text(
        json.dumps({"action": "active_shard_running"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("slop_sftdiv.cli.continue_phase2_generation_plan._process_alive", lambda pid: False)

    decision = continue_phase2_generation_plan(
        _args("--plan", str(plan), "--selection-dir", str(selection_dir), "--selection-prefix", "run")
    )

    assert decision["action"] == "would_launch"


def test_continue_phase2_generation_plan_refuses_partial_missing_shard(tmp_path, monkeypatch):
    plan = tmp_path / "plan.csv"
    selection_dir = tmp_path / "selections"
    _write_plan(plan)
    (tmp_path / "base.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    monkeypatch.setattr("slop_sftdiv.cli.continue_phase2_generation_plan._process_alive", lambda pid: False)

    decision = continue_phase2_generation_plan(
        _args("--plan", str(plan), "--selection-dir", str(selection_dir), "--selection-prefix", "run")
    )

    assert decision["action"] == "blocked_partial_shard"
    assert decision["existing_generations"] == 1


def test_continue_phase2_generation_plan_resumes_partial_when_allowed(tmp_path, monkeypatch):
    plan = tmp_path / "plan.csv"
    selection_dir = tmp_path / "selections"
    _write_plan(plan)
    (tmp_path / "base.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    monkeypatch.setattr("slop_sftdiv.cli.continue_phase2_generation_plan._process_alive", lambda pid: False)

    decision = continue_phase2_generation_plan(
        _args(
            "--plan",
            str(plan),
            "--selection-dir",
            str(selection_dir),
            "--selection-prefix",
            "run",
            "--allow-partial-retry",
        )
    )

    assert decision["action"] == "would_launch"
    selection = json.loads((selection_dir / "run_base_t0p0.json").read_text(encoding="utf-8"))
    assert selection["command"] == "uv run echo base --resume"


def test_continue_phase2_generation_plan_overrides_generation_batch_size(tmp_path, monkeypatch):
    plan = tmp_path / "plan.csv"
    selection_dir = tmp_path / "selections"
    rows = [
        {
            "stage": "base",
            "temperature": "0.0",
            "estimated_a100_hours": "3.0",
            "expected_generations": "4",
            "generations_output": str(tmp_path / "base.jsonl"),
            "summary_output": str(tmp_path / "base_summary.csv"),
            "completed": "False",
            "command": "uv run slop-free-running-emission --generation-batch-size 64 --other value",
        }
    ]
    _write_plan(plan, rows)
    monkeypatch.setattr("slop_sftdiv.cli.continue_phase2_generation_plan._process_alive", lambda pid: False)

    decision = continue_phase2_generation_plan(
        _args(
            "--plan",
            str(plan),
            "--selection-dir",
            str(selection_dir),
            "--selection-prefix",
            "run",
            "--generation-batch-size-override",
            "32",
        )
    )

    assert decision["action"] == "would_launch"
    selection = json.loads((selection_dir / "run_base_t0p0.json").read_text(encoding="utf-8"))
    assert "--generation-batch-size 32" in selection["command"]
    assert "--generation-batch-size 64" not in selection["command"]


def test_continue_phase2_generation_plan_dry_run_writes_selection(tmp_path, monkeypatch):
    plan = tmp_path / "plan.csv"
    selection_dir = tmp_path / "selections"
    _write_plan(plan)
    monkeypatch.setattr("slop_sftdiv.cli.continue_phase2_generation_plan._process_alive", lambda pid: False)

    decision = continue_phase2_generation_plan(
        _args("--plan", str(plan), "--selection-dir", str(selection_dir), "--selection-prefix", "run")
    )

    assert decision["action"] == "would_launch"
    selection = json.loads((selection_dir / "run_base_t0p0.json").read_text(encoding="utf-8"))
    assert selection["executed"] is False
    assert selection["command"] == "uv run echo base"


def test_continue_phase2_generation_plan_prioritizes_temperature(tmp_path, monkeypatch):
    plan = tmp_path / "plan.csv"
    selection_dir = tmp_path / "selections"
    rows = [
        {
            "stage": "base",
            "temperature": "0.0",
            "estimated_a100_hours": "3.0",
            "expected_generations": "4",
            "generations_output": str(tmp_path / "base_t0.jsonl"),
            "summary_output": str(tmp_path / "base_t0_summary.csv"),
            "completed": "False",
            "command": "uv run echo base-t0",
        },
        {
            "stage": "base",
            "temperature": "1.0",
            "estimated_a100_hours": "3.0",
            "expected_generations": "4",
            "generations_output": str(tmp_path / "base_t1.jsonl"),
            "summary_output": str(tmp_path / "base_t1_summary.csv"),
            "completed": "False",
            "command": "uv run echo base-t1",
        },
    ]
    _write_plan(plan, rows)
    monkeypatch.setattr("slop_sftdiv.cli.continue_phase2_generation_plan._process_alive", lambda pid: False)

    decision = continue_phase2_generation_plan(
        _args(
            "--plan",
            str(plan),
            "--selection-dir",
            str(selection_dir),
            "--selection-prefix",
            "run",
            "--priority-temperature",
            "1.0",
        )
    )

    assert decision["action"] == "would_launch"
    assert decision["temperature"] == 1.0
    selection = json.loads((selection_dir / "run_base_t1p0.json").read_text(encoding="utf-8"))
    assert selection["command"] == "uv run echo base-t1"


def test_continue_phase2_generation_plan_ignores_stale_completed_flag(
    tmp_path,
    monkeypatch,
):
    plan = tmp_path / "plan.csv"
    selection_dir = tmp_path / "selections"
    rows = [
        {
            "stage": "base",
            "temperature": "1.0",
            "estimated_a100_hours": "3.0",
            "expected_generations": "4",
            "generations_output": str(tmp_path / "base_t1.jsonl"),
            "summary_output": str(tmp_path / "base_t1_summary.csv"),
            "completed": "True",
            "command": "uv run echo base-t1",
        }
    ]
    _write_plan(plan, rows)
    monkeypatch.setattr("slop_sftdiv.cli.continue_phase2_generation_plan._process_alive", lambda pid: False)

    decision = continue_phase2_generation_plan(
        _args("--plan", str(plan), "--selection-dir", str(selection_dir), "--selection-prefix", "run")
    )

    assert decision["action"] == "would_launch"
    assert decision["stage"] == "base"
    assert decision["temperature"] == 1.0


def test_continue_phase2_generation_plan_ignores_completed_active_selection_for_priority(
    tmp_path,
    monkeypatch,
):
    plan = tmp_path / "plan.csv"
    selection_dir = tmp_path / "selections"
    selection_dir.mkdir()
    rows = [
        {
            "stage": "base",
            "temperature": "0.0",
            "estimated_a100_hours": "3.0",
            "expected_generations": "2",
            "generations_output": str(tmp_path / "base_t0.jsonl"),
            "summary_output": str(tmp_path / "base_t0_summary.csv"),
            "completed": "False",
            "command": "uv run echo base-t0",
        },
        {
            "stage": "base",
            "temperature": "0.7",
            "estimated_a100_hours": "3.0",
            "expected_generations": "2",
            "generations_output": str(tmp_path / "base_t07.jsonl"),
            "summary_output": str(tmp_path / "base_t07_summary.csv"),
            "completed": "False",
            "command": "uv run echo base-t07",
        },
        {
            "stage": "base",
            "temperature": "1.0",
            "estimated_a100_hours": "3.0",
            "expected_generations": "2",
            "generations_output": str(tmp_path / "base_t1.jsonl"),
            "summary_output": str(tmp_path / "base_t1_summary.csv"),
            "completed": "False",
            "command": "uv run echo base-t1",
        },
    ]
    _write_plan(plan, rows)
    (tmp_path / "base_t0.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (tmp_path / "base_t0_summary.csv").write_text("feature,count\nslop,1\n", encoding="utf-8")
    (selection_dir / "run_base_t0p0.json").write_text(
        json.dumps(
            {
                "stage": "base",
                "temperature": 0.0,
                "pid": 123,
                "generations_output": str(tmp_path / "base_t0.jsonl"),
                "summary_output": str(tmp_path / "base_t0_summary.csv"),
                "expected_generations": 2,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("slop_sftdiv.cli.continue_phase2_generation_plan._process_alive", lambda pid: True)

    decision = continue_phase2_generation_plan(
        _args(
            "--plan",
            str(plan),
            "--selection-dir",
            str(selection_dir),
            "--selection-prefix",
            "run",
            "--priority-temperature",
            "1.0",
        )
    )

    assert decision["action"] == "would_launch"
    assert decision["temperature"] == 1.0
    selection = json.loads((selection_dir / "run_base_t1p0.json").read_text(encoding="utf-8"))
    assert selection["command"] == "uv run echo base-t1"


def test_continue_phase2_generation_plan_execute_detaches_next_shard(tmp_path, monkeypatch):
    plan = tmp_path / "plan.csv"
    selection_dir = tmp_path / "selections"
    _write_plan(plan)
    popen_calls = []

    class FakeProcess:
        pid = 456

    def fake_popen(command, *, stdout, stderr, start_new_session):
        popen_calls.append((command, stdout.name, stderr, start_new_session))
        return FakeProcess()

    monkeypatch.setattr("slop_sftdiv.cli.continue_phase2_generation_plan._process_alive", lambda pid: False)
    monkeypatch.setattr("slop_sftdiv.cli.continue_phase2_generation_plan.subprocess.Popen", fake_popen)

    decision = continue_phase2_generation_plan(
        _args(
            "--plan",
            str(plan),
            "--selection-dir",
            str(selection_dir),
            "--selection-prefix",
            "run",
            "--execute",
        )
    )

    assert decision["action"] == "launched"
    assert decision["pid"] == 456
    assert popen_calls == [
        (["uv", "run", "echo", "base"], str(selection_dir / "run_base_t0p0.log"), subprocess.STDOUT, True)
    ]


def test_continue_phase2_generation_plan_refuses_expensive_next_shard(tmp_path, monkeypatch):
    plan = tmp_path / "plan.csv"
    selection_dir = tmp_path / "selections"
    rows = [
        {
            "stage": "base",
            "temperature": "0.0",
            "estimated_a100_hours": "9.0",
            "expected_generations": "4",
            "generations_output": str(tmp_path / "base.jsonl"),
            "summary_output": str(tmp_path / "base_summary.csv"),
            "completed": "False",
            "command": "uv run echo base",
        }
    ]
    _write_plan(plan, rows)
    monkeypatch.setattr("slop_sftdiv.cli.continue_phase2_generation_plan._process_alive", lambda pid: False)

    with pytest.raises(ValueError, match="above limit"):
        continue_phase2_generation_plan(
            _args(
                "--plan",
                str(plan),
                "--selection-dir",
                str(selection_dir),
                "--max-estimated-a100-hours",
                "8",
            )
        )
