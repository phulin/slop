import json

import pytest

from slop_sftdiv.cli.run_phase2_post_shard_analysis import (
    DEFAULT_FEATURES,
    build_parser,
    run_phase2_post_shard_analysis,
)


def _write_selection(tmp_path, *, expected=2, rows=2, summary=True):
    generations_path = tmp_path / "generations.jsonl"
    summary_path = tmp_path / "summary.csv"
    log_path = tmp_path / "launch.log"
    selection_path = tmp_path / "selection.json"
    generations_path.write_text("{}\n" * rows, encoding="utf-8")
    if summary:
        summary_path.write_text("feature,count\nslop_lexicon,1\n", encoding="utf-8")
    log_path.write_text("free-run:pkg: 16prompt [05:44, 21.56s/prompt]\n", encoding="utf-8")
    selection_path.write_text(
        json.dumps(
            {
                "stage": "dpo",
                "temperature": 1.0,
                "pid": "",
                "command": "uv run slop-free-running-emission --completions-per-prompt 8",
                "expected_generations": expected,
                "generations_output": str(generations_path),
                "summary_output": str(summary_path),
                "log_output": str(log_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return selection_path, generations_path, summary_path


def test_post_shard_analysis_refuses_incomplete_without_wait(tmp_path):
    selection_path, _, _ = _write_selection(tmp_path, expected=2, rows=1, summary=True)
    args = build_parser().parse_args(["--selection", str(selection_path)])

    with pytest.raises(ValueError, match="not complete"):
        run_phase2_post_shard_analysis(args)


def test_post_shard_analysis_dry_run_reports_planned_outputs(tmp_path, capsys):
    selection_path, generations_path, summary_path = _write_selection(tmp_path)
    args = build_parser().parse_args(
        [
            "--selection",
            str(selection_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    payload = run_phase2_post_shard_analysis(args)

    assert payload["completed"] is True
    assert payload["generations_output"] == str(generations_path)
    assert payload["summary_output"] == str(summary_path)
    assert payload["latest_log_prompts"] == 16
    assert payload["latest_log_generations_estimate"] == 128
    assert payload["latest_log_elapsed_seconds"] == 344
    assert payload["latest_log_seconds_per_prompt"] == 21.56
    assert payload["eta_hms"] == "0:00:00"
    assert "rerun with --execute" in capsys.readouterr().out


def test_post_shard_analysis_execute_calls_existing_analyzers(tmp_path, monkeypatch):
    selection_path, generations_path, summary_path = _write_selection(tmp_path)
    scale_calls = []
    compounding_calls = []

    def fake_scale(args):
        scale_calls.append(args)
        return {"grid_rows": 12, "comparison_rows": 2}

    def fake_compounding(args):
        compounding_calls.append(args)
        return [{"feature": "slop_lexicon"}]

    monkeypatch.setattr(
        "slop_sftdiv.cli.run_phase2_post_shard_analysis.run_assemble_phase2_generation_grid",
        fake_scale,
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.run_phase2_post_shard_analysis.run_analyze_phase2_compounding",
        fake_compounding,
    )
    args = build_parser().parse_args(
        [
            "--selection",
            str(selection_path),
            "--execute",
            "--wandb-mode",
            "disabled",
        ]
    )

    payload = run_phase2_post_shard_analysis(args)

    assert payload["scale_rows"] == 12
    assert payload["compounding_rows"] == 1
    assert scale_calls[0].generation_summary == [
        (
            "a_dpo512=artifacts/phase2/generations/"
            "olmo3_dpo_promptpkg5000_free_run_512prompt_8comp_t1_batched1024_summary.csv"
        ),
        f"b_dpo1024={summary_path}",
    ]
    assert compounding_calls[0].generation_cache == [f"dpo={generations_path}"]
    assert compounding_calls[0].feature == DEFAULT_FEATURES
    assert compounding_calls[0].wandb_mode == "disabled"
