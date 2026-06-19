import csv

import pytest

from slop_sftdiv.cli.materialize_sglang_generation_plan import (
    build_parser,
    materialize_sglang_generation_plan,
)


def _write_plan(path, *, model_revision="", apply_chat_template="False"):
    rows = [
        {
            "stage": "base",
            "model": "model-base",
            "model_revision": model_revision,
            "temperature": "0.7",
            "top_p": "0.95",
            "sample_size": "2",
            "completions_per_prompt": "2",
            "max_new_tokens": "8",
            "apply_chat_template": apply_chat_template,
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
            "command": (
                "uv run slop-free-running-emission --model model-base "
                "--input prompts.jsonl --sample-size 2 --seed 1729 "
                "--temperature 0.7 --top-p 0.95 --max-new-tokens 8 "
                "--completions-per-prompt 2 --generation-batch-size 64 "
                "--max-prompt-tokens 1024 --dtype bfloat16 "
                "--generations-output base.jsonl --summary-output base_summary.csv "
                "--wandb-mode online --missing-chat-template error --no-torch-compile"
            ),
        }
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_materialize_sglang_generation_plan_rewrites_commands_and_counts(tmp_path):
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
            "--sglang-python",
            "/tmp/sg/bin/python",
            "--sglang-script",
            "scripts/benchmark_sglang_generation.py",
            "--context-length",
            "4096",
            "--prompt-batch-size",
            "512",
            "--sampling-strategy",
            "hash_reservoir",
            "--feature-text-mode",
            "final_answer",
            "--ignore-eos",
            "--wandb-mode",
            "disabled",
        ]
    )

    rows = materialize_sglang_generation_plan(args)

    assert rows[0]["existing_generations"] == 2
    assert rows[0]["completed"] is False
    assert rows[0]["command"].startswith("env LD_LIBRARY_PATH=")
    assert " /tmp/sg/bin/python scripts/benchmark_sglang_generation.py" in rows[0]["command"]
    assert "--model model-base" in rows[0]["command"]
    assert "--input prompts.jsonl" in rows[0]["command"]
    assert "--sampling-strategy hash_reservoir" in rows[0]["command"]
    assert "--context-length 4096" in rows[0]["command"]
    assert "--prompt-batch-size 512" in rows[0]["command"]
    assert "--attention-backend triton" in rows[0]["command"]
    assert "--feature-text-mode final_answer" in rows[0]["command"]
    assert "--disable-cuda-graph" in rows[0]["command"]
    assert "--ignore-eos" in rows[0]["command"]
    assert "--no-trust-remote-code" in rows[0]["command"]
    with output.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["existing_generations"] == "2"
    summary_text = summary.read_text(encoding="utf-8")
    assert "SGLang Python: `/tmp/sg/bin/python`" in summary_text
    assert "Prompt batch size: `512`" in summary_text
    assert "Feature text mode: `final_answer`" in summary_text


def test_materialize_sglang_generation_plan_rejects_revision_rows(tmp_path):
    input_plan = tmp_path / "input.csv"
    _write_plan(input_plan, model_revision="it-SFT")
    args = build_parser().parse_args(
        [
            "--input-plan",
            str(input_plan),
            "--output",
            str(tmp_path / "output.csv"),
            "--summary-output",
            str(tmp_path / "summary.md"),
        ]
    )

    with pytest.raises(ValueError, match="model_revision"):
        materialize_sglang_generation_plan(args)


def test_materialize_sglang_generation_plan_rejects_chat_template_rows(tmp_path):
    input_plan = tmp_path / "input.csv"
    _write_plan(input_plan, apply_chat_template="True")
    args = build_parser().parse_args(
        [
            "--input-plan",
            str(input_plan),
            "--output",
            str(tmp_path / "output.csv"),
            "--summary-output",
            str(tmp_path / "summary.md"),
        ]
    )

    with pytest.raises(ValueError, match="plain prompt"):
        materialize_sglang_generation_plan(args)
