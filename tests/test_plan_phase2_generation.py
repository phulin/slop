import csv
import json

from slop_sftdiv.cli.plan_phase2_generation import build_parser, run_plan_phase2_generation


def test_plan_phase2_generation_writes_commands_and_completion_status(tmp_path, monkeypatch):
    prompt_package = tmp_path / "olmo3_dolci_sft_phase2_prompt_package_128.jsonl"
    output_dir = tmp_path / "generations"
    plan_path = tmp_path / "plan.csv"
    summary_path = tmp_path / "plan.md"
    prompt_package.write_text(
        "\n".join(json.dumps({"prompt": f"Prompt {index}", "text": "Target"}) for index in range(4))
        + "\n",
        encoding="utf-8",
    )
    completed_jsonl = (
        output_dir
        / "olmo3_sft_promptpkg128_free_run_2prompt_2comp_t0_batched4.jsonl"
    )
    completed_summary = (
        output_dir
        / "olmo3_sft_promptpkg128_free_run_2prompt_2comp_t0_batched4_summary.csv"
    )
    output_dir.mkdir()
    completed_jsonl.write_text("{}\n{}\n{}\n{}\n", encoding="utf-8")
    completed_summary.write_text("feature,count\nslop_lexicon,0\n", encoding="utf-8")
    logged_payloads = []
    logged_tables = {}

    class FakeRun:
        def log(self, payload):
            logged_payloads.append(payload)

        def finish(self):
            pass

    monkeypatch.setattr(
        "slop_sftdiv.cli.plan_phase2_generation.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.plan_phase2_generation.log_summary_table",
        lambda _run, table_name, rows: logged_tables.setdefault(table_name, rows),
    )

    args = build_parser().parse_args(
        [
            "--prompt-package",
            str(prompt_package),
            "--output-dir",
            str(output_dir),
            "--sample-size",
            "2",
            "--completions-per-prompt",
            "2",
            "--temperature",
            "0.0",
            "--temperature",
            "1.0",
            "--generation-batch-size",
            "4",
            "--apply-chat-template",
            "--chat-template-kwargs-json",
            '{"enable_thinking": false}',
            "--max-new-tokens",
            "8",
            "--tokens-per-sec-estimate",
            "4",
            "--stage",
            "sft=fake-sft@it-SFT",
            "--output",
            str(plan_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    rows = run_plan_phase2_generation(args)

    assert len(rows) == 2
    by_temp = {float(row["temperature"]): row for row in rows}
    assert by_temp[0.0]["completed"] is True
    assert by_temp[1.0]["completed"] is False
    assert by_temp[1.0]["expected_generations"] == 4
    assert by_temp[1.0]["estimated_a100_hours"] == 8 / 3600
    assert by_temp[1.0]["model"] == "fake-sft"
    assert by_temp[1.0]["model_revision"] == "it-SFT"
    assert "uv run slop-free-running-emission" in by_temp[1.0]["command"]
    assert "--model-revision it-SFT" in by_temp[1.0]["command"]
    assert "--apply-chat-template" in by_temp[1.0]["command"]
    assert "--chat-template-kwargs-json '{\"enable_thinking\": false}'" in by_temp[1.0]["command"]
    assert "--torch-compile" in by_temp[1.0]["command"]
    with plan_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["stage"] == "sft"
    assert csv_rows[0]["model_revision"] == "it-SFT"
    assert csv_rows[0]["apply_chat_template"] == "True"
    assert summary_path.exists()
    assert logged_payloads[-1]["generation_plan/shards"] == 2
    assert logged_tables["phase2_generation_plan"] == rows
