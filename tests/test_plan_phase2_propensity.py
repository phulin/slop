import csv
import json

from slop_sftdiv.cli.plan_phase2_propensity import build_parser, run_plan_phase2_propensity


def test_plan_phase2_propensity_writes_revision_aware_commands(tmp_path, monkeypatch):
    prompt_package = tmp_path / "smoltalk2_no_think.jsonl"
    output_dir = tmp_path / "propensity"
    plan_path = tmp_path / "plan.csv"
    summary_path = tmp_path / "plan.md"
    prompt_package.write_text(
        "\n".join(json.dumps({"prompt": f"Prompt {index}", "text": "Target"}) for index in range(3))
        + "\n",
        encoding="utf-8",
    )
    completed_opportunities = (
        output_dir
        / "smollm3_sft_smoke_slop-vs-neutral_promptpkg2_cached_branch4_sequence_opportunities.csv"
    )
    completed_summary = (
        output_dir
        / "smollm3_sft_smoke_slop-vs-neutral_promptpkg2_cached_branch4_sequence_summary.csv"
    )
    output_dir.mkdir()
    completed_opportunities.write_text("feature,prob_mass\nslop_lexicon,0.2\n", encoding="utf-8")
    completed_summary.write_text("feature,amplification_factor\nslop_lexicon,1.5\n", encoding="utf-8")
    logged_payloads = []
    logged_tables = {}

    class FakeRun:
        def log(self, payload):
            logged_payloads.append(payload)

        def finish(self):
            pass

    monkeypatch.setattr(
        "slop_sftdiv.cli.plan_phase2_propensity.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.plan_phase2_propensity.log_summary_table",
        lambda _run, table_name, rows: logged_tables.setdefault(table_name, rows),
    )

    args = build_parser().parse_args(
        [
            "--prompt-package",
            str(prompt_package),
            "--output-dir",
            str(output_dir),
            "--artifact-prefix",
            "smollm3",
            "--package-tag",
            "smoke",
            "--sample-size",
            "2",
            "--feature",
            "slop_lexicon",
            "--feature",
            "neutral_common_controls",
            "--normalization-feature",
            "neutral_common_controls",
            "--reference-subset",
            "no_think:mode=no_think",
            "--max-opportunities",
            "5",
            "--opportunities-per-sec-estimate",
            "10",
            "--stage",
            "sft=HuggingFaceTB/SmolLM3-3B-checkpoints@it-SFT",
            "--output",
            str(plan_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    rows = run_plan_phase2_propensity(args)

    assert len(rows) == 1
    row = rows[0]
    assert row["stage"] == "sft"
    assert row["model"] == "HuggingFaceTB/SmolLM3-3B-checkpoints"
    assert row["model_revision"] == "it-SFT"
    assert row["features"] == "slop_lexicon,neutral_common_controls"
    assert row["estimated_opportunities"] == 20
    assert row["estimated_a100_hours"] == 2 / 3600
    assert row["completed"] is True
    assert row["existing_opportunities"] == 1
    assert "--model-revision it-SFT" in row["command"]
    assert "--reference-subset no_think:mode=no_think" in row["command"]
    assert "--normalization-feature neutral_common_controls" in row["command"]
    assert "--sequence-cache" in row["command"]
    assert "--torch-compile" in row["command"]
    with plan_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["model_revision"] == "it-SFT"
    assert summary_path.exists()
    assert logged_payloads[-1]["propensity_plan/shards"] == 1
    assert logged_tables["phase2_propensity_plan"] == rows
