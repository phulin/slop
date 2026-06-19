from __future__ import annotations

import argparse
import csv
from pathlib import Path

from slop_sftdiv.cli.materialize_phase3_full_sglang_finalize_plan import (
    materialize_finalize_plan,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plan_row(
    tmp_path: Path,
    *,
    stage: str,
    temperature: float,
    expected: int = 2,
    existing: int = 2,
    summary_exists: bool = True,
) -> dict[str, object]:
    generations = tmp_path / f"{stage}_t{temperature:g}.jsonl"
    summary = tmp_path / f"{stage}_t{temperature:g}_summary.csv"
    generations.write_text("".join("{}\n" for _ in range(existing)), encoding="utf-8")
    if summary_exists:
        summary.write_text("feature,count\nslop_lexicon,1\n", encoding="utf-8")
    return {
        "stage": stage,
        "model": f"model-{stage}",
        "model_revision": "",
        "temperature": temperature,
        "top_p": 0.95,
        "sample_size": 1,
        "completions_per_prompt": expected,
        "max_new_tokens": 1024,
        "apply_chat_template": "",
        "chat_template_kwargs_json": "",
        "missing_chat_template": "",
        "expected_generations": expected,
        "expected_generated_tokens": expected * 1024,
        "estimated_seconds": 1,
        "estimated_a100_hours": 1,
        "generations_output": generations,
        "summary_output": summary,
        "completed": existing >= expected and summary_exists,
        "existing_generations": existing,
        "command": "echo generate",
    }


def _args(tmp_path: Path, plan: Path) -> argparse.Namespace:
    return argparse.Namespace(
        generation_plan=plan,
        output_script=tmp_path / "finalize.sh",
        summary_output=tmp_path / "finalize.md",
        artifact_prefix=str(tmp_path / "olmo3_full"),
        primary_temperature=1.0,
        bootstrap_samples=7,
        bootstrap_seed=1729,
        feature_rate=[str(tmp_path / "feature_rates.csv")],
        weighted_pretrain_baseline=[],
        propensity_grid=[str(tmp_path / "propensity.csv"), str(tmp_path / "stock.csv")],
        compounding_propensity_grid=tmp_path / "compounding_propensity.csv",
        denominator_support=[str(tmp_path / "denominators.csv")],
        preference_analysis=[str(tmp_path / "pairs.csv")],
        teacher_forced_stage_effects=[str(tmp_path / "tf_effects.csv")],
        free_run_stage_effects=[],
        right_spectrum=tmp_path / "smollm3.csv",
        wandb_mode="disabled",
    )


def _write_prerequisites(tmp_path: Path) -> None:
    for name in (
        "feature_rates.csv",
        "propensity.csv",
        "stock.csv",
        "compounding_propensity.csv",
        "denominators.csv",
        "pairs.csv",
        "tf_effects.csv",
        "smollm3.csv",
    ):
        (tmp_path / name).write_text("feature,value\nslop_lexicon,1\n", encoding="utf-8")


def test_materialize_finalize_plan_comments_incomplete_temperature(tmp_path: Path) -> None:
    rows = [
        _plan_row(tmp_path, stage=stage, temperature=1.0, existing=2)
        for stage in ("base", "sft", "dpo")
    ]
    rows.append(_plan_row(tmp_path, stage="final", temperature=1.0, existing=1))
    plan = tmp_path / "plan.csv"
    _write_csv(plan, rows)

    summary = materialize_finalize_plan(_args(tmp_path, plan))

    script = (tmp_path / "finalize.sh").read_text(encoding="utf-8")
    report = (tmp_path / "finalize.md").read_text(encoding="utf-8")
    assert summary["enabled_commands"] == 0
    assert "# incomplete: 7/8 rows" in script
    assert "# uv run slop-assemble-phase2-generation-grid" in script
    assert "incomplete primary temperature" in script
    assert "| 1 | no | 7 | 8 | 0 |" in report
    assert "| 1 | final | no | 1 | 2 | yes |" in report


def test_materialize_finalize_plan_enables_complete_primary_temperature(tmp_path: Path) -> None:
    _write_prerequisites(tmp_path)
    rows = [
        _plan_row(tmp_path, stage=stage, temperature=1.0)
        for stage in ("base", "sft", "dpo", "final")
    ]
    plan = tmp_path / "plan.csv"
    _write_csv(plan, rows)

    summary = materialize_finalize_plan(_args(tmp_path, plan))

    script = (tmp_path / "finalize.sh").read_text(encoding="utf-8")
    report = (tmp_path / "finalize.md").read_text(encoding="utf-8")
    assert summary["enabled_commands"] == 5
    assert "uv run slop-assemble-phase2-generation-grid" in script
    assert "uv run slop-analyze-phase2-compounding" in script
    assert "uv run slop-assemble-amplification-spectrum" in script
    assert "uv run slop-classify-amplification-spectrum" in script
    assert "uv run slop-compare-phase3-ladders" in script
    assert "--bootstrap-samples 7" in script
    assert "--propensity-grid " + str(tmp_path / "compounding_propensity.csv") in script
    assert "--propensity-grid " + str(tmp_path / "stock.csv") in script
    assert "| feature_rates | 1 |  |" in report
    assert "| 1 | yes | 8 | 8 | 2 |" in report
    assert "| 1 | base | yes | 2 | 2 | yes |" in report


def test_materialize_finalize_plan_disables_commands_with_missing_prerequisites(
    tmp_path: Path,
) -> None:
    rows = [
        _plan_row(tmp_path, stage=stage, temperature=1.0)
        for stage in ("base", "sft", "dpo", "final")
    ]
    plan = tmp_path / "plan.csv"
    _write_csv(plan, rows)

    summary = materialize_finalize_plan(_args(tmp_path, plan))

    script = (tmp_path / "finalize.sh").read_text(encoding="utf-8")
    report = (tmp_path / "finalize.md").read_text(encoding="utf-8")
    assert summary["enabled_commands"] == 1
    assert "uv run slop-assemble-phase2-generation-grid" in script
    assert "compounding prerequisites missing; command disabled" in script
    assert "amplification spectrum prerequisites missing; command disabled" in script
    assert "classification prerequisites missing; command disabled" in script
    assert "cross-ladder prerequisites missing; command disabled" in script
    assert "| feature_rates | 1 | `" + str(tmp_path / "feature_rates.csv") + "` |" in report


def test_materialize_finalize_plan_skips_existing_downstream_outputs(tmp_path: Path) -> None:
    rows = [
        _plan_row(tmp_path, stage=stage, temperature=1.0)
        for stage in ("base", "sft", "dpo", "final")
    ]
    plan = tmp_path / "plan.csv"
    _write_csv(plan, rows)

    output_paths = [
        "olmo3_full_generation_stage_grid_5000prompt_8comp_t1_1024.csv",
        "olmo3_full_generation_stage_grid_5000prompt_8comp_t1_1024_primary_feature_comparison.csv",
        "olmo3_full_generation_stage_grid_5000prompt_8comp_t1_1024_summary.md",
        "olmo3_full_compounding_analysis_5000prompt_8comp_t1_1024.csv",
        "olmo3_full_compounding_analysis_5000prompt_8comp_t1_1024_summary.md",
        "olmo3_full_compounding_analysis_5000prompt_8comp_t1_1024_realized_af.svg",
        "olmo3_full_compounding_analysis_5000prompt_8comp_t1_1024_cluster_counts.jsonl",
        "olmo3_full_amplification_spectrum_5000prompt_8comp_t1_1024.csv",
        "olmo3_full_amplification_spectrum_5000prompt_8comp_t1_1024_summary.md",
        "olmo3_full_feature_classification_5000prompt_8comp_t1_1024.csv",
        "olmo3_full_feature_classification_5000prompt_8comp_t1_1024_summary.md",
        "olmo3_full_vs_smollm3_fullgrid_t1_aligned.csv",
        "olmo3_full_vs_smollm3_fullgrid_t1_correlations.csv",
        "olmo3_full_vs_smollm3_fullgrid_t1_summary.md",
    ]
    for relative_path in output_paths:
        (tmp_path / relative_path).write_text("done\n", encoding="utf-8")

    summary = materialize_finalize_plan(_args(tmp_path, plan))

    script = (tmp_path / "finalize.sh").read_text(encoding="utf-8")
    report = (tmp_path / "finalize.md").read_text(encoding="utf-8")
    assert summary["enabled_commands"] == 0
    assert "uv run slop-assemble-phase2-generation-grid" not in script
    assert "uv run slop-analyze-phase2-compounding" not in script
    assert "uv run slop-assemble-amplification-spectrum" not in script
    assert "uv run slop-classify-amplification-spectrum" not in script
    assert "uv run slop-compare-phase3-ladders" not in script
    assert "generation grid outputs already exist; command skipped" in script
    assert "cross-ladder outputs already exist; command skipped" in script
    assert "| 1 | yes | 8 | 8 | 0 |" in report
