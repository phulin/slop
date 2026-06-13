from __future__ import annotations

import csv
from pathlib import Path

import pytest

from slop_sftdiv.cli.assemble_phase2_generation_grid import (
    build_parser,
    run_assemble_phase2_generation_grid,
)


HEADER = [
    "source",
    "temperature",
    "top_p",
    "feature",
    "count",
    "repeated_count",
    "generations",
    "generations_with_feature",
    "repeat_generations",
    "generated_tokens",
    "per_1k_tokens",
    "per_generation",
    "repeat_per_generation",
    "repeat_share_after_first",
    "share_generations_with_feature",
    "share_repeat_generations",
]


def _write_summary(path: Path, *, slop_count: int, temperature: float = 0.7) -> None:
    rows = [
        {
            "source": "prompt_package",
            "temperature": temperature,
            "top_p": 0.95,
            "feature": "slop_lexicon",
            "count": slop_count,
            "repeated_count": max(0, slop_count - 1),
            "generations": 4,
            "generations_with_feature": 1 if slop_count else 0,
            "repeat_generations": 1 if slop_count > 1 else 0,
            "generated_tokens": 100,
            "per_1k_tokens": 10.0 * slop_count,
            "per_generation": slop_count / 4,
            "repeat_per_generation": max(0, slop_count - 1) / 4,
            "repeat_share_after_first": (slop_count - 1) / slop_count if slop_count else 0.0,
            "share_generations_with_feature": 0.25 if slop_count else 0.0,
            "share_repeat_generations": 0.25 if slop_count > 1 else 0.0,
        },
        {
            "source": "prompt_package",
            "temperature": temperature,
            "top_p": 0.95,
            "feature": "stock_openers",
            "count": 1,
            "repeated_count": 0,
            "generations": 4,
            "generations_with_feature": 1,
            "repeat_generations": 0,
            "generated_tokens": 100,
            "per_1k_tokens": 10.0,
            "per_generation": 0.25,
            "repeat_per_generation": 0.0,
            "repeat_share_after_first": 0.0,
            "share_generations_with_feature": 0.25,
            "share_repeat_generations": 0.0,
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def test_assemble_phase2_generation_grid_writes_comparison(tmp_path, monkeypatch):
    base_path = tmp_path / "base.csv"
    sft_path = tmp_path / "sft.csv"
    _write_summary(base_path, slop_count=1)
    _write_summary(sft_path, slop_count=3)
    output_path = tmp_path / "grid.csv"
    comparison_path = tmp_path / "comparison.csv"
    summary_path = tmp_path / "summary.md"

    logged_payloads = []
    logged_tables = {}

    class FakeRun:
        def log(self, payload):
            logged_payloads.append(payload)

        def finish(self):
            pass

    def fake_log_summary_table(_run, table_name, rows):
        logged_tables[table_name] = rows

    monkeypatch.setattr(
        "slop_sftdiv.cli.assemble_phase2_generation_grid.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.assemble_phase2_generation_grid.log_summary_table",
        fake_log_summary_table,
    )

    args = build_parser().parse_args(
        [
            "--generation-summary",
            f"base={base_path}",
            "--generation-summary",
            f"sft={sft_path}",
            "--checkpoint-label",
            "base=OLMo base",
            "--output",
            str(output_path),
            "--comparison-output",
            str(comparison_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )
    summary = run_assemble_phase2_generation_grid(args)

    assert summary["grid_rows"] == 4
    assert summary["comparison_rows"] == 2
    assert summary["max_per_1k_stage"] == "sft"
    with comparison_path.open("r", encoding="utf-8", newline="") as handle:
        comparison_rows = list(csv.DictReader(handle))
    assert [row["stage"] for row in comparison_rows] == ["base", "sft"]
    assert float(comparison_rows[0]["delta_per_1k_vs_baseline_stage"]) == pytest.approx(0.0)
    assert float(comparison_rows[1]["delta_per_1k_vs_baseline_stage"]) == pytest.approx(20.0)
    assert "Phase 2 Generation Stage Grid" in summary_path.read_text(encoding="utf-8")
    assert logged_payloads[-1]["phase2_generation_grid/max_per_1k_stage"] == "sft"
    assert len(logged_tables["phase2_generation_grid"]) == 4
    assert len(logged_tables["phase2_generation_primary_feature_comparison"]) == 2


def test_assemble_phase2_generation_grid_rejects_duplicate_stage(tmp_path):
    path = tmp_path / "summary.csv"
    _write_summary(path, slop_count=1)
    args = build_parser().parse_args(
        [
            "--generation-summary",
            f"sft={path}",
            "--generation-summary",
            f"sft={path}",
        ]
    )

    with pytest.raises(ValueError, match="duplicate stage"):
        run_assemble_phase2_generation_grid(args)
