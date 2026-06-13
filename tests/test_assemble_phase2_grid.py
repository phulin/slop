from __future__ import annotations

import csv
from pathlib import Path

import pytest

from slop_sftdiv.cli.assemble_phase2_grid import build_parser, run_assemble_phase2_grid


HEADER = [
    "feature",
    "opportunities",
    "reference_initiations",
    "reference_rate",
    "reference_rate_ci_low",
    "reference_rate_ci_high",
    "mean_prob_mass",
    "mean_prob_mass_ci_low",
    "mean_prob_mass_ci_high",
    "amplification_factor",
    "amplification_factor_ci_low",
    "amplification_factor_ci_high",
    "normalization_feature",
    "normalized_amplification_factor",
    "normalized_amplification_factor_ci_low",
    "normalized_amplification_factor_ci_high",
]


def _write_summary(path: Path, *, slop_normalized_af: float, neutral_af: float = 0.3) -> None:
    rows = [
        {
            "feature": "neutral_common_controls",
            "opportunities": 10,
            "reference_initiations": 4,
            "reference_rate": 0.4,
            "reference_rate_ci_low": 0.3,
            "reference_rate_ci_high": 0.5,
            "mean_prob_mass": 0.12,
            "mean_prob_mass_ci_low": 0.1,
            "mean_prob_mass_ci_high": 0.14,
            "amplification_factor": neutral_af,
            "amplification_factor_ci_low": neutral_af - 0.01,
            "amplification_factor_ci_high": neutral_af + 0.01,
            "normalization_feature": "neutral_common_controls",
            "normalized_amplification_factor": 1.0,
            "normalized_amplification_factor_ci_low": 1.0,
            "normalized_amplification_factor_ci_high": 1.0,
        },
        {
            "feature": "slop_lexicon",
            "opportunities": 10,
            "reference_initiations": 1,
            "reference_rate": 0.1,
            "reference_rate_ci_low": 0.0,
            "reference_rate_ci_high": 0.2,
            "mean_prob_mass": 0.06,
            "mean_prob_mass_ci_low": 0.04,
            "mean_prob_mass_ci_high": 0.08,
            "amplification_factor": 0.6,
            "amplification_factor_ci_low": 0.4,
            "amplification_factor_ci_high": 0.8,
            "normalization_feature": "neutral_common_controls",
            "normalized_amplification_factor": slop_normalized_af,
            "normalized_amplification_factor_ci_low": slop_normalized_af - 0.2,
            "normalized_amplification_factor_ci_high": slop_normalized_af + 0.2,
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def test_assemble_phase2_grid_writes_stage_comparison(tmp_path, monkeypatch):
    base_path = tmp_path / "base.csv"
    dpo_path = tmp_path / "dpo.csv"
    _write_summary(base_path, slop_normalized_af=1.5)
    _write_summary(dpo_path, slop_normalized_af=2.5)
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

    monkeypatch.setattr("slop_sftdiv.cli.assemble_phase2_grid.init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr("slop_sftdiv.cli.assemble_phase2_grid.log_summary_table", fake_log_summary_table)

    args = build_parser().parse_args(
        [
            "--propensity-summary",
            f"base={base_path}",
            "--propensity-summary",
            f"dpo={dpo_path}",
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
    summary = run_assemble_phase2_grid(args)

    assert summary["grid_rows"] == 4
    assert summary["comparison_rows"] == 2
    assert summary["max_normalized_af_stage"] == "dpo"
    with comparison_path.open("r", encoding="utf-8", newline="") as handle:
        comparison_rows = list(csv.DictReader(handle))
    assert [row["stage"] for row in comparison_rows] == ["base", "dpo"]
    assert float(comparison_rows[0]["delta_normalized_af_vs_base"]) == pytest.approx(0.0)
    assert float(comparison_rows[1]["delta_normalized_af_vs_base"]) == pytest.approx(1.0)
    assert "Phase 2 Propensity Stage Grid" in summary_path.read_text(encoding="utf-8")
    assert logged_payloads[-1]["phase2_grid/max_normalized_af_stage"] == "dpo"
    assert len(logged_tables["phase2_propensity_grid"]) == 4
    assert len(logged_tables["phase2_primary_feature_comparison"]) == 2


def test_assemble_phase2_grid_rejects_duplicate_stage(tmp_path):
    path = tmp_path / "base.csv"
    _write_summary(path, slop_normalized_af=1.0)
    args = build_parser().parse_args(
        [
            "--propensity-summary",
            f"base={path}",
            "--propensity-summary",
            f"base={path}",
        ]
    )

    with pytest.raises(ValueError, match="duplicate stage"):
        run_assemble_phase2_grid(args)
