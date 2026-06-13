from __future__ import annotations

import csv
import json

import pytest

from slop_sftdiv.cli.analyze_phase2_compounding import (
    build_parser,
    run_analyze_phase2_compounding,
)


def test_analyze_phase2_compounding_writes_conditional_rates(tmp_path, monkeypatch):
    generation_path = tmp_path / "generations.jsonl"
    propensity_path = tmp_path / "propensity.csv"
    output_path = tmp_path / "compounding.csv"
    summary_path = tmp_path / "compounding.md"
    generation_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source": "prompt_package",
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "generation": "alpha delve beta gamma delta delve epsilon zeta",
                    }
                ),
                json.dumps(
                    {
                        "source": "prompt_package",
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "generation": "alpha beta gamma delta",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with propensity_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "feature",
                "opportunities",
                "reference_initiations",
                "reference_rate",
                "mean_prob_mass",
                "amplification_factor",
                "normalized_amplification_factor",
                "normalization_feature",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "stage": "sft",
                "feature": "slop_lexicon",
                "opportunities": 100,
                "reference_initiations": 1,
                "reference_rate": 0.01,
                "mean_prob_mass": 0.1,
                "amplification_factor": 10.0,
                "normalized_amplification_factor": 2.0,
                "normalization_feature": "neutral_common_controls",
            }
        )
    logged_payloads = []
    logged_tables = {}

    class FakeRun:
        def log(self, payload):
            logged_payloads.append(payload)

        def finish(self):
            pass

    monkeypatch.setattr(
        "slop_sftdiv.cli.analyze_phase2_compounding.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.analyze_phase2_compounding.log_summary_table",
        lambda _run, table_name, rows: logged_tables.setdefault(table_name, rows),
    )

    args = build_parser().parse_args(
        [
            "--generation-cache",
            f"sft={generation_path}",
            "--feature",
            "slop_lexicon",
            "--window-tokens",
            "2",
            "--propensity-grid",
            str(propensity_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    rows = run_analyze_phase2_compounding(args)

    assert len(rows) == 1
    row = rows[0]
    assert row["stage"] == "sft"
    assert row["windows"] == 6
    assert row["prior_windows"] == 3
    assert row["no_prior_windows"] == 3
    assert row["hit_windows_after_prior"] == 1
    assert row["hit_windows_without_prior"] == 1
    assert row["p_hit_after_prior"] == pytest.approx(1 / 3)
    assert row["p_hit_without_prior"] == pytest.approx(1 / 3)
    assert row["feature_hits"] == 2
    assert row["repeated_hits"] == 1
    assert row["repeat_generations"] == 1
    assert row["generation_opportunities"] == 12
    assert row["observed_initiations"] == 2
    assert row["observed_rate"] == pytest.approx(1 / 6)
    assert row["expected_rate"] == pytest.approx(0.1)
    assert row["excess_rate"] == pytest.approx(1 / 6 - 0.1)
    assert row["observed_over_expected"] == pytest.approx(5 / 3)
    assert row["realized_af"] == pytest.approx(50 / 3)
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        output_rows = list(csv.DictReader(handle))
    assert output_rows[0]["feature"] == "slop_lexicon"
    assert "Phase 2 Compounding Analysis" in summary_path.read_text(encoding="utf-8")
    assert logged_payloads[-1]["phase2_compounding/rows"] == 1
    assert logged_tables["phase2_compounding"] == rows


def test_analyze_phase2_compounding_rejects_duplicate_stage(tmp_path):
    generation_path = tmp_path / "generations.jsonl"
    generation_path.write_text("", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--generation-cache",
            f"sft={generation_path}",
            "--generation-cache",
            f"sft={generation_path}",
            "--output",
            str(tmp_path / "out.csv"),
            "--summary-output",
            str(tmp_path / "out.md"),
        ]
    )

    with pytest.raises(ValueError, match="duplicate stage"):
        run_analyze_phase2_compounding(args)
