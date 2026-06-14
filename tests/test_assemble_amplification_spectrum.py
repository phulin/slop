import csv
import json

from slop_sftdiv.cli.assemble_amplification_spectrum import build_parser, run


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_assemble_amplification_spectrum_merges_existing_artifacts(tmp_path, monkeypatch):
    feature_rates = tmp_path / "feature_rates.csv"
    propensity = tmp_path / "propensity.csv"
    generation = tmp_path / "generation.csv"
    compounding = tmp_path / "compounding.csv"
    denominators = tmp_path / "denominators.csv"
    output = tmp_path / "spectrum.csv"
    summary = tmp_path / "spectrum.md"

    _write_csv(
        feature_rates,
        [
            {
                "source": "sft",
                "role": "target_response",
                "feature": "slop_lexicon",
                "count": "10",
                "docs": "2",
                "tokens": "1000",
                "per_1k_tokens": "10.0",
            },
            {
                "source": "dpo",
                "role": "chosen",
                "feature": "slop_lexicon",
                "count": "15",
                "docs": "2",
                "tokens": "1000",
                "per_1k_tokens": "15.0",
            },
        ],
    )
    _write_csv(
        propensity,
        [
            {
                "stage": "dpo",
                "feature": "slop_lexicon",
                "opportunities": "100",
                "reference_initiations": "5",
                "reference_rate": "0.05",
                "mean_prob_mass": "0.1",
                "amplification_factor": "2.0",
                "amplification_factor_ci_low": "1.0",
                "amplification_factor_ci_high": "3.0",
                "normalization_feature": "neutral_common_controls",
                "normalized_amplification_factor": "1.5",
                "normalized_amplification_factor_ci_low": "1.1",
                "normalized_amplification_factor_ci_high": "2.0",
            }
        ],
    )
    _write_csv(
        generation,
        [
            {
                "stage": "dpo",
                "feature": "slop_lexicon",
                "count": "8",
                "repeated_count": "3",
                "generations": "4",
                "generated_tokens": "1000",
                "per_1k_tokens": "8.0",
                "share_generations_with_feature": "0.5",
                "share_repeat_generations": "0.25",
            }
        ],
    )
    _write_csv(
        compounding,
        [
            {
                "stage": "dpo",
                "feature": "slop_lexicon",
                "observed_per_1k_opportunities": "6.0",
                "expected_per_1k_opportunities": "4.0",
                "excess_per_1k_opportunities": "2.0",
                "realized_af": "1.2",
                "repeat_generations": "2",
                "risk_diff_after_prior": "0.1",
                "risk_ratio_after_prior": "2.0",
            }
        ],
    )
    _write_csv(
        denominators,
        [
            {
                "feature": "slop_lexicon",
                "opportunities": "100",
                "reference_initiations": "5",
                "reference_rate": "0.05",
                "documents_with_reference": "4",
            },
            {
                "feature": "contrastive_negation",
                "opportunities": "1000",
                "reference_initiations": "7",
                "reference_rate": "0.007",
                "documents_with_reference": "7",
            },
        ],
    )

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(
        "slop_sftdiv.cli.assemble_amplification_spectrum.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    logged_tables = {}
    monkeypatch.setattr(
        "slop_sftdiv.cli.assemble_amplification_spectrum.log_summary_table",
        lambda _run, name, rows: logged_tables.setdefault(name, rows),
    )

    args = build_parser().parse_args(
        [
            "--feature-rate",
            str(feature_rates),
            "--propensity-grid",
            str(propensity),
            "--generation-grid",
            str(generation),
            "--compounding",
            str(compounding),
            "--denominator-support",
            str(denominators),
            "--feature",
            "slop_lexicon",
            "--feature",
            "contrastive_negation",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
            "--wandb-mode",
            "disabled",
        ]
    )

    rows = run(args)

    by_key = {(row["feature"], row["stage"]): row for row in rows}
    dpo_slop = by_key[("slop_lexicon", "dpo")]
    assert dpo_slop["sft_target_per_1k_tokens"] == 10.0
    assert dpo_slop["dpo_chosen_per_1k_tokens"] == 15.0
    assert dpo_slop["teacher_forced_normalized_af"] == 1.5
    assert dpo_slop["free_run_per_1k_tokens"] == 8.0
    assert dpo_slop["compounding_excess_per_1k_opportunities"] == 2.0
    assert by_key[("contrastive_negation", "base")]["coverage_note"].startswith(
        "teacher-forced missing"
    )
    assert "sparse held-out references: 7" in by_key[
        ("contrastive_negation", "base")
    ]["coverage_note"]
    assert output.exists()
    assert "SGLang target-shape generations are not used here" in summary.read_text()
    assert "amplification_spectrum" in logged_tables
