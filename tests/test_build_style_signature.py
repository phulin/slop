import csv

import pytest

from slop_sftdiv.cli.build_style_signature import build_parser, run


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=sorted({key for row in rows for key in row}),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_build_style_signature_joins_tier1_compounding_and_pybiber(tmp_path, monkeypatch):
    tier1 = tmp_path / "tier1.csv"
    compounding = tmp_path / "compounding.csv"
    register = tmp_path / "register.csv"
    output = tmp_path / "signature.csv"
    distances = tmp_path / "distances.csv"
    summary = tmp_path / "signature.md"
    _write_csv(
        tier1,
        [
            {"stage": "base", "feature": "slop_lexicon", "per_1k_tokens": "1.0"},
            {
                "stage": "dpo",
                "feature": "slop_lexicon",
                "per_1k_tokens": "2.0",
                "per_1k_tokens_ci_low": "1.5",
                "per_1k_tokens_ci_high": "2.5",
            },
            {"stage": "final", "feature": "slop_lexicon", "per_1k_tokens": "1.5"},
        ],
    )
    _write_csv(
        compounding,
        [
            {
                "stage": "base",
                "feature": "slop_lexicon",
                "risk_ratio_after_prior": "2.0",
                "risk_diff_after_prior": "-0.1",
                "observed_per_1k_opportunities": "0.5",
            },
            {
                "stage": "final",
                "feature": "slop_lexicon",
                "risk_ratio_after_prior": "4.0",
                "risk_diff_after_prior": "0.3",
                "observed_per_1k_opportunities": "0.8",
                "observed_per_1k_opportunities_ci_low": "0.6",
                "observed_per_1k_opportunities_ci_high": "1.0",
            },
        ],
    )
    _write_csv(
        register,
        [
            {
                "stage": "base",
                "feature": "f_07_second_person_pronouns",
                "generation_per_1k_tokens": "4.0",
                "sft_target_per_1k_tokens": "2.0",
                "dpo_chosen_per_1k_tokens": "5.0",
                "pretrain_per_1k_tokens": "10.0",
            },
            {
                "stage": "final",
                "feature": "f_07_second_person_pronouns",
                "generation_per_1k_tokens": "8.0",
                "generation_per_1k_tokens_ci_low": "7.0",
                "generation_per_1k_tokens_ci_high": "9.0",
                "sft_target_per_1k_tokens": "2.0",
                "dpo_chosen_per_1k_tokens": "5.0",
                "pretrain_per_1k_tokens": "10.0",
            },
        ],
    )

    logged_tables = {}

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(
        "slop_sftdiv.cli.build_style_signature.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.build_style_signature.log_summary_table",
        lambda _run, table_name, rows: logged_tables.setdefault(table_name, rows),
    )
    args = build_parser().parse_args(
        [
            "--tier1-generation-grid",
            str(tier1),
            "--compounding",
            str(compounding),
            "--register-comparison",
            str(register),
            "--output",
            str(output),
            "--distance-output",
            str(distances),
            "--summary-output",
            str(summary),
            "--wandb-mode",
            "disabled",
        ]
    )

    rows = run(args)

    final_slop = [
        row
        for row in rows
        if row["stage"] == "final"
        and row["feature_group"] == "tier1"
        and row["feature"] == "slop_lexicon"
    ][0]
    final_register = [
        row
        for row in rows
        if row["stage"] == "final"
        and row["feature_group"] == "pybiber"
        and row["feature"] == "f_07_second_person_pronouns"
    ][0]
    final_compounding = [
        row
        for row in rows
        if row["stage"] == "final"
        and row["feature_group"] == "compounding"
        and row["metric"] == "observed_per_1k_opportunities"
    ][0]
    dpo_tier1 = [
        row
        for row in rows
        if row["stage"] == "dpo"
        and row["feature_group"] == "tier1"
        and row["feature"] == "slop_lexicon"
    ][0]
    assert final_slop["delta_vs_base"] == pytest.approx(0.5)
    assert final_slop["delta_final_vs_dpo"] == pytest.approx(-0.5)
    assert final_register["log2_ratio_vs_sft_target"] == pytest.approx(2.0)
    assert final_register["value_ci_low"] == pytest.approx(7.0)
    assert final_register["value_ci_high"] == pytest.approx(9.0)
    assert final_compounding["value_ci_low"] == pytest.approx(0.6)
    assert final_compounding["value_ci_high"] == pytest.approx(1.0)
    assert dpo_tier1["value_ci_low"] == pytest.approx(1.5)
    with distances.open(encoding="utf-8", newline="") as handle:
        distance_rows = list(csv.DictReader(handle))
    assert distance_rows
    assert "Final vs Base" in summary.read_text(encoding="utf-8")
    assert logged_tables["style_signature"] == rows
    assert "style_signature_distances" in logged_tables
