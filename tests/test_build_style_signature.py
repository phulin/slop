import csv

import pytest

from slop_sftdiv.cli.build_style_signature import build_parser, run


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_build_style_signature_joins_tier1_compounding_and_biber(tmp_path, monkeypatch):
    tier1 = tmp_path / "tier1.csv"
    compounding = tmp_path / "compounding.csv"
    biber = tmp_path / "biber.csv"
    output = tmp_path / "signature.csv"
    distances = tmp_path / "distances.csv"
    summary = tmp_path / "signature.md"
    _write_csv(
        tier1,
        [
            {"stage": "base", "feature": "slop_lexicon", "per_1k_tokens": "1.0"},
            {"stage": "dpo", "feature": "slop_lexicon", "per_1k_tokens": "2.0"},
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
            },
        ],
    )
    _write_csv(
        biber,
        [
            {
                "stage": "base",
                "feature": "biber_lite_second_person_pronouns",
                "generation_per_1k_tokens": "4.0",
                "sft_target_per_1k_tokens": "2.0",
                "dpo_chosen_per_1k_tokens": "5.0",
                "pretrain_per_1k_tokens": "10.0",
            },
            {
                "stage": "final",
                "feature": "biber_lite_second_person_pronouns",
                "generation_per_1k_tokens": "8.0",
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
            "--biber-comparison",
            str(biber),
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
    final_biber = [
        row
        for row in rows
        if row["stage"] == "final"
        and row["feature_group"] == "biber_lite"
        and row["feature"] == "biber_lite_second_person_pronouns"
    ][0]
    assert final_slop["delta_vs_base"] == pytest.approx(0.5)
    assert final_slop["delta_final_vs_dpo"] == pytest.approx(-0.5)
    assert final_biber["log2_ratio_vs_sft_target"] == pytest.approx(2.0)
    with distances.open(encoding="utf-8", newline="") as handle:
        distance_rows = list(csv.DictReader(handle))
    assert distance_rows
    assert "Final vs Base" in summary.read_text(encoding="utf-8")
    assert logged_tables["style_signature"] == rows
    assert "style_signature_distances" in logged_tables
