from __future__ import annotations

from argparse import Namespace

import pandas as pd

from slop_sftdiv.cli.render_paper_figures import run_render_paper_figures


def test_render_paper_figures_writes_svg_outputs(tmp_path):
    pybiber = tmp_path / "pybiber.csv"
    pd.DataFrame(
        [
            {"sample": sample, "feature": feature, "token_weighted_mean": value}
            for sample, base in [
                ("dolma_pretrain", 10),
                ("sft_target", 20),
                ("dpo_chosen", 30),
                ("dpo_rejected", 25),
            ]
            for feature, value in [
                ("f_01_past_tense", base + 1),
                ("f_06_first_person_pronouns", base + 2),
                ("f_08_third_person_pronouns", base + 3),
                ("f_14_nominalizations", base + 4),
                ("f_39_prepositions", base + 5),
                ("f_40_adj_attr", base + 6),
                ("f_42_adverbs", base + 7),
                ("f_47_hedges", base + 8),
                ("f_48_amplifiers", base + 9),
                ("f_49_emphatics", base + 10),
            ]
        ]
    ).to_csv(pybiber, index=False)
    pybiber_intervals = tmp_path / "pybiber_intervals.csv"
    pd.DataFrame(
        [
            {
                "statistic": "mean",
                "sample": sample,
                "feature": feature,
                "ci_low": value - 0.5,
                "ci_high": value + 0.5,
            }
            for sample, base in [
                ("dolma_pretrain", 10),
                ("sft_target", 20),
                ("dpo_chosen", 30),
                ("dpo_rejected", 25),
            ]
            for feature, value in [
                ("f_01_past_tense", base + 1),
                ("f_06_first_person_pronouns", base + 2),
                ("f_08_third_person_pronouns", base + 3),
                ("f_14_nominalizations", base + 4),
                ("f_39_prepositions", base + 5),
                ("f_40_adj_attr", base + 6),
                ("f_42_adverbs", base + 7),
                ("f_47_hedges", base + 8),
                ("f_48_amplifiers", base + 9),
                ("f_49_emphatics", base + 10),
            ]
        ]
    ).to_csv(pybiber_intervals, index=False)

    eqbench = tmp_path / "eqbench.csv"
    eqbench_rows = [
        {"ladder": ladder, "stage": stage, "eqbench_slop_score": score}
        for ladder, offset in [("olmo3", 0), ("smollm3_no_think", 2)]
        for stage, score in [
            ("base", 1 + offset),
            ("sft", 2 + offset),
            ("dpo", 3 + offset),
            ("final", 2.5 + offset),
        ]
    ]
    pd.DataFrame(eqbench_rows).to_csv(eqbench, index=False)
    eqbench_intervals = tmp_path / "eqbench_intervals.csv"
    pd.DataFrame(
        [
            {
                **row,
                "eqbench_slop_score_ci_low": row["eqbench_slop_score"] - 0.1,
                "eqbench_slop_score_ci_high": row["eqbench_slop_score"] + 0.1,
            }
            for row in eqbench_rows
        ]
    ).to_csv(eqbench_intervals, index=False)

    spectrum = tmp_path / "spectrum.csv"
    pd.DataFrame(
        [
            {
                "feature": "slop_lexicon",
                "stage": stage,
                "teacher_forced_normalized_af": 1 + index,
                "free_run_per_1k_tokens": 0.1 + index,
                "compounding_excess_per_1k_opportunities": 0.2 + index,
            }
            for index, stage in enumerate(["base", "sft", "dpo", "final"])
        ]
    ).to_csv(spectrum, index=False)

    cross_ladder = tmp_path / "cross_ladder.csv"
    pd.DataFrame(
        [
            {
                "feature": feature,
                "stage_label": stage,
                "left_label": "OLMo",
                "right_label": "SmolLM3",
                "left_af": left_af,
                "right_af": right_af,
            }
            for feature, left_af, right_af in [
                ("slop_lexicon", 1.5, 7.0),
                ("rule_of_three_approx", 0.7, 0.8),
            ]
            for stage in ["Base", "SFT"]
        ]
    ).to_csv(cross_ladder, index=False)

    phase4 = tmp_path / "phase4.csv"
    pd.DataFrame(
        [
            {
                "feature": feature,
                "stage": stage,
                "amplification_factor": 1 + feature_index + stage_index,
                "reference_initiations": 10 + feature_index,
            }
            for feature_index, feature in enumerate(
                [
                    "phase4_ig_process_framing",
                    "phase4_ig_additive_transition",
                    "phase4_ig_prescriptive_instruction",
                    "phase4_ig_response_constraint",
                    "phase4_ig_followup_offer",
                    "neutral_common_controls",
                ]
            )
            for stage_index, stage in enumerate(["base", "sft", "dpo", "final"])
        ]
    ).to_csv(phase4, index=False)

    output_dir = tmp_path / "figures"
    outputs = run_render_paper_figures(
        Namespace(
            pybiber_means=pybiber,
            pybiber_intervals=pybiber_intervals,
            eqbench_stage_comparison=eqbench,
            eqbench_intervals=eqbench_intervals,
            olmo_spectrum=spectrum,
            cross_ladder_aligned=cross_ladder,
            phase4_stage_grid=phase4,
            output_dir=output_dir,
        )
    )

    assert [path.name for path in outputs] == [
        "figure1_pybiber_register_selected.svg",
        "figure2_eqbench_stage_scores.svg",
        "figure3_olmo_slop_lexicon_views.svg",
        "figure4_cross_ladder_af_scatter.svg",
        "figure5_phase4_tier3_raw_af.svg",
    ]
    for path in outputs:
        assert path.exists()
        assert "<svg" in path.read_text(encoding="utf-8")
        export_dir = path.parent / "submission_exports"
        assert (export_dir / f"{path.stem}.pdf").exists()
        assert (export_dir / f"{path.stem}.png").exists()
