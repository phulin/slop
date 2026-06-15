import csv

from slop_sftdiv.cli.classify_amplification_spectrum import build_parser, run


def _write_csv(path, rows):
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _stage_rows(feature, *, sft_rate, chosen, rejected, afs, free=None, compounding=None):
    rows = []
    for stage, af in afs.items():
        row = {
            "feature": feature,
            "stage": stage,
            "stage_label": stage,
            "pretrain_per_1k_tokens": "0.1",
            "sft_target_per_1k_tokens": str(sft_rate),
            "dpo_chosen_per_1k_tokens": str(chosen),
            "dpo_rejected_per_1k_tokens": str(rejected),
            "teacher_forced_af": str(af),
            "teacher_forced_af_ci_low": str(af - 0.1),
            "teacher_forced_af_ci_high": str(af + 0.1),
            "free_run_per_1k_tokens": str((free or {}).get(stage, 0.0)),
            "coverage_note": "covered in bounded OLMo slice",
        }
        if compounding and stage in compounding:
            row.update(
                {
                    "compounding_excess_per_1k_opportunities": str(
                        compounding[stage]["excess"]
                    ),
                    "compounding_realized_af": str(compounding[stage]["realized_af"]),
                }
            )
        rows.append(row)
    return rows


def test_classify_amplification_spectrum_rules(tmp_path, monkeypatch):
    spectrum = tmp_path / "spectrum.csv"
    preference = tmp_path / "preference.csv"
    output = tmp_path / "phase3.csv"
    summary = tmp_path / "phase3.md"
    rows = [
        *_stage_rows(
            "pref_signal",
            sft_rate=0.8,
            chosen=1.4,
            rejected=0.9,
            afs={"base": 1.0, "sft": 1.1, "dpo": 1.7, "final": 1.5},
            free={"base": 0.2, "sft": 0.1, "dpo": 0.4, "final": 0.3},
        ),
        *_stage_rows(
            "pref_dynamics",
            sft_rate=0.8,
            chosen=0.7,
            rejected=0.9,
            afs={"base": 1.0, "sft": 1.1, "dpo": 1.7, "final": 1.5},
        ),
        *_stage_rows(
            "inherited_feature",
            sft_rate=0.8,
            chosen=0.7,
            rejected=0.7,
            afs={"base": 0.95, "sft": 1.0, "dpo": 1.05, "final": 1.02},
        ),
        *_stage_rows(
            "compounding_feature",
            sft_rate=0.1,
            chosen=0.1,
            rejected=0.1,
            afs={"base": 1.0, "sft": 1.0, "dpo": 1.1, "final": 1.0},
            compounding={"dpo": {"excess": 0.5, "realized_af": 1.4}},
        ),
        *_stage_rows(
            "high_af_tiny_relative_jump",
            sft_rate=0.8,
            chosen=1.0,
            rejected=0.8,
            afs={"base": 70.0, "sft": 65.0, "dpo": 68.0, "final": 71.0},
        ),
    ]
    _write_csv(spectrum, rows)
    _write_csv(
        preference,
        [
            {
                "feature": "pref_signal",
                "pairs": "100",
                "mean_delta": "0.5",
                "sign_test_p": "0.001",
                "bh_q": "0.01",
                "direction": "chosen_gt_rejected",
                "fdr_significant": "True",
            },
            {
                "feature": "pref_dynamics",
                "pairs": "100",
                "mean_delta": "0.5",
                "sign_test_p": "0.001",
                "bh_q": "0.01",
                "direction": "chosen_gt_rejected",
                "fdr_significant": "False",
            },
            {
                "feature": "pref_dynamics",
                "pairs": "10",
                "mean_delta": "5.0",
                "sign_test_p": "0.001",
                "bh_q": "0.01",
                "direction": "chosen_gt_rejected",
                "fdr_significant": "True",
            },
        ],
    )

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(
        "slop_sftdiv.cli.classify_amplification_spectrum.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    logged_tables = {}
    monkeypatch.setattr(
        "slop_sftdiv.cli.classify_amplification_spectrum.log_summary_table",
        lambda _run, name, rows: logged_tables.setdefault(name, rows),
    )

    args = build_parser().parse_args(
        [
            "--spectrum",
            str(spectrum),
            "--preference-analysis",
            str(preference),
            "--output",
            str(output),
            "--summary-output",
            str(summary),
            "--wandb-mode",
            "disabled",
        ]
    )
    classified = run(args)

    by_feature = {row["feature"]: row for row in classified}
    assert by_feature["pref_signal"]["primary_class"] == "preference-amplified"
    assert by_feature["pref_signal"]["preference_cause"] == "signal-driven"
    assert by_feature["pref_signal"]["preference_evidence"] == "paired_sign_test_bh_fdr"
    assert by_feature["pref_signal"]["preference_bh_q"] == 0.01
    assert by_feature["pref_signal"]["preference_data_complicit"] is True
    assert by_feature["pref_dynamics"]["primary_class"] == "preference-amplified"
    assert by_feature["pref_dynamics"]["preference_cause"] == "dynamics-driven"
    assert by_feature["pref_dynamics"]["preference_pairs"] == 100
    assert "without chosen>rejected" in by_feature["pref_dynamics"]["notes"]
    assert by_feature["inherited_feature"]["primary_class"] == "inherited"
    assert by_feature["compounding_feature"]["primary_class"] == "compounding-dominant"
    assert by_feature["high_af_tiny_relative_jump"]["primary_class"] == "sft-amplified"
    assert by_feature["high_af_tiny_relative_jump"]["preference_amplified"] is False
    assert by_feature["pref_signal"]["fdr_status"] == "preference_pair_fdr_available_af_fdr_missing"
    assert by_feature["inherited_feature"]["fdr_status"] == "not_computed_missing_p_values"
    assert output.exists()
    assert "Phase 3 Bounded Amplification-Spectrum Classification" in summary.read_text()
    assert "phase3_feature_classification" in logged_tables
