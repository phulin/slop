import pandas as pd

from slop_sftdiv.cli.assemble_phase1 import build_parser, run_assemble_phase1


def test_assemble_phase1_writes_parquets_and_summary(tmp_path, monkeypatch):
    feature_rates = tmp_path / "features.csv"
    pair_deltas = tmp_path / "pairs.csv"
    pd.DataFrame(
        [
            {
                "source": "sft",
                "subset": "math",
                "role": "target_response",
                "feature": "slop_lexicon",
                "count": 2,
                "docs": 1,
                "tokens": 100,
                "per_1k_tokens": 20.0,
                "per_doc": 2.0,
            },
            {
                "source": "sft",
                "subset": "code",
                "role": "target_response",
                "feature": "stock_openers_closers",
                "count": 1,
                "docs": 1,
                "tokens": 50,
                "per_1k_tokens": 20.0,
                "per_doc": 1.0,
            },
        ]
    ).to_csv(feature_rates, index=False)
    pd.DataFrame(
        [
            {
                "source": "dpo",
                "subset": "unknown",
                "pair_id": "pair-1",
                "preference_type": "delta_learning",
                "chosen_model": "strong",
                "rejected_model": "weak",
                "feature": "slop_lexicon",
                "chosen_count": 1,
                "rejected_count": 0,
                "chosen_tokens": 10,
                "rejected_tokens": 10,
                "chosen_rate_per_1k_tokens": 100.0,
                "rejected_rate_per_1k_tokens": 0.0,
                "chosen_minus_rejected": 100.0,
            }
        ]
    ).to_csv(pair_deltas, index=False)
    logged = {}

    class FakeRun:
        def log(self, payload):
            logged.update(payload)

        def finish(self):
            pass

    monkeypatch.setattr("slop_sftdiv.cli.assemble_phase1.init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr(
        "slop_sftdiv.cli.assemble_phase1.log_summary_table",
        lambda _run, table_name, rows: logged.setdefault(table_name, rows),
    )

    corpus_output = tmp_path / "feature_rates_by_corpus.parquet"
    stratum_output = tmp_path / "feature_rates_by_stratum.parquet"
    pair_output = tmp_path / "preference_pair_deltas.parquet"
    summary_output = tmp_path / "census_summary.md"
    args = build_parser().parse_args(
        [
            "--feature-rates",
            str(feature_rates),
            "--pair-deltas",
            str(pair_deltas),
            "--output-feature-corpus",
            str(corpus_output),
            "--output-feature-stratum",
            str(stratum_output),
            "--output-pair-deltas",
            str(pair_output),
            "--summary-output",
            str(summary_output),
            "--wandb-mode",
            "disabled",
        ]
    )

    summary = run_assemble_phase1(args)

    assert summary["feature_rates_by_corpus_rows"] == 2
    assert summary["preference_pair_ids"] == 1
    assert pd.read_parquet(corpus_output).shape[0] == 2
    assert pd.read_parquet(stratum_output).shape[0] == 2
    assert pd.read_parquet(pair_output).shape[0] == 1
    assert "Full pybiber extraction is handled by the separate" in summary_output.read_text(
        encoding="utf-8"
    )
    assert logged["assembly/preference_pair_delta_rows"] == 1
