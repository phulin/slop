import csv
import json

import pytest

from slop_sftdiv.cli.compare_phase2_biber import build_parser, run


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_compare_phase2_biber_joins_generation_to_corpus_rates(tmp_path, monkeypatch):
    generation_path = tmp_path / "generation.jsonl"
    corpus_path = tmp_path / "corpus.csv"
    output_path = tmp_path / "biber.csv"
    summary_path = tmp_path / "biber.md"
    generation_path.write_text(
        json.dumps(
            {
                "generation": "You can test this because you can inspect it.",
                "temperature": 1.0,
                "top_p": 0.95,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        corpus_path,
        [
            {
                "source": "sft",
                "role": "target_response",
                "feature": "biber_lite_second_person_pronouns",
                "count": "10",
                "docs": "2",
                "tokens": "1000",
                "per_1k_tokens": "10.0",
            },
            {
                "source": "dpo",
                "role": "chosen",
                "feature": "biber_lite_second_person_pronouns",
                "count": "5",
                "docs": "2",
                "tokens": "1000",
                "per_1k_tokens": "5.0",
            },
            {
                "source": "sft",
                "role": "target_response",
                "feature": "biber_lite_possibility_modals",
                "count": "20",
                "docs": "2",
                "tokens": "1000",
                "per_1k_tokens": "20.0",
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
        "slop_sftdiv.cli.compare_phase2_biber.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.compare_phase2_biber.log_summary_table",
        lambda _run, table_name, rows: logged_tables.setdefault(table_name, rows),
    )
    args = build_parser().parse_args(
        [
            "--generation-cache",
            f"dpo={generation_path}",
            "--corpus-rates",
            str(corpus_path),
            "--feature",
            "biber_lite_second_person_pronouns",
            "--feature",
            "biber_lite_possibility_modals",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    rows = run(args)

    by_feature = {row["feature"]: row for row in rows}
    second_person = by_feature["biber_lite_second_person_pronouns"]
    possibility = by_feature["biber_lite_possibility_modals"]
    assert second_person["generation_count"] == 2
    assert second_person["generation_tokens"] == 9
    assert second_person["generation_per_1k_tokens"] == pytest.approx(2 / 9 * 1000)
    assert second_person["sft_target_per_1k_tokens"] == pytest.approx(10.0)
    assert second_person["delta_vs_sft_target_per_1k_tokens"] == pytest.approx(
        2 / 9 * 1000 - 10.0
    )
    assert second_person["ratio_vs_dpo_chosen_per_1k_tokens"] == pytest.approx(
        (2 / 9 * 1000) / 5.0
    )
    assert possibility["generation_count"] == 2
    assert "Largest Generation-vs-SFT Target Deltas" in summary_path.read_text(
        encoding="utf-8"
    )
    assert logged_tables["phase2_biber_generation_vs_corpus"] == rows
