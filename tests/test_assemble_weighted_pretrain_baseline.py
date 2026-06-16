import csv

import pytest

from slop_sftdiv.cli.assemble_weighted_pretrain_baseline import build_parser, run


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_assemble_weighted_pretrain_baseline_reports_coverage(tmp_path):
    weights = tmp_path / "weights.csv"
    rates = tmp_path / "rates.csv"
    output = tmp_path / "weighted.csv"
    summary = tmp_path / "weighted.md"
    _write_csv(
        weights,
        [
            {"source_name": "fineweb-edu", "share": "0.3", "tokens": "300"},
            {"source_name": "dclm", "share": "0.6", "tokens": "600"},
            {"source_name": "stackexchange", "share": "0.1", "tokens": "100"},
        ],
    )
    _write_csv(
        rates,
        [
            {
                "source": "smollm3_pretrain_fineweb_edu_2k",
                "subset": "unknown",
                "role": "pretrain_document",
                "feature": "slop_lexicon",
                "count": "9",
                "docs": "2",
                "tokens": "3000",
                "per_1k_tokens": "3.0",
                "per_doc": "4.5",
            },
            {
                "source": "smollm3_pretrain_stackexchange_apple_2k",
                "subset": "unknown",
                "role": "pretrain_document",
                "feature": "slop_lexicon",
                "count": "1",
                "docs": "2",
                "tokens": "1000",
                "per_1k_tokens": "1.0",
                "per_doc": "0.5",
            },
        ],
    )
    args = build_parser().parse_args(
        [
            "--source-weights",
            str(weights),
            "--feature-rates",
            str(rates),
            "--source-map",
            "smollm3_pretrain_fineweb_edu_2k=fineweb-edu",
            "--source-map",
            "smollm3_pretrain_stackexchange_apple_2k=stackexchange",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ]
    )

    rows = run(args)

    assert len(rows) == 1
    row = rows[0]
    assert row["feature"] == "slop_lexicon"
    assert row["covered_recipe_share"] == pytest.approx(0.4)
    assert row["missing_recipe_share"] == pytest.approx(0.6)
    assert row["weighted_per_1k_tokens_covered_only"] == pytest.approx(2.5)
    assert row["weighted_per_1k_tokens_missing_as_zero"] == pytest.approx(1.0)
    assert row["sample_tokens"] == pytest.approx(4000.0)
    assert "missing recipe sources" in summary.read_text(encoding="utf-8")


def test_assemble_weighted_pretrain_baseline_token_weights_multi_subset_source(tmp_path):
    weights = tmp_path / "weights.csv"
    rates = tmp_path / "rates.csv"
    output = tmp_path / "weighted.csv"
    summary = tmp_path / "weighted.md"
    _write_csv(
        weights,
        [
            {"source_name": "infiwebmath-4plus", "share": "0.25", "tokens": "250"},
            {"source_name": "dclm", "share": "0.75", "tokens": "750"},
        ],
    )
    _write_csv(
        rates,
        [
            {
                "source": "smollm3_pretrain_infiwebmath_4plus_2k",
                "subset": "web_cc",
                "role": "pretrain_document",
                "feature": "slop_lexicon",
                "count": "20",
                "docs": "10",
                "tokens": "1000",
                "per_1k_tokens": "20.0",
                "per_doc": "2.0",
            },
            {
                "source": "smollm3_pretrain_infiwebmath_4plus_2k",
                "subset": "forums_qa",
                "role": "pretrain_document",
                "feature": "slop_lexicon",
                "count": "1",
                "docs": "1",
                "tokens": "1000",
                "per_1k_tokens": "1.0",
                "per_doc": "1.0",
            },
        ],
    )
    args = build_parser().parse_args(
        [
            "--source-weights",
            str(weights),
            "--feature-rates",
            str(rates),
            "--source-map",
            "smollm3_pretrain_infiwebmath_4plus_2k=infiwebmath-4plus",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ]
    )

    rows = run(args)

    assert len(rows) == 1
    row = rows[0]
    assert row["covered_recipe_share"] == pytest.approx(0.25)
    assert row["weighted_per_1k_tokens_covered_only"] == pytest.approx(10.5)
    assert row["weighted_per_1k_tokens_missing_as_zero"] == pytest.approx(2.625)
    assert row["sample_count"] == pytest.approx(21.0)
    assert row["sample_tokens"] == pytest.approx(2000.0)
