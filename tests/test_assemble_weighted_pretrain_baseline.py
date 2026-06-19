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


def test_assemble_weighted_pretrain_baseline_tracks_explicit_source_proxies(tmp_path):
    weights = tmp_path / "weights.csv"
    rates = tmp_path / "rates.csv"
    output = tmp_path / "weighted.csv"
    summary = tmp_path / "weighted.md"
    _write_csv(
        weights,
        [
            {"source_name": "stack-edu-Python", "share": "0.2", "tokens": "200"},
            {"source_name": "stack-edu-real-shuffled-Python", "share": "0.3", "tokens": "300"},
            {"source_name": "dclm", "share": "0.5", "tokens": "500"},
        ],
    )
    _write_csv(
        rates,
        [
            {
                "source": "smollm3_pretrain_stack_edu_python_2k",
                "subset": "unknown",
                "role": "pretrain_document",
                "feature": "slop_lexicon",
                "count": "6",
                "docs": "2",
                "tokens": "3000",
                "per_1k_tokens": "2.0",
                "per_doc": "3.0",
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
            "smollm3_pretrain_stack_edu_python_2k=stack-edu-Python",
            "--source-proxy",
            "stack-edu-Python=stack-edu-real-shuffled-Python",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ]
    )

    rows = run(args)

    assert len(rows) == 1
    row = rows[0]
    assert row["exact_matched_source_count"] == 1
    assert row["proxy_source_count"] == 1
    assert row["exact_covered_recipe_share"] == pytest.approx(0.2)
    assert row["proxy_covered_recipe_share"] == pytest.approx(0.3)
    assert row["covered_recipe_share"] == pytest.approx(0.5)
    assert row["missing_recipe_share"] == pytest.approx(0.5)
    assert row["weighted_per_1k_tokens_covered_only"] == pytest.approx(2.0)
    assert row["weighted_per_1k_tokens_missing_as_zero"] == pytest.approx(1.0)
    assert row["sample_count"] == pytest.approx(6.0)
    assert row["sample_tokens"] == pytest.approx(3000.0)
    assert "stack-edu-real-shuffled-Python:2.000000@0.300000~proxy_from=stack-edu-Python" in row[
        "proxy_recipe_sources"
    ]
    assert "explicit source proxies" in summary.read_text(encoding="utf-8")
