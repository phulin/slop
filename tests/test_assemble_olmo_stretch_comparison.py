import csv

from slop_sftdiv.cli.assemble_olmo_stretch_comparison import (
    build_parser,
    run_assemble_olmo_stretch_comparison,
)


HEADER = [
    "stage",
    "temperature",
    "top_p",
    "feature",
    "count",
    "generations",
    "generated_tokens",
    "per_1k_tokens",
    "per_generation",
    "share_generations_with_feature",
]


def _write_grid(path):
    rows = [
        ("base", "slop_lexicon", 1.0),
        ("instruct_final", "slop_lexicon", 2.0),
        ("think_sft", "slop_lexicon", 3.0),
        ("think_dpo", "slop_lexicon", 4.0),
        ("think_final", "slop_lexicon", 5.0),
        ("rl_zero_general", "slop_lexicon", 6.0),
        ("rl_zero_if", "slop_lexicon", 8.0),
        ("base", "stock_openers", 0.5),
        ("instruct_final", "stock_openers", 1.5),
        ("think_final", "stock_openers", 1.0),
        ("rl_zero_general", "stock_openers", 0.75),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for stage, feature, rate in rows:
            writer.writerow(
                {
                    "stage": stage,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "feature": feature,
                    "count": int(rate * 10),
                    "generations": 10,
                    "generated_tokens": 1000,
                    "per_1k_tokens": rate,
                    "per_generation": rate / 10,
                    "share_generations_with_feature": 0.1,
                }
            )


def test_assemble_olmo_stretch_comparison_writes_deltas(tmp_path, monkeypatch):
    grid_path = tmp_path / "grid.csv"
    output_path = tmp_path / "comparison.csv"
    feature_summary_path = tmp_path / "feature_summary.csv"
    summary_path = tmp_path / "summary.md"
    _write_grid(grid_path)
    logged_payloads = []
    logged_tables = {}

    class FakeRun:
        def log(self, payload):
            logged_payloads.append(payload)

        def finish(self):
            pass

    monkeypatch.setattr(
        "slop_sftdiv.cli.assemble_olmo_stretch_comparison.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.assemble_olmo_stretch_comparison.log_summary_table",
        lambda _run, table_name, rows: logged_tables.setdefault(table_name, rows),
    )

    args = build_parser().parse_args(
        [
            "--generation-grid",
            str(grid_path),
            "--primary-feature",
            "slop_lexicon",
            "--output",
            str(output_path),
            "--feature-summary-output",
            str(feature_summary_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    summary = run_assemble_olmo_stretch_comparison(args)

    assert summary["rows"] == 11
    assert summary["primary_feature_max_stage"] == "rl_zero_if"
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    slop_rl_if = next(
        row
        for row in rows
        if row["feature"] == "slop_lexicon" and row["stage"] == "rl_zero_if"
    )
    assert float(slop_rl_if["delta_vs_instruct_final"]) == 6.0
    assert float(slop_rl_if["delta_vs_think_final"]) == 3.0
    with feature_summary_path.open("r", encoding="utf-8", newline="") as handle:
        feature_rows = {row["feature"]: row for row in csv.DictReader(handle)}
    assert feature_rows["slop_lexicon"]["max_stage"] == "rl_zero_if"
    assert float(feature_rows["slop_lexicon"]["rl_zero_mean_per_1k_tokens"]) == 7.0
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "OLMo Think/RL-Zero Stretch Comparison" in summary_text
    assert "Primary-feature maximum: `rl_zero_if`" in summary_text
    assert logged_payloads[-1]["phase3_olmo_stretch/primary_feature_max_stage"] == "rl_zero_if"
    assert len(logged_tables["phase3_olmo_stretch_feature_summary"]) == 2
