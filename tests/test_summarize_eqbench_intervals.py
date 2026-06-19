from __future__ import annotations

import csv
import json

from slop_sftdiv.cli.summarize_eqbench_intervals import build_parser, run_summarize_eqbench_intervals


SLOPPY_OUTPUT = (
    "It is not just a checklist, but a paradigm shift for the workflow. "
    "The assistant took deep breath, showed unwavering resolve, and handled "
    "the meticulous transition with scalable insights. "
) * 3


def test_summarize_eqbench_intervals_writes_csv_and_markdown(tmp_path):
    panel = tmp_path / "panel.jsonl"
    panel.write_text(
        "\n".join(
            [
                json.dumps({"generation": SLOPPY_OUTPUT}),
                json.dumps({"generation": SLOPPY_OUTPUT.replace("paradigm", "robust")}),
                json.dumps({"generation": "short"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "intervals.csv"
    summary = tmp_path / "intervals.md"
    args = build_parser().parse_args(
        [
            "--panel",
            f"test_ladder,test_stage,{panel}",
            "--min-chars",
            "80",
            "--bootstrap-samples",
            "20",
            "--seed",
            "7",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ]
    )

    rows = run_summarize_eqbench_intervals(args)

    assert len(rows) == 1
    row = rows[0]
    assert row["ladder"] == "test_ladder"
    assert row["stage"] == "test_stage"
    assert row["sample_count"] == 3
    assert row["valid_sample_count"] == 2
    assert row["eqbench_slop_score_ci_low"] <= row["eqbench_slop_score"]
    assert row["eqbench_slop_score_ci_high"] >= row["eqbench_slop_score"]
    with output.open(encoding="utf-8", newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert persisted[0]["ladder"] == "test_ladder"
    assert "95% bootstrap CI" in summary.read_text(encoding="utf-8")

