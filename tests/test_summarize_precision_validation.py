from __future__ import annotations

from argparse import Namespace

import pandas as pd

from slop_sftdiv.cli.summarize_precision_validation import run_summarize_precision_validation


def test_summarize_precision_validation_reports_core_gate_status(tmp_path):
    queue = tmp_path / "queue.csv"
    empty_queue = tmp_path / "empty_queue.csv"
    pd.DataFrame(
        [
            {"feature": "slop_lexicon", "sample_key": f"s{i}", "label": ""}
            for i in range(60)
        ]
        + [{"feature": "stock_openers_closers", "sample_key": "p0", "label": ""}]
    ).to_csv(queue, index=False)
    pd.DataFrame(
        columns=["feature", "sample_key", "label"]
    ).to_csv(empty_queue, index=False)

    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {"feature": "slop_lexicon", "sample_key": f"s{i}", "label": "exact"}
            for i in range(55)
        ]
        + [
            {"feature": "slop_lexicon", "sample_key": "s55", "label": "false_positive"},
            {"feature": "eqbench_slop_score", "sample_key": "e0", "label": "exact"},
        ]
    ).to_csv(labels, index=False)

    output_csv = tmp_path / "status.csv"
    output_md = tmp_path / "status.md"
    rows = run_summarize_precision_validation(
        Namespace(
            label_input=[labels],
            queue_input=[queue, empty_queue],
            feature=[],
            target_precision=0.90,
            max_ambiguous_rate=0.10,
            min_labeled=50,
            output_csv=output_csv,
            output_md=output_md,
        )
    )

    by_feature = {row["feature"]: row for row in rows}
    assert by_feature["slop_lexicon"]["gate_status"] == "pass"
    assert by_feature["stock_openers_closers"]["is_core"] is False
    assert by_feature["stock_openers_closers"]["is_derived"] is True
    assert by_feature["stock_openers_closers"]["gate_status"] == "derived"
    assert by_feature["eqbench_slop_score"]["gate_status"] == "incomplete"
    assert output_csv.exists()
    markdown = output_md.read_text(encoding="utf-8")
    assert "Precision Validation Status" in markdown
    assert "Queue files with zero retained feature hits" in markdown
