from __future__ import annotations

import json
from pathlib import Path

from slop_sftdiv.cli.pangram_detector_ar_report import summarize


def _write_summary(
    path: Path,
    *,
    initial: float,
    final: float,
    continuation: str,
    experiment: str = "pangram_detector_ar_search",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment": experiment,
                "initial_editlens_score": initial,
                "final_editlens_score": final,
                "final_bucket_probabilities": [0.1, 0.2, 0.3, 0.4],
                "final_mean_lm_log_probability": -2.0,
                "lm_logprob_coeff": 0.1,
                "repetition_penalty": 0.08,
                "beam_size": 4,
                "candidate_top_k": 48,
                "prompt": "Prompt",
                "optimized_continuation": continuation,
            }
        ),
        encoding="utf-8",
    )


def test_summarize_finds_nested_ar_search_summaries(tmp_path: Path) -> None:
    _write_summary(
        tmp_path / "run_00" / "pangram_detector_ar_search_summary.json",
        initial=0.1,
        final=0.6,
        continuation="hello\nworld",
    )
    _write_summary(
        tmp_path / "run_01" / "pangram_detector_ar_search_summary.json",
        initial=0.2,
        final=0.5,
        continuation="short",
    )
    _write_summary(
        tmp_path / "sample_00" / "pangram_detector_sample_search_summary.json",
        initial=0.3,
        final=0.7,
        continuation="sample",
        experiment="pangram_detector_sample_search",
    )

    rows = summarize([tmp_path], max_continuation_chars=12)

    assert [round(row["delta_score"], 1) for row in rows] == [0.5, 0.3, 0.4]
    assert rows[0]["bucket3_probability"] == 0.4
    assert rows[0]["computed_sequence_quality_penalty"] == 0.0
    assert rows[0]["optimized_continuation"] == "hello\\nworld"
    assert rows[2]["experiment"] == "pangram_detector_sample_search"
