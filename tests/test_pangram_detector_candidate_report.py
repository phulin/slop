from __future__ import annotations

import csv
from pathlib import Path

from slop_sftdiv.cli.pangram_detector_candidate_report import summarize_candidates


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_candidates_reads_ar_and_sample_outputs(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "ar" / "pangram_detector_ar_search_final_beams.csv",
        [
            {
                "beam_rank": 0,
                "editlens_score": 0.8,
                "objective": 0.4,
                "mean_lm_log_probability": -2.0,
                "sequence_quality_penalty": 2.0,
                "continuation": "prefix=twitter%20schema",
            }
        ],
    )
    _write_csv(
        tmp_path / "sample" / "pangram_detector_sample_search_samples.csv",
        [
            {
                "sample_index": 0,
                "editlens_score": 0.5,
                "objective": 0.5,
                "mean_lm_log_probability": -1.0,
                "sequence_quality_penalty": 0.0,
                "continuation": "ordinary natural continuation",
            }
        ],
    )

    rows = summarize_candidates(
        [tmp_path],
        sort_by="objective",
        top_k=2,
        max_continuation_chars=100,
    )

    assert [row["candidate_kind"] for row in rows] == ["sample_search", "ar_search"]
    assert rows[0]["editlens_score"] == 0.5
    assert rows[1]["computed_sequence_quality_penalty"] > 1.0
