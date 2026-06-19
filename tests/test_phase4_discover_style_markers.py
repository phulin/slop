from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

from slop_sftdiv.cli.phase4_discover_style_markers import run


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_phase4_discovery_outputs_candidates_clusters_and_census(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    generation = tmp_path / "generation.jsonl"
    _write_jsonl(
        reference,
        [
            {"doc_id": f"r{i}", "text": "This answer gives a direct short result."}
            for i in range(12)
        ],
    )
    _write_jsonl(
        generation,
        [
            {
                "record_id": f"g{i}",
                "generation": "assistant\nLet's break this down step by step. "
                "This is a comprehensive answer with clear steps.",
            }
            for i in range(12)
        ],
    )

    output_dir = tmp_path / "phase4"
    summary = run(
        Namespace(
            reference_jsonl=[f"ref={reference}"],
            generation_jsonl=[f"model={generation}"],
            output_dir=output_dir,
            max_docs_per_source=100,
            max_examples_per_candidate=2,
            ngram_min=1,
            ngram_max=4,
            min_model_docs=2,
            min_model_count=2,
            min_lift=1.1,
            top_k=20,
            matcher_top_k=5,
            feature_text_mode="final_answer",
        )
    )

    assert summary["candidate_count"] > 0
    candidates = list(csv.DictReader((output_dir / "phase4_candidate_spans.csv").open()))
    assert any(row["candidate"] == "let's break" for row in candidates)

    clusters = list(csv.DictReader((output_dir / "phase4_candidate_clusters.csv").open()))
    assert any(row["cluster"] == "assistant_process_framing" for row in clusters)

    specs = json.loads((output_dir / "phase4_matcher_specs.json").read_text(encoding="utf-8"))
    assert specs
    assert specs[0]["precision_status"] == "requires_200_hit_manual_validation_before_core_claim"

    census = list(csv.DictReader((output_dir / "phase4_candidate_census.csv").open()))
    model_rows = [row for row in census if row["source"] == "model"]
    assert model_rows
    assert max(float(row["per_1k_tokens"]) for row in model_rows) > 0.0
