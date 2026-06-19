from __future__ import annotations

import pytest
import pandas as pd

from slop_sftdiv.cli.phase4_train_detector import (
    _build_hape_rows_from_by_doc,
    _hape_by_doc_from_texts_frame,
)


def _by_doc(doc_count: int = 8) -> dict[str, dict[str, str]]:
    return {
        f"doc_{index:02d}": {
            "chunk_1": f"prompt {index}",
            "chunk_2": f"human continuation {index}",
            "author_a": f"author a continuation {index}",
            "author_b": f"author b continuation {index}",
        }
        for index in range(doc_count)
    }


def test_hape_doc_percent_split_holds_out_base_docs_without_author_leakage() -> None:
    train_rows, eval_rows, summary = _build_hape_rows_from_by_doc(
        by_doc=_by_doc(8),
        split_mode="doc_percent",
        hape_authors=["author_a", "author_b"],
        train_authors=set(),
        heldout_authors=set(),
        holdout_percent=25.0,
        max_train_pairs_per_author=0,
        max_eval_pairs_per_author=0,
        seed=1729,
    )

    train_doc_ids = {row.doc_id for row in train_rows}
    eval_doc_ids = {row.doc_id for row in eval_rows}
    assert train_doc_ids.isdisjoint(eval_doc_ids)
    assert summary["train_doc_count"] == 6
    assert summary["heldout_doc_count"] == 2
    assert summary["train_pairs_by_author"] == {"author_a": 6, "author_b": 6}
    assert summary["heldout_pairs_by_author"] == {"author_a": 2, "author_b": 2}
    assert len(train_rows) == 24
    assert len(eval_rows) == 8


def test_hape_doc_percent_split_can_cap_pairs_per_author() -> None:
    train_rows, eval_rows, summary = _build_hape_rows_from_by_doc(
        by_doc=_by_doc(8),
        split_mode="doc_percent",
        hape_authors=["author_a", "author_b"],
        train_authors=set(),
        heldout_authors=set(),
        holdout_percent=25.0,
        max_train_pairs_per_author=1,
        max_eval_pairs_per_author=1,
        seed=1729,
    )

    assert summary["train_pairs_by_author"] == {"author_a": 1, "author_b": 1}
    assert summary["heldout_pairs_by_author"] == {"author_a": 1, "author_b": 1}
    assert len(train_rows) == 4
    assert len(eval_rows) == 4


def test_hape_author_split_preserves_explicit_author_train_eval_behavior() -> None:
    train_rows, eval_rows, summary = _build_hape_rows_from_by_doc(
        by_doc=_by_doc(4),
        split_mode="author",
        hape_authors=[],
        train_authors={"author_a"},
        heldout_authors={"author_b"},
        holdout_percent=25.0,
        max_train_pairs_per_author=0,
        max_eval_pairs_per_author=0,
        seed=1729,
    )

    assert summary["train_authors"] == ["author_a"]
    assert summary["heldout_authors"] == ["author_b"]
    assert {row.source for row in train_rows} == {"hape_author_a", "hape_human_for_author_a"}
    assert {row.source for row in eval_rows} == {"hape_author_b", "hape_human_for_author_b"}
    assert len(train_rows) == 8
    assert len(eval_rows) == 8


def test_hape_doc_percent_split_validates_holdout_percent() -> None:
    with pytest.raises(ValueError, match="--hape-holdout-percent"):
        _build_hape_rows_from_by_doc(
            by_doc=_by_doc(4),
            split_mode="doc_percent",
            hape_authors=["author_a"],
            train_authors=set(),
            heldout_authors=set(),
            holdout_percent=100.0,
            max_train_pairs_per_author=0,
            max_eval_pairs_per_author=0,
            seed=1729,
        )


def test_compiled_hape_texts_frame_maps_long_table_roles_to_parallel_docs() -> None:
    frame = pd.DataFrame(
        [
            {
                "base_doc_id": "doc_1",
                "role": "prompt",
                "model": "human_prompt",
                "text": "prompt one",
            },
            {
                "base_doc_id": "doc_1",
                "role": "human_continuation",
                "model": "human",
                "text": "human one",
            },
            {
                "base_doc_id": "doc_1",
                "role": "llm_continuation",
                "model": "gpt-5.5",
                "text": "gpt one",
            },
            {
                "base_doc_id": "doc_1",
                "role": "llm_continuation",
                "model": "gemini-3.5-flash",
                "text": "gemini one",
            },
        ]
    )

    by_doc, source_summary = _hape_by_doc_from_texts_frame(frame, source_path="compiled.parquet")

    assert by_doc["doc_1"] == {
        "chunk_1": "prompt one",
        "chunk_2": "human one",
        "gemini-3.5-flash": "gemini one",
        "gpt-5.5": "gpt one",
    }
    assert source_summary["hape_source"] == "compiled_parquet"
    assert source_summary["hape_texts_parquet"] == "compiled.parquet"
    assert source_summary["compiled_llm_authors"] == ["gemini-3.5-flash", "gpt-5.5"]

    train_rows, eval_rows, split_summary = _build_hape_rows_from_by_doc(
        by_doc=by_doc,
        split_mode="doc_percent",
        hape_authors=[],
        train_authors=set(),
        heldout_authors=set(),
        holdout_percent=0.0,
        max_train_pairs_per_author=0,
        max_eval_pairs_per_author=0,
        seed=1729,
    )
    assert len(train_rows) == 4
    assert not eval_rows
    assert split_summary["train_pairs_by_author"] == {
        "gemini-3.5-flash": 1,
        "gpt-5.5": 1,
    }
