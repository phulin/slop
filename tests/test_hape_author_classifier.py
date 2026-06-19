from __future__ import annotations

from collections import Counter

import pandas as pd

from slop_sftdiv.cli.hape_author_classifier import _load_author_rows, _wsd_lr_multiplier


def _write_compiled_hape_parquet(path) -> None:
    pd.DataFrame(
        [
            {
                "base_doc_id": "doc_1",
                "record_id": "doc_1_chunk_2",
                "role": "human_continuation",
                "model": "human",
                "text": "human one",
            },
            {
                "base_doc_id": "doc_1",
                "record_id": "doc_1_gpt",
                "role": "llm_continuation",
                "model": "gpt-5.5",
                "text": "gpt one",
            },
            {
                "base_doc_id": "doc_1",
                "record_id": "doc_1_gemini",
                "role": "llm_continuation",
                "model": "gemini-3.5-flash",
                "text": "gemini one",
            },
            {
                "base_doc_id": "doc_2",
                "record_id": "doc_2_chunk_2",
                "role": "human_continuation",
                "model": "human",
                "text": "human two",
            },
            {
                "base_doc_id": "doc_2",
                "record_id": "doc_2_gpt",
                "role": "llm_continuation",
                "model": "gpt-5.5",
                "text": "gpt two",
            },
            {
                "base_doc_id": "doc_2",
                "record_id": "doc_2_gemini",
                "role": "llm_continuation",
                "model": "gemini-3.5-flash",
                "text": "gemini two",
            },
        ]
    ).to_parquet(path, index=False)


def test_binary_label_mode_uses_flat_rows_without_repeating_humans(tmp_path) -> None:
    path = tmp_path / "compiled.parquet"
    _write_compiled_hape_parquet(path)

    train_rows, eval_rows, label_by_author, summary = _load_author_rows(
        path,
        holdout_percent=0.0,
        seed=1729,
        label_mode="binary",
    )

    assert not eval_rows
    assert len(train_rows) == 6
    assert label_by_author == {"human": 0, "ai": 1}
    assert Counter(row.author for row in train_rows) == {"human": 2, "ai": 4}
    assert Counter(row.source_model for row in train_rows) == {
        "human": 2,
        "gemini-3.5-flash": 2,
        "gpt-5.5": 2,
    }
    assert summary["label_mode"] == "binary"
    assert summary["train_examples_by_author"] == {"ai": 4, "human": 2}


def test_author_label_mode_preserves_model_classes(tmp_path) -> None:
    path = tmp_path / "compiled.parquet"
    _write_compiled_hape_parquet(path)

    train_rows, _, label_by_author, summary = _load_author_rows(
        path,
        holdout_percent=0.0,
        seed=1729,
        label_mode="author",
    )

    assert len(train_rows) == 6
    assert set(label_by_author) == {"gemini-3.5-flash", "gpt-5.5", "human"}
    assert summary["author_count"] == 3


def test_wsd_lr_multiplier_warms_plateaus_and_decays() -> None:
    kwargs = {
        "total_steps": 100,
        "warmup_steps": 10,
        "decay_steps": 20,
        "min_lr_ratio": 0.1,
    }

    assert _wsd_lr_multiplier(0, **kwargs) == 0.0
    assert _wsd_lr_multiplier(5, **kwargs) == 0.5
    assert _wsd_lr_multiplier(10, **kwargs) == 1.0
    assert _wsd_lr_multiplier(50, **kwargs) == 1.0
    assert _wsd_lr_multiplier(80, **kwargs) == 1.0
    assert _wsd_lr_multiplier(90, **kwargs) == 0.55
    assert _wsd_lr_multiplier(100, **kwargs) == 0.1
