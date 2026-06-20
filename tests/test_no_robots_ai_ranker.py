from __future__ import annotations

import math

import pandas as pd
import torch

from slop_sftdiv.cli.no_robots_ai_ranker import _load_prompt_pairs, _pairwise_loss


def _write_triplet_parquet(path) -> None:
    pd.DataFrame(
        [
            {
                "base_doc_id": "doc_1",
                "record_id": "doc_1_human",
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
                "record_id": "doc_2_human",
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
        ]
    ).to_parquet(path, index=False)


def test_load_prompt_pairs_groups_human_with_all_ai_continuations(tmp_path) -> None:
    path = tmp_path / "triplets.parquet"
    _write_triplet_parquet(path)

    train_rows, eval_rows, summary = _load_prompt_pairs(
        path,
        holdout_percent=0.0,
        seed=1729,
    )

    assert not eval_rows
    assert len(train_rows) == 2
    by_doc = {row.base_doc_id: row for row in train_rows}
    assert by_doc["doc_1"].human_text == "human one"
    assert [(model, text) for model, _record_id, text in by_doc["doc_1"].ai_records] == [
        ("gemini-3.5-flash", "gemini one"),
        ("gpt-5.5", "gpt one"),
    ]
    assert summary["train_prompt_count"] == 2
    assert summary["train_pair_count"] == 3
    assert summary["ai_models"] == ["gemini-3.5-flash", "gpt-5.5"]


def test_pairwise_loss_is_prompt_mean_of_model_softplus_margins() -> None:
    scores = torch.tensor(
        [
            1.0,
            3.0,
            2.0,
            -1.0,
            -2.0,
        ]
    )

    loss, metrics = _pairwise_loss(scores, [3, 2])

    expected = (
        math.log1p(math.exp(-(3.0 - 1.0)))
        + math.log1p(math.exp(-(2.0 - 1.0)))
        + math.log1p(math.exp(-(-2.0 - -1.0)))
    ) / 2.0
    assert loss.item() == torch.tensor(expected).item()
    assert metrics["pair_accuracy"] == torch.tensor(2 / 3).item()
    assert metrics["mean_margin"] == torch.tensor((2.0 + 1.0 - 1.0) / 3.0).item()
