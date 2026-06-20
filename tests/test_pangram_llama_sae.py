from __future__ import annotations

from argparse import Namespace
from typing import Any

import torch
from torch import nn

from slop_sftdiv.cli.pangram_llama_sae import (
    collect_last_token_activation_cache,
    collect_last_token_activations,
)
from slop_sftdiv.cli.phase4_batchtopk_sae import SAEDoc


class _TinyTokenizer:
    pad_token = None
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token_id = 0

    def __call__(
        self,
        texts: list[str],
        *,
        padding: str | bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
        return_special_tokens_mask: bool,
    ) -> dict[str, torch.Tensor]:
        _ = (texts, padding, truncation, max_length, return_tensors, return_special_tokens_mask)
        return {
            "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])[: len(texts)],
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])[: len(texts)],
            "special_tokens_mask": torch.tensor([[1, 0, 0, 1], [1, 0, 1, 1]])[: len(texts)],
        }


class _LastHiddenStateModel(nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> Namespace:
        _ = (attention_mask, kwargs)
        early = torch.zeros((*input_ids.shape, 2), dtype=torch.float32)
        last = torch.stack([input_ids.float(), input_ids.float() + 10.0], dim=-1)
        return Namespace(hidden_states=(early, last))


def test_collect_last_token_activations_uses_final_hidden_state_and_token_mask() -> None:
    docs = [SAEDoc(source="unit", doc_id="doc-1", text="hello", label=0)]

    activations = collect_last_token_activations(
        model=_LastHiddenStateModel(),
        tokenizer=_TinyTokenizer(),
        docs=docs,
        max_length=4,
        batch_size=1,
        max_activations=10,
        device=torch.device("cpu"),
        progress_every=0,
        pad_to_max_length=True,
    )

    assert torch.equal(activations, torch.tensor([[2.0, 12.0], [3.0, 13.0]], dtype=torch.float16))


def test_collect_last_token_activation_cache_tracks_document_and_token_indexes() -> None:
    docs = [
        SAEDoc(source="unit-a", doc_id="doc-a", text="hello", label=0),
        SAEDoc(source="unit-b", doc_id="doc-b", text="world", label=1),
    ]

    cache = collect_last_token_activation_cache(
        model=_LastHiddenStateModel(),
        tokenizer=_TinyTokenizer(),
        docs=docs,
        max_length=4,
        batch_size=2,
        max_activations=3,
        device=torch.device("cpu"),
        progress_every=0,
        pad_to_max_length=True,
    )

    assert torch.equal(
        cache.activations,
        torch.tensor([[2.0, 12.0], [3.0, 13.0], [5.0, 15.0]], dtype=torch.float16),
    )
    assert cache.doc_indices.tolist() == [0, 0, 1]
    assert cache.token_indices.tolist() == [1, 2, 1]
    assert cache.document_rows == [
        {"document_index": 0, "source": "unit-a", "doc_id": "doc-a", "label": 0},
        {"document_index": 1, "source": "unit-b", "doc_id": "doc-b", "label": 1},
    ]
