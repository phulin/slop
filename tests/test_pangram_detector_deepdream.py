from __future__ import annotations

from argparse import Namespace

import pytest
import torch
from torch import nn

from slop_sftdiv.cli.pangram_detector_deepdream import (
    editlens_score_from_logits,
    kl_divergence_to_anchor,
    optimize_soft_completion,
    static_base_anchor_log_probs,
)


class _TinyTokenizer:
    all_special_ids = [0]
    pad_token_id = 0
    bos_token_id = 0
    eos_token_id = 0

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        return " ".join(f"tok{token_id}" for token_id in token_ids if token_id != 0)


class _TinyBaseLM(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> Namespace:
        batch, seq_len = input_ids.shape
        logits = torch.zeros((batch, seq_len, 5), dtype=torch.float32)
        for pos in range(seq_len):
            logits[:, pos, :] = torch.tensor([0.0, 1.0 + pos, 2.0 + pos, 3.0 + pos, 4.0 + pos])
        return Namespace(logits=logits)


class _TinyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(5, 2)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 2.0],
                        [0.0, -2.0],
                        [0.0, -1.0],
                    ]
                )
            )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, *, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> Namespace:
        _ = attention_mask
        score = inputs_embeds[:, 1:, 1].sum(dim=-1)
        logits = torch.stack([-score, score], dim=-1)
        return Namespace(logits=logits)


def test_static_base_anchor_uses_prompt_last_position_for_first_completion_token() -> None:
    prompt_ids = torch.tensor([[1, 2]])
    completion_ids = torch.tensor([3, 4])

    anchor = static_base_anchor_log_probs(
        model=_TinyBaseLM(),
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
    )

    assert anchor.shape == (2, 5)
    expected = torch.log_softmax(
        torch.tensor([[0.0, 2.0, 3.0, 4.0, 5.0], [0.0, 3.0, 4.0, 5.0, 6.0]]),
        dim=-1,
    )
    assert torch.allclose(anchor, expected)


def test_kl_divergence_to_anchor_matches_manual_value() -> None:
    q = torch.tensor([[0.25, 0.75]])
    anchor_log_probs = torch.log(torch.tensor([[0.5, 0.5]]))

    value = kl_divergence_to_anchor(q, anchor_log_probs)

    expected = 0.25 * torch.log(torch.tensor(0.5)) + 0.75 * torch.log(torch.tensor(1.5))
    assert value.item() == pytest.approx(float(expected))


def test_editlens_score_from_logits_is_expected_normalized_bucket() -> None:
    logits = torch.tensor([-10.0, -10.0, -10.0, 10.0])

    score = editlens_score_from_logits(logits)

    assert score.item() == pytest.approx(1.0)


def test_optimize_soft_completion_moves_toward_detector_target_token() -> None:
    result = optimize_soft_completion(
        detector_model=_TinyDetector(),
        tokenizer=_TinyTokenizer(),
        prompt_ids=torch.tensor([[1]]),
        initial_completion_ids=torch.tensor([3, 3]),
        anchor_log_probs=None,
        target_label=1,
        target_objective="logit",
        steps=35,
        learning_rate=1.0,
        softmax_temperature=1.0,
        kl_coeff=0.0,
        init_logit_bias=0.0,
        banned_token_ids={0},
        log_every=10,
    )

    assert result.token_ids == [2, 2]
    assert result.target_probability > 0.95
    assert "tok2 tok2" in result.text
