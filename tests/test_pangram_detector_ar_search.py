from __future__ import annotations

from argparse import Namespace

import torch
from torch import nn

from slop_sftdiv.cli.pangram_detector_ar_search import (
    _beam_result_row,
    _topk_candidate_ids,
    _valid_decoded_token,
    beam_search_continuation,
    choose_next_token,
    sequence_quality_penalty,
)


class _TinyTokenizer:
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        return " ".join(f"tok{token_id}" for token_id in token_ids)


class _TinyBaseLM(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> Namespace:
        _ = input_ids
        logits = torch.tensor([[[0.0, 4.0, 3.0, 2.0, 1.0]]], dtype=torch.float32)
        return Namespace(logits=logits)


class _TinyDetector(nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Namespace:
        _ = attention_mask
        last = input_ids[:, -1]
        score = (last == 2).float() * 8.0 + (last == 1).float() * 2.0
        logits = torch.stack([-score, torch.zeros_like(score), score], dim=-1)
        return Namespace(logits=logits)


def test_topk_candidate_ids_masks_banned_tokens() -> None:
    ids, _ = _topk_candidate_ids(
        logits=torch.tensor([10.0, 9.0, 8.0]),
        top_k=2,
        banned_token_ids={0},
    )

    assert ids.tolist() == [1, 2]


def test_valid_decoded_token_rejects_allcaps_and_overlong_tokens() -> None:
    assert not _valid_decoded_token(" REALLY", ascii_candidates=True)
    assert not _valid_decoded_token(" QU", ascii_candidates=True)
    assert _valid_decoded_token(" Really", ascii_candidates=True)
    assert _valid_decoded_token(" I", ascii_candidates=True)
    assert not _valid_decoded_token(" ordinary-but-far-too-long", ascii_candidates=True, max_token_chars=8)
    assert not _valid_decoded_token("%20schema", ascii_candidates=True)
    assert not _valid_decoded_token("+%", ascii_candidates=True)
    assert not _valid_decoded_token("[", ascii_candidates=True)
    assert not _valid_decoded_token("]", ascii_candidates=True)
    assert not _valid_decoded_token("...", ascii_candidates=True)
    assert not _valid_decoded_token(" S", ascii_candidates=True)
    assert _valid_decoded_token(" A", ascii_candidates=True)
    assert not _valid_decoded_token(" XTerm", ascii_candidates=True)
    assert not _valid_decoded_token(".lua", ascii_candidates=True)
    assert not _valid_decoded_token(" argument", ascii_candidates=True)
    assert not _valid_decoded_token("prefix", ascii_candidates=True, allow_bare_subword_token=False)
    assert not _valid_decoded_token("-import", ascii_candidates=True, allow_bare_subword_token=False)
    assert not _valid_decoded_token(":title", ascii_candidates=True, allow_bare_subword_token=False)
    assert not _valid_decoded_token("201", ascii_candidates=True, allow_bare_subword_token=False)
    assert _valid_decoded_token("est", ascii_candidates=True, allow_bare_subword_token=True)
    assert _valid_decoded_token(" hello", ascii_candidates=True, allow_bare_subword_token=False)


def test_sequence_quality_penalty_flags_markup_and_punctuation_artifacts() -> None:
    prose = "I would appreciate your guidance. This has been an intense learning process."
    markup = "prefix=twitter%20schema%3CA%20property%3Div:application/x-wikidata"
    punctuation_loop = ". . . . . . . . : : : : :"
    title_run = "Advanced Programming Basic Utility Math Open Source Public Domain"
    title_list = (
        "Advanced Programming in the Interactive, Networked, and Web-based Visuals, "
        "Physical, and Mathematical Behaviors of Everyday, Commonly Uninteresting "
        "and Often Ordinary, but Sometimes Very Interesting and Even Quite Important"
    )
    seo_title = "2020-11-07Newest Games Like God Of War. All the latest and updated Games to play."
    changelog = "Changelog: [v1.0] - added character and fixed collision detection."

    assert sequence_quality_penalty(prose) < 0.1
    assert sequence_quality_penalty(markup) > sequence_quality_penalty(prose) + 1.0
    assert sequence_quality_penalty(punctuation_loop) > sequence_quality_penalty(prose) + 1.0
    assert sequence_quality_penalty(title_run) > sequence_quality_penalty(prose)
    assert sequence_quality_penalty(title_list) > sequence_quality_penalty(prose) + 0.5
    assert sequence_quality_penalty(seo_title) > sequence_quality_penalty(prose) + 0.5
    assert sequence_quality_penalty(changelog) > sequence_quality_penalty(prose) + 0.5


def test_choose_next_token_can_trade_lm_probability_for_detector_score() -> None:
    choice = choose_next_token(
        base_lm=_TinyBaseLM(),
        detector_model=_TinyDetector(),
        tokenizer=_TinyTokenizer(),
        context_ids=torch.tensor([[4]]),
        candidate_top_k=3,
        lm_logprob_coeff=0.0,
        banned_token_ids=set(),
        max_length=16,
    )

    assert choice.token_id == 2
    assert choice.editlens_score > 0.9


def test_beam_search_returns_best_detector_favored_continuation() -> None:
    best, rows, final_beams = beam_search_continuation(
        base_lm=_TinyBaseLM(),
        detector_model=_TinyDetector(),
        tokenizer=_TinyTokenizer(),
        prefix_ids=torch.tensor([[4]]),
        continuation_tokens=2,
        beam_size=2,
        beam_branch_k=2,
        candidate_top_k=3,
        candidate_pool_factor=1,
        lm_logprob_coeff=0.0,
        banned_token_ids=set(),
        max_length=16,
        ascii_candidates=False,
        min_lm_logprob=None,
        repetition_penalty=0.0,
        sequence_quality_penalty_coeff=0.0,
        progress_every=0,
        reject_allcaps_tokens=True,
        max_token_chars=24,
        reject_symbol_tokens=True,
        reject_single_uppercase_tokens=True,
        reject_camelcase_tokens=True,
        reject_code_substrings=True,
        reject_bare_subword_tokens=True,
        reject_punctuation_run_tokens=True,
    )

    assert len(rows) == 2
    assert len(final_beams) == 2
    assert best.token_ids[-1] == 2
    assert best.editlens_score > 0.9
    result_row = _beam_result_row(_TinyTokenizer(), best, beam_rank=0)
    assert result_row["beam_rank"] == 0
    assert result_row["continuation"].endswith("tok2")
