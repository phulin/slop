from __future__ import annotations

import argparse
import csv
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from slop_sftdiv.cli.pangram_detector_deepdream import (
    _decode_completion,
    _generate_initial_completion_ids,
    _load_base_lm,
    _load_pangram_detector,
    _read_prompt,
    _special_token_ids,
    _token_ids,
    editlens_score_from_logits,
)
from slop_sftdiv.cli.pangram_llama_sae import LLAMA_BASE_ID, PANGRAM_ADAPTER_ID, _device


@dataclass(frozen=True)
class CandidateScore:
    token_id: int
    token_text: str
    lm_log_probability: float
    editlens_score: float
    bucket_probabilities: list[float]
    objective: float


@dataclass(frozen=True)
class BeamState:
    token_ids: tuple[int, ...]
    cumulative_lm_log_probability: float
    editlens_score: float
    objective: float
    sequence_quality_penalty: float
    bucket_probabilities: list[float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a fixed base-Llama prefix, then autoregressively append tokens chosen "
            "to maximize Pangram/EditLens score among base-LM top-k candidates."
        )
    )
    parser.add_argument("--base-model", default=LLAMA_BASE_ID)
    parser.add_argument("--adapter-model", default=PANGRAM_ADAPTER_ID)
    parser.add_argument(
        "--local-model-dir",
        type=Path,
        default=Path("artifacts/models/pangram_editlens_Llama-3.2-3B"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--adapter-revision", default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--no-hf-token", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--prefix-tokens", type=int, default=50)
    parser.add_argument("--continuation-tokens", type=int, default=50)
    parser.add_argument("--candidate-top-k", type=int, default=64)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--beam-branch-k", type=int, default=None)
    parser.add_argument(
        "--candidate-pool-factor",
        type=int,
        default=8,
        help="Read top_k * factor LM candidates before text filtering, then keep --candidate-top-k.",
    )
    parser.add_argument(
        "--ascii-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject non-ASCII decoded candidate tokens. Useful for English-only probes.",
    )
    parser.add_argument(
        "--reject-allcaps-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject alphabetic all-caps candidate tokens with at least two letters.",
    )
    parser.add_argument(
        "--max-token-chars",
        type=int,
        default=24,
        help="Reject decoded candidate tokens whose stripped text exceeds this many characters.",
    )
    parser.add_argument(
        "--reject-symbol-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject candidate tokens containing URL/code-like symbols outside common prose punctuation.",
    )
    parser.add_argument(
        "--reject-single-uppercase-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject isolated uppercase initial tokens except I and A.",
    )
    parser.add_argument(
        "--reject-camelcase-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject candidate tokens with lower-to-upper camelCase transitions.",
    )
    parser.add_argument(
        "--reject-code-substrings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject candidate tokens containing common code, markup, or file-format substrings.",
    )
    parser.add_argument(
        "--reject-bare-subword-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject alphabetic candidate tokens without leading whitespace unless continuing a word.",
    )
    parser.add_argument(
        "--reject-punctuation-run-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject multi-character tokens made only of punctuation.",
    )
    parser.add_argument(
        "--lm-logprob-coeff",
        type=float,
        default=0.02,
        help=(
            "Candidate objective is editlens_score + coeff * base_lm_logprob. "
            "Set to 0 for pure detector-greedy search within top-k."
        ),
    )
    parser.add_argument(
        "--min-lm-logprob",
        type=float,
        default=None,
        help=(
            "Reject candidate next tokens below this base-LM log probability. "
            "If filtering removes every decoded-valid candidate, the step falls back "
            "to decoded-valid candidates without this threshold."
        ),
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=0.0,
        help="Subtract this from the objective for each previous occurrence of the candidate token.",
    )
    parser.add_argument(
        "--sequence-quality-penalty-coeff",
        type=float,
        default=0.0,
        help=(
            "Subtract coeff * heuristic prose-quality penalty from beam objectives. "
            "Useful as a full-continuation reranker after detector-guided proposals."
        ),
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--generate-temperature", type=float, default=0.8)
    parser.add_argument("--generate-top-p", type=float, default=0.95)
    parser.add_argument("--allow-special-tokens", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _topk_candidate_ids(
    *,
    logits: torch.Tensor,
    top_k: int,
    banned_token_ids: set[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 1:
        raise ValueError("next-token logits must be 1D")
    if top_k <= 0:
        raise ValueError("--candidate-top-k must be positive")
    masked = logits.float().clone()
    for token_id in banned_token_ids:
        if 0 <= token_id < masked.numel():
            masked[token_id] = -torch.inf
    effective_k = min(top_k, int(torch.isfinite(masked).sum().item()))
    if effective_k <= 0:
        raise RuntimeError("no valid candidate tokens after applying special-token mask")
    log_probs = F.log_softmax(masked, dim=-1)
    values, ids = torch.topk(log_probs, k=effective_k)
    return ids, values


CODELIKE_SUBSTRINGS = (
    ".css",
    ".html",
    ".js",
    ".json",
    ".lua",
    ".md",
    ".py",
    ".txt",
    ".xml",
    "application/",
    "argument",
    "encapsulat",
    "prefix",
    "schema",
    "twitter",
    "unref",
    "wikidata",
    "wikivocab",
    "xterm",
)


def sequence_quality_penalty(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 1.0
    lower = stripped.lower()
    nonspace_chars = [char for char in stripped if not char.isspace()]
    alpha_chars = [char for char in nonspace_chars if char.isalpha()]
    symbol_chars = [
        char
        for char in nonspace_chars
        if not char.isalnum() and char not in set(".,;:!?\"'()-")
    ]
    penalty = 0.0
    penalty += 0.55 * sum(lower.count(substring) for substring in CODELIKE_SUBSTRINGS)
    penalty += min(2.0, 8.0 * len(symbol_chars) / float(max(1, len(nonspace_chars))))
    penalty += 0.25 * len(re.findall(r"[^\w\s]{2,}", stripped))
    penalty += 0.55 * len(re.findall(r"\b\d{4}-\d{2}-\d{2}[A-Za-z]", stripped))
    penalty += 0.6 * len(re.findall(r"(?:^|\s)([.:;!?])(?:\s+\1){2,}", stripped))
    penalty += 0.25 * len(re.findall(r"\b[A-Z]{2,}\b", stripped))
    if "changelog" in lower:
        penalty += 0.6
    if re.search(r"\bnewest\s+games\s+like\b", lower):
        penalty += 0.6
    if alpha_chars and len(alpha_chars) / float(max(1, len(nonspace_chars))) < 0.55:
        penalty += 1.0

    words = re.findall(r"\b[A-Za-z]+\b", stripped)
    title_run = 0
    title_words = 0
    connectors = 0
    intensifiers = 0
    for word in words:
        is_title = len(word) > 2 and word[0].isupper() and word[1:].islower()
        if is_title:
            title_words += 1
            title_run += 1
            if title_run >= 5:
                penalty += 0.12
        else:
            title_run = 0
        lower_word = word.lower()
        if lower_word in {"and", "or", "but"}:
            connectors += 1
        if lower_word in {"even", "extremely", "often", "occasionally", "pretty", "quite", "sometimes", "very"}:
            intensifiers += 1
    if len(words) >= 8:
        title_ratio = title_words / float(len(words))
        connector_ratio = connectors / float(len(words))
        intensifier_ratio = intensifiers / float(len(words))
        if title_ratio > 0.25:
            penalty += 2.0 * (title_ratio - 0.25)
        if connector_ratio > 0.16:
            penalty += 3.0 * (connector_ratio - 0.16)
        if intensifier_ratio > 0.08:
            penalty += 4.0 * (intensifier_ratio - 0.08)
    return penalty


def _valid_decoded_token(
    text: str,
    *,
    ascii_candidates: bool,
    reject_allcaps_tokens: bool = True,
    max_token_chars: int | None = 24,
    reject_symbol_tokens: bool = True,
    reject_single_uppercase_tokens: bool = True,
    reject_camelcase_tokens: bool = True,
    reject_code_substrings: bool = True,
    reject_bare_subword_tokens: bool = True,
    allow_bare_subword_token: bool = True,
    reject_punctuation_run_tokens: bool = True,
) -> bool:
    if not text:
        return False
    if "\ufffd" in text:
        return False
    if any((ord(char) < 32 and char not in "\n\t") or ord(char) == 127 for char in text):
        return False
    if ascii_candidates and any(ord(char) > 127 for char in text):
        return False
    stripped = text.strip()
    if max_token_chars is not None and max_token_chars > 0 and len(stripped) > max_token_chars:
        return False
    if (
        reject_bare_subword_tokens
        and not allow_bare_subword_token
        and stripped
        and (substantive := stripped.lstrip(".,;:!?\"'()[]{}-"))
        and substantive[0].isalnum()
        and text[0] not in " \n\t"
    ):
        return False
    allowed_prose_punctuation = set(".,;:!?\"'()-")
    if reject_symbol_tokens and any(
        not char.isalnum() and not char.isspace() and char not in allowed_prose_punctuation
        for char in stripped
    ):
        return False
    if (
        reject_punctuation_run_tokens
        and len(stripped) > 1
        and all(not char.isalnum() and not char.isspace() for char in stripped)
    ):
        return False
    if reject_camelcase_tokens and any(
        previous.islower() and current.isupper()
        for previous, current in zip(stripped, stripped[1:], strict=False)
    ):
        return False
    if reject_code_substrings:
        lower_stripped = stripped.lower()
        if any(substring in lower_stripped for substring in CODELIKE_SUBSTRINGS):
            return False
    alpha_chars = [char for char in stripped if char.isalpha()]
    if (
        reject_single_uppercase_tokens
        and len(alpha_chars) == 1
        and alpha_chars[0].isupper()
        and alpha_chars[0] not in {"A", "I"}
    ):
        return False
    if (
        reject_allcaps_tokens
        and len(alpha_chars) >= 2
        and all(char.isupper() for char in alpha_chars)
    ):
        return False
    return True


def _filter_candidate_ids(
    *,
    tokenizer: Any,
    candidate_ids: torch.Tensor,
    lm_log_probs: torch.Tensor,
    candidate_top_k: int,
    ascii_candidates: bool,
    reject_allcaps_tokens: bool = True,
    max_token_chars: int | None = 24,
    reject_symbol_tokens: bool = True,
    reject_single_uppercase_tokens: bool = True,
    reject_camelcase_tokens: bool = True,
    reject_code_substrings: bool = True,
    reject_bare_subword_tokens: bool = True,
    allow_bare_subword_token: bool = True,
    reject_punctuation_run_tokens: bool = True,
    min_lm_logprob: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    kept_ids: list[torch.Tensor] = []
    kept_log_probs: list[torch.Tensor] = []
    fallback_ids: list[torch.Tensor] = []
    fallback_log_probs: list[torch.Tensor] = []
    for token_id, log_probability in zip(candidate_ids, lm_log_probs, strict=True):
        text = _decode_completion(tokenizer, [int(token_id.detach().cpu())])
        if not _valid_decoded_token(
            text,
            ascii_candidates=ascii_candidates,
            reject_allcaps_tokens=reject_allcaps_tokens,
            max_token_chars=max_token_chars,
            reject_symbol_tokens=reject_symbol_tokens,
            reject_single_uppercase_tokens=reject_single_uppercase_tokens,
            reject_camelcase_tokens=reject_camelcase_tokens,
            reject_code_substrings=reject_code_substrings,
            reject_bare_subword_tokens=reject_bare_subword_tokens,
            allow_bare_subword_token=allow_bare_subword_token,
            reject_punctuation_run_tokens=reject_punctuation_run_tokens,
        ):
            continue
        fallback_ids.append(token_id)
        fallback_log_probs.append(log_probability)
        if min_lm_logprob is not None and float(log_probability.detach().cpu()) < min_lm_logprob:
            continue
        kept_ids.append(token_id)
        kept_log_probs.append(log_probability)
        if len(kept_ids) >= candidate_top_k:
            break
    if not kept_ids:
        if not fallback_ids:
            raise RuntimeError("no valid candidate tokens after decoded-text filtering")
        kept_ids = fallback_ids[:candidate_top_k]
        kept_log_probs = fallback_log_probs[:candidate_top_k]
    return torch.stack(kept_ids), torch.stack(kept_log_probs)


def _context_ends_with_alnum(tokenizer: Any, context_ids: torch.Tensor) -> bool:
    context_text = _decode_completion(tokenizer, context_ids[0].detach().cpu().tolist())
    return bool(context_text and context_text[-1].isalnum())


@torch.inference_mode()
def _detector_scores_for_candidates(
    *,
    detector_model: Any,
    context_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    if context_ids.ndim != 2 or context_ids.shape[0] != 1:
        raise ValueError("context_ids must have shape (1, sequence_length)")
    rows = [
        torch.cat([context_ids[0], candidate_id.reshape(1)], dim=0)[-max_length:]
        for candidate_id in candidate_ids
    ]
    input_ids = torch.stack(rows, dim=0)
    attention_mask = torch.ones_like(input_ids)
    logits = detector_model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    return torch.stack([editlens_score_from_logits(row) for row in logits], dim=0)


@torch.inference_mode()
def _detector_scores_for_sequences(
    *,
    detector_model: Any,
    prefix_ids: torch.Tensor,
    sequence_ids: list[tuple[int, ...]],
    max_length: int,
) -> tuple[torch.Tensor, list[list[float]]]:
    if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1:
        raise ValueError("prefix_ids must have shape (1, sequence_length)")
    if not sequence_ids:
        raise ValueError("sequence_ids must not be empty")
    rows = [
        torch.cat(
            [
                prefix_ids[0],
                torch.tensor(sequence, dtype=prefix_ids.dtype, device=prefix_ids.device),
            ],
            dim=0,
        )[-max_length:]
        for sequence in sequence_ids
    ]
    input_ids = torch.stack(rows, dim=0)
    attention_mask = torch.ones_like(input_ids)
    logits = detector_model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    probabilities = F.softmax(logits, dim=-1)
    scores = torch.stack([editlens_score_from_logits(row) for row in logits], dim=0)
    probability_rows = [
        [float(value) for value in row.detach().cpu().tolist()] for row in probabilities
    ]
    return scores, probability_rows


@torch.inference_mode()
def _score_text(
    *,
    detector_model: Any,
    input_ids: torch.Tensor,
    max_length: int,
) -> tuple[float, list[float]]:
    scored_ids = input_ids[:, -max_length:]
    attention_mask = torch.ones_like(scored_ids)
    logits = detector_model(input_ids=scored_ids, attention_mask=attention_mask).logits[0].float()
    probabilities = F.softmax(logits, dim=-1)
    return (
        float(editlens_score_from_logits(logits).detach().cpu()),
        [float(value) for value in probabilities.detach().cpu().tolist()],
    )


def choose_next_token(
    *,
    base_lm: Any,
    detector_model: Any,
    tokenizer: Any,
    context_ids: torch.Tensor,
    candidate_top_k: int,
    lm_logprob_coeff: float,
    banned_token_ids: set[int],
    max_length: int,
    candidate_pool_factor: int = 8,
    ascii_candidates: bool = True,
    reject_allcaps_tokens: bool = True,
    max_token_chars: int | None = 24,
    reject_symbol_tokens: bool = True,
    reject_single_uppercase_tokens: bool = True,
    reject_camelcase_tokens: bool = True,
    reject_code_substrings: bool = True,
    reject_bare_subword_tokens: bool = True,
    reject_punctuation_run_tokens: bool = True,
    min_lm_logprob: float | None = None,
) -> CandidateScore:
    with torch.inference_mode():
        lm_logits = base_lm(input_ids=context_ids[:, -max_length:]).logits[0, -1]
    candidate_ids, lm_log_probs = _topk_candidate_ids(
        logits=lm_logits,
        top_k=max(candidate_top_k, candidate_top_k * max(1, candidate_pool_factor)),
        banned_token_ids=banned_token_ids,
    )
    candidate_ids, lm_log_probs = _filter_candidate_ids(
        tokenizer=tokenizer,
        candidate_ids=candidate_ids,
        lm_log_probs=lm_log_probs,
        candidate_top_k=candidate_top_k,
        ascii_candidates=ascii_candidates,
        reject_allcaps_tokens=reject_allcaps_tokens,
        max_token_chars=max_token_chars,
        reject_symbol_tokens=reject_symbol_tokens,
        reject_single_uppercase_tokens=reject_single_uppercase_tokens,
        reject_camelcase_tokens=reject_camelcase_tokens,
        reject_code_substrings=reject_code_substrings,
        reject_bare_subword_tokens=reject_bare_subword_tokens,
        allow_bare_subword_token=_context_ends_with_alnum(tokenizer, context_ids),
        reject_punctuation_run_tokens=reject_punctuation_run_tokens,
        min_lm_logprob=min_lm_logprob,
    )
    editlens_scores = _detector_scores_for_candidates(
        detector_model=detector_model,
        context_ids=context_ids,
        candidate_ids=candidate_ids,
        max_length=max_length,
    )
    objectives = editlens_scores + float(lm_logprob_coeff) * lm_log_probs.to(editlens_scores.device)
    best_index = int(objectives.argmax().item())
    best_token_id = int(candidate_ids[best_index].detach().cpu())
    _, bucket_probabilities = _score_text(
        detector_model=detector_model,
        input_ids=torch.cat(
            [context_ids, torch.tensor([[best_token_id]], dtype=context_ids.dtype, device=context_ids.device)],
            dim=1,
        ),
        max_length=max_length,
    )
    return CandidateScore(
        token_id=best_token_id,
        token_text=_decode_completion(tokenizer, [best_token_id]),
        lm_log_probability=float(lm_log_probs[best_index].detach().cpu()),
        editlens_score=float(editlens_scores[best_index].detach().cpu()),
        bucket_probabilities=bucket_probabilities,
        objective=float(objectives[best_index].detach().cpu()),
    )


def _candidate_choices_for_context(
    *,
    base_lm: Any,
    detector_model: Any,
    tokenizer: Any,
    context_ids: torch.Tensor,
    candidate_top_k: int,
    lm_logprob_coeff: float,
    banned_token_ids: set[int],
    max_length: int,
    candidate_pool_factor: int,
    ascii_candidates: bool,
    reject_allcaps_tokens: bool,
    max_token_chars: int | None,
    reject_symbol_tokens: bool,
    reject_single_uppercase_tokens: bool,
    reject_camelcase_tokens: bool,
    reject_code_substrings: bool,
    reject_bare_subword_tokens: bool,
    reject_punctuation_run_tokens: bool,
    min_lm_logprob: float | None,
    repetition_penalty: float,
    previous_token_ids: tuple[int, ...],
) -> list[CandidateScore]:
    with torch.inference_mode():
        lm_logits = base_lm(input_ids=context_ids[:, -max_length:]).logits[0, -1]
    candidate_ids, lm_log_probs = _topk_candidate_ids(
        logits=lm_logits,
        top_k=max(candidate_top_k, candidate_top_k * max(1, candidate_pool_factor)),
        banned_token_ids=banned_token_ids,
    )
    candidate_ids, lm_log_probs = _filter_candidate_ids(
        tokenizer=tokenizer,
        candidate_ids=candidate_ids,
        lm_log_probs=lm_log_probs,
        candidate_top_k=candidate_top_k,
        ascii_candidates=ascii_candidates,
        reject_allcaps_tokens=reject_allcaps_tokens,
        max_token_chars=max_token_chars,
        reject_symbol_tokens=reject_symbol_tokens,
        reject_single_uppercase_tokens=reject_single_uppercase_tokens,
        reject_camelcase_tokens=reject_camelcase_tokens,
        reject_code_substrings=reject_code_substrings,
        reject_bare_subword_tokens=reject_bare_subword_tokens,
        allow_bare_subword_token=_context_ends_with_alnum(tokenizer, context_ids),
        reject_punctuation_run_tokens=reject_punctuation_run_tokens,
        min_lm_logprob=min_lm_logprob,
    )
    editlens_scores, probability_rows = _detector_scores_for_sequences(
        detector_model=detector_model,
        prefix_ids=context_ids,
        sequence_ids=[(int(token_id.detach().cpu()),) for token_id in candidate_ids],
        max_length=max_length,
    )
    token_counts: dict[int, int] = {}
    for token_id in previous_token_ids:
        token_counts[token_id] = token_counts.get(token_id, 0) + 1
    choices: list[CandidateScore] = []
    for index, token_id_tensor in enumerate(candidate_ids):
        token_id = int(token_id_tensor.detach().cpu())
        repeat_cost = float(repetition_penalty) * float(token_counts.get(token_id, 0))
        objective = (
            float(editlens_scores[index].detach().cpu())
            + float(lm_logprob_coeff) * float(lm_log_probs[index].detach().cpu())
            - repeat_cost
        )
        choices.append(
            CandidateScore(
                token_id=token_id,
                token_text=_decode_completion(tokenizer, [token_id]),
                lm_log_probability=float(lm_log_probs[index].detach().cpu()),
                editlens_score=float(editlens_scores[index].detach().cpu()),
                bucket_probabilities=probability_rows[index],
                objective=objective,
            )
        )
    return sorted(choices, key=lambda choice: choice.objective, reverse=True)


def beam_search_continuation(
    *,
    base_lm: Any,
    detector_model: Any,
    tokenizer: Any,
    prefix_ids: torch.Tensor,
    continuation_tokens: int,
    beam_size: int,
    beam_branch_k: int,
    candidate_top_k: int,
    candidate_pool_factor: int,
    lm_logprob_coeff: float,
    banned_token_ids: set[int],
    max_length: int,
    ascii_candidates: bool,
    reject_allcaps_tokens: bool,
    max_token_chars: int | None,
    reject_symbol_tokens: bool,
    reject_single_uppercase_tokens: bool,
    reject_camelcase_tokens: bool,
    reject_code_substrings: bool,
    reject_bare_subword_tokens: bool,
    reject_punctuation_run_tokens: bool,
    min_lm_logprob: float | None,
    repetition_penalty: float,
    sequence_quality_penalty_coeff: float,
    progress_every: int,
) -> tuple[BeamState, list[dict[str, Any]], list[BeamState]]:
    if beam_size <= 0:
        raise ValueError("--beam-size must be positive")
    if beam_branch_k <= 0:
        raise ValueError("--beam-branch-k must be positive")
    beams = [
        BeamState(
            token_ids=(),
            cumulative_lm_log_probability=0.0,
            editlens_score=0.0,
            objective=0.0,
            sequence_quality_penalty=0.0,
            bucket_probabilities=[],
        )
    ]
    rows: list[dict[str, Any]] = []
    for step in range(1, continuation_tokens + 1):
        expanded: list[BeamState] = []
        for beam in beams:
            context_ids = prefix_ids
            if beam.token_ids:
                context_ids = torch.cat(
                    [
                        prefix_ids,
                        torch.tensor(
                            [list(beam.token_ids)],
                            dtype=prefix_ids.dtype,
                            device=prefix_ids.device,
                        ),
                    ],
                    dim=1,
                )
            choices = _candidate_choices_for_context(
                base_lm=base_lm,
                detector_model=detector_model,
                tokenizer=tokenizer,
                context_ids=context_ids,
                candidate_top_k=candidate_top_k,
                lm_logprob_coeff=lm_logprob_coeff,
                banned_token_ids=banned_token_ids,
                max_length=max_length,
                candidate_pool_factor=candidate_pool_factor,
                ascii_candidates=ascii_candidates,
                reject_allcaps_tokens=reject_allcaps_tokens,
                max_token_chars=max_token_chars,
                reject_symbol_tokens=reject_symbol_tokens,
                reject_single_uppercase_tokens=reject_single_uppercase_tokens,
                reject_camelcase_tokens=reject_camelcase_tokens,
                reject_code_substrings=reject_code_substrings,
                reject_bare_subword_tokens=reject_bare_subword_tokens,
                reject_punctuation_run_tokens=reject_punctuation_run_tokens,
                min_lm_logprob=min_lm_logprob,
                repetition_penalty=repetition_penalty,
                previous_token_ids=beam.token_ids,
            )[:beam_branch_k]
            for choice in choices:
                token_ids = (*beam.token_ids, choice.token_id)
                cumulative_lm_log_probability = (
                    beam.cumulative_lm_log_probability + choice.lm_log_probability
                )
                continuation_text = _decode_completion(tokenizer, list(token_ids))
                quality_penalty = sequence_quality_penalty(continuation_text)
                objective = (
                    choice.editlens_score
                    + float(lm_logprob_coeff) * cumulative_lm_log_probability / float(len(token_ids))
                    - float(repetition_penalty)
                    * float(len(token_ids) - len(set(token_ids)))
                    / float(len(token_ids))
                    - float(sequence_quality_penalty_coeff) * quality_penalty
                )
                expanded.append(
                    BeamState(
                        token_ids=token_ids,
                        cumulative_lm_log_probability=cumulative_lm_log_probability,
                        editlens_score=choice.editlens_score,
                        objective=objective,
                        sequence_quality_penalty=quality_penalty,
                        bucket_probabilities=choice.bucket_probabilities,
                    )
                )
        beams = sorted(expanded, key=lambda beam: beam.objective, reverse=True)[:beam_size]
        best = beams[0]
        row = {
            "step": step,
            "beam_rank": 0,
            "editlens_score": best.editlens_score,
            "objective": best.objective,
            "sequence_quality_penalty": best.sequence_quality_penalty,
            "mean_lm_log_probability": best.cumulative_lm_log_probability
            / float(max(1, len(best.token_ids))),
            "bucket_probabilities_json": json.dumps(best.bucket_probabilities),
            "generated_continuation": _decode_completion(tokenizer, list(best.token_ids)),
        }
        rows.append(row)
        if progress_every > 0 and (step == 1 or step % progress_every == 0):
            last_token = _decode_completion(tokenizer, [best.token_ids[-1]])
            print(
                f"step={step} score={best.editlens_score:.4f} "
                f"obj={best.objective:.4f} qpen={best.sequence_quality_penalty:.3f} "
                f"token={last_token!r}",
                flush=True,
            )
    return beams[0], rows, beams


def _beam_result_row(tokenizer: Any, beam: BeamState, *, beam_rank: int) -> dict[str, Any]:
    token_ids = list(beam.token_ids)
    return {
        "beam_rank": beam_rank,
        "editlens_score": beam.editlens_score,
        "objective": beam.objective,
        "sequence_quality_penalty": beam.sequence_quality_penalty,
        "mean_lm_log_probability": beam.cumulative_lm_log_probability
        / float(max(1, len(token_ids))),
        "bucket_probabilities_json": json.dumps(beam.bucket_probabilities),
        "continuation": _decode_completion(tokenizer, token_ids),
        "token_ids_json": json.dumps(token_ids),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.prefix_tokens <= 0:
        raise ValueError("--prefix-tokens must be positive")
    if args.continuation_tokens <= 0:
        raise ValueError("--continuation-tokens must be positive")
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt = _read_prompt(args)
    device = _device(args.device)
    tokenizer = _load_pangram_tokenizer_for_ar(args)
    prompt_ids = _token_ids(tokenizer, prompt, add_special_tokens=True, device=device)
    base_lm = _load_base_lm(args, device=device)
    initial_ids = _generate_initial_completion_ids(
        model=base_lm,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        completion_tokens=args.prefix_tokens,
        temperature=args.generate_temperature,
        top_p=args.generate_top_p,
    )
    detector_model, _ = _load_pangram_detector(args, device=device)
    banned_token_ids = set() if args.allow_special_tokens else _special_token_ids(tokenizer)
    context_ids = torch.cat([prompt_ids, initial_ids.unsqueeze(0)], dim=1)
    initial_score, initial_bucket_probabilities = _score_text(
        detector_model=detector_model,
        input_ids=context_ids,
        max_length=args.max_length,
    )

    beam_branch_k = args.beam_branch_k if args.beam_branch_k is not None else args.candidate_top_k
    best_beam, rows, final_beams = beam_search_continuation(
        base_lm=base_lm,
        detector_model=detector_model,
        tokenizer=tokenizer,
        prefix_ids=context_ids,
        continuation_tokens=args.continuation_tokens,
        beam_size=args.beam_size,
        beam_branch_k=beam_branch_k,
        candidate_top_k=args.candidate_top_k,
        candidate_pool_factor=args.candidate_pool_factor,
        lm_logprob_coeff=args.lm_logprob_coeff,
        banned_token_ids=banned_token_ids,
        max_length=args.max_length,
        ascii_candidates=args.ascii_candidates,
        reject_allcaps_tokens=args.reject_allcaps_tokens,
        max_token_chars=args.max_token_chars,
        reject_symbol_tokens=args.reject_symbol_tokens,
        reject_single_uppercase_tokens=args.reject_single_uppercase_tokens,
        reject_camelcase_tokens=args.reject_camelcase_tokens,
        reject_code_substrings=args.reject_code_substrings,
        reject_bare_subword_tokens=args.reject_bare_subword_tokens,
        reject_punctuation_run_tokens=args.reject_punctuation_run_tokens,
        min_lm_logprob=args.min_lm_logprob,
        repetition_penalty=args.repetition_penalty,
        sequence_quality_penalty_coeff=args.sequence_quality_penalty_coeff,
        progress_every=args.progress_every,
    )
    generated_ids = list(best_beam.token_ids)
    if generated_ids:
        context_ids = torch.cat(
            [
                context_ids,
                torch.tensor([generated_ids], dtype=context_ids.dtype, device=context_ids.device),
            ],
            dim=1,
        )

    final_score, final_bucket_probabilities = _score_text(
        detector_model=detector_model,
        input_ids=context_ids,
        max_length=args.max_length,
    )
    summary = {
        "experiment": "pangram_detector_ar_search",
        "base_model": args.base_model,
        "adapter_model": args.adapter_model,
        "local_model_dir": str(args.local_model_dir),
        "prompt": prompt,
        "initial_prefix": _decode_completion(tokenizer, initial_ids.detach().cpu().tolist()),
        "optimized_continuation": _decode_completion(tokenizer, generated_ids),
        "full_text": _decode_completion(tokenizer, context_ids[0].detach().cpu().tolist()),
        "initial_editlens_score": initial_score,
        "initial_bucket_probabilities": initial_bucket_probabilities,
        "final_editlens_score": final_score,
        "final_bucket_probabilities": final_bucket_probabilities,
        "prefix_tokens": args.prefix_tokens,
        "continuation_tokens": args.continuation_tokens,
        "candidate_top_k": args.candidate_top_k,
        "candidate_pool_factor": args.candidate_pool_factor,
        "beam_size": args.beam_size,
        "beam_branch_k": beam_branch_k,
        "ascii_candidates": bool(args.ascii_candidates),
        "reject_allcaps_tokens": bool(args.reject_allcaps_tokens),
        "max_token_chars": args.max_token_chars,
        "reject_symbol_tokens": bool(args.reject_symbol_tokens),
        "reject_single_uppercase_tokens": bool(args.reject_single_uppercase_tokens),
        "reject_camelcase_tokens": bool(args.reject_camelcase_tokens),
        "reject_code_substrings": bool(args.reject_code_substrings),
        "reject_bare_subword_tokens": bool(args.reject_bare_subword_tokens),
        "reject_punctuation_run_tokens": bool(args.reject_punctuation_run_tokens),
        "lm_logprob_coeff": args.lm_logprob_coeff,
        "min_lm_logprob": args.min_lm_logprob,
        "repetition_penalty": args.repetition_penalty,
        "sequence_quality_penalty_coeff": args.sequence_quality_penalty_coeff,
        "final_sequence_quality_penalty": best_beam.sequence_quality_penalty,
        "final_objective": best_beam.objective,
        "final_mean_lm_log_probability": best_beam.cumulative_lm_log_probability
        / float(max(1, len(best_beam.token_ids))),
        "max_length": args.max_length,
        "seed": args.seed,
        "generated_token_ids": generated_ids,
    }
    _write_json(args.output_dir / "pangram_detector_ar_search_summary.json", summary)
    _write_csv(args.output_dir / "pangram_detector_ar_search_steps.csv", rows)
    _write_csv(
        args.output_dir / "pangram_detector_ar_search_final_beams.csv",
        [
            _beam_result_row(tokenizer, beam, beam_rank=beam_rank)
            for beam_rank, beam in enumerate(final_beams)
        ],
    )
    print(f"Wrote Pangram detector AR-search outputs to {args.output_dir}", flush=True)
    del base_lm, detector_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _load_pangram_tokenizer_for_ar(args: argparse.Namespace) -> Any:
    from slop_sftdiv.cli.pangram_llama_sae import load_pangram_tokenizer

    return load_pangram_tokenizer(args)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
