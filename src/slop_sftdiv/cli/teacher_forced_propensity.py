from __future__ import annotations

import argparse
import copy
import csv
import os
import resource
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from slop_sftdiv.cli.census import _id_fields_with_pair_id, _measurement_texts, _source_from_input
from slop_sftdiv.propensity import PHASE2_OPPORTUNITY_SPECS, iter_feature_opportunities
from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


OUTPUT_COLUMNS = [
    "source",
    "record_id",
    "role",
    "feature",
    "opportunity_kind",
    "char_offset",
    "prefix_tokens",
    "reference_initiates",
    "matched_subtype",
    "prob_mass",
    "initiator_sequences",
]


@dataclass
class FeatureAccumulator:
    opportunities: int = 0
    reference_initiations: int = 0
    prob_mass_sum: float = 0.0

    @property
    def reference_rate(self) -> float:
        return self.reference_initiations / self.opportunities if self.opportunities else 0.0

    @property
    def mean_prob_mass(self) -> float:
        return self.prob_mass_sum / self.opportunities if self.opportunities else 0.0

    @property
    def amplification_factor(self) -> float:
        if self.reference_rate <= 0.0:
            return 0.0
        return self.mean_prob_mass / self.reference_rate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 2 teacher-forced propensity smoke.")
    parser.add_argument("--model", required=True, help="Hugging Face causal LM ID or local path.")
    parser.add_argument("--input", action="append", required=True, help="Local JSONL file or HF dataset id.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--max-scan", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--max-opportunities", type=int, default=512)
    parser.add_argument("--max-token-start-opportunities", type=int, default=128)
    parser.add_argument("--max-prefix-tokens", type=int, default=1024)
    parser.add_argument(
        "--mass-mode",
        default="sequence",
        choices=["sequence", "first_token"],
        help="Use exact multi-token continuation mass or smoke-only first-token mass.",
    )
    parser.add_argument(
        "--fixed-prefix-tokens",
        type=int,
        default=256,
        help="Left-pad prefixes to this length for stable compiled shapes; set 0 for dynamic.",
    )
    parser.add_argument(
        "--opportunity-batch-size",
        type=int,
        default=16,
        help="Number of same-feature opportunities to score per batched model call; set 1 for scalar.",
    )
    parser.add_argument(
        "--sequence-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use KV-cache continuations for exact sequence mass.",
    )
    parser.add_argument(
        "--cache-branch-batch-size",
        type=int,
        default=4,
        help="Maximum unique continuation branches per cached forward; lower to reduce VRAM.",
    )
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase2-propensity")
    parser.add_argument("--wandb-job-type", default="teacher-forced-smoke")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _load_corpus_components():
    from slop_sftdiv.data import SamplingConfig, iter_corpus_records
    from slop_sftdiv.data.corpora import DEFAULT_ID_FIELDS

    return SamplingConfig, iter_corpus_records, DEFAULT_ID_FIELDS


def _load_model(*, model_name: str, dtype_name: str, device_name: str, compile_model: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _dtype(dtype_name, torch)
    device = _device(device_name, torch)
    if device.type == "cuda":
        _ensure_cuda_library_path()
        torch.set_float32_matmul_precision("high")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_kwargs: dict[str, Any] = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()
    if compile_model:
        model = torch.compile(model)
    return tokenizer, model, device


def _ensure_cuda_library_path() -> None:
    candidates = (
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
    )
    existing = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(":") if item]
    additions = [
        item
        for item in candidates
        if Path(item, "libcuda.so.1").exists() and item not in existing
    ]
    if additions:
        os.environ["LD_LIBRARY_PATH"] = ":".join([*additions, *existing])


def _dtype(dtype_name: str, torch_module: Any):
    if dtype_name == "auto":
        return None
    return {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }[dtype_name]


def _device(device_name: str, torch_module: Any):
    if device_name != "auto":
        return torch_module.device(device_name)
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _initiator_token_sequences(tokenizer: Any, feature: str) -> tuple[tuple[int, ...], ...]:
    spec = PHASE2_OPPORTUNITY_SPECS[feature]
    sequences: set[tuple[int, ...]] = set()
    for phrase in spec.initiators:
        for variant in (phrase, " " + phrase):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if ids:
                sequences.add(tuple(int(item) for item in ids[:3]))
    return tuple(sorted(sequences))


def _first_token_ids(sequences: tuple[tuple[int, ...], ...]) -> set[int]:
    return {sequence[0] for sequence in sequences if sequence}


def _prefix_input_ids(
    tokenizer: Any,
    text: str,
    char_offset: int,
    *,
    max_prefix_tokens: int,
    fixed_prefix_tokens: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    ids, attention_mask, prefix_tokens = _prefix_ids_and_mask(
        tokenizer,
        text,
        char_offset,
        max_prefix_tokens=max_prefix_tokens,
        fixed_prefix_tokens=fixed_prefix_tokens,
    )
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long, device=device),
        "prefix_tokens": prefix_tokens,
    }


def _prefix_ids_and_mask(
    tokenizer: Any,
    text: str,
    char_offset: int,
    *,
    max_prefix_tokens: int,
    fixed_prefix_tokens: int,
) -> tuple[list[int], list[int], int]:
    prefix = text[:char_offset]
    ids = tokenizer.encode(prefix, add_special_tokens=False)
    if ids:
        ids = ids[-max_prefix_tokens:]
    else:
        fallback_id = tokenizer.bos_token_id
        if fallback_id is None:
            fallback_id = tokenizer.eos_token_id
        if fallback_id is None:
            fallback_id = tokenizer.pad_token_id
        if fallback_id is None:
            raise ValueError("tokenizer has no BOS/EOS/PAD token for empty-prefix probability")
        ids = [fallback_id]
    attention_mask = [1] * len(ids)
    if fixed_prefix_tokens > 0:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = tokenizer.bos_token_id
        if pad_id is None:
            raise ValueError("tokenizer has no PAD/EOS/BOS token for fixed-prefix padding")
        ids = ids[-fixed_prefix_tokens:]
        attention_mask = attention_mask[-fixed_prefix_tokens:]
        pad_count = max(0, fixed_prefix_tokens - len(ids))
        ids = [pad_id] * pad_count + ids
        attention_mask = [0] * pad_count + attention_mask
    return ids, attention_mask, sum(attention_mask)


def _prefix_input_batch(
    tokenizer: Any,
    text: str,
    opportunities: list[Any],
    *,
    max_prefix_tokens: int,
    fixed_prefix_tokens: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    items = [
        _prefix_ids_and_mask(
            tokenizer,
            text,
            opportunity.char_offset,
            max_prefix_tokens=max_prefix_tokens,
            fixed_prefix_tokens=fixed_prefix_tokens,
        )
        for opportunity in opportunities
    ]
    max_length = max(len(ids) for ids, _mask, _prefix_tokens in items)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = tokenizer.bos_token_id
    if pad_id is None:
        raise ValueError("tokenizer has no PAD/EOS/BOS token for batch padding")
    input_rows = []
    mask_rows = []
    prefix_tokens = []
    for ids, attention_mask, prefix_token_count in items:
        pad_count = max_length - len(ids)
        input_rows.append([pad_id] * pad_count + ids)
        mask_rows.append([0] * pad_count + attention_mask)
        prefix_tokens.append(prefix_token_count)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long, device=device),
        "prefix_tokens": prefix_tokens,
    }


def _first_token_prob_mass(model: Any, model_inputs: dict[str, Any], initiator_ids: set[int]) -> float:
    import torch

    if not initiator_ids:
        return 0.0
    with torch.inference_mode():
        logits = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
        ).logits[0, -1]
        probabilities = torch.softmax(logits, dim=-1)
        index = torch.tensor(sorted(initiator_ids), dtype=torch.long, device=probabilities.device)
        return float(probabilities.index_select(0, index).sum().item())


def _first_token_prob_mass_batch(
    model: Any,
    model_inputs: dict[str, Any],
    initiator_ids: set[int],
) -> list[float]:
    import torch

    if not initiator_ids:
        return [0.0 for _row in range(int(model_inputs["input_ids"].shape[0]))]
    with torch.inference_mode():
        logits = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
        ).logits[:, -1]
        probabilities = torch.softmax(logits, dim=-1)
        index = torch.tensor(sorted(initiator_ids), dtype=torch.long, device=probabilities.device)
        return [float(item) for item in probabilities.index_select(1, index).sum(dim=1).detach().cpu()]


def _sequence_prob_mass(
    model: Any,
    model_inputs: dict[str, Any],
    sequences: tuple[tuple[int, ...], ...],
) -> float:
    import torch

    if not sequences:
        return 0.0
    max_length = max(len(sequence) for sequence in sequences)
    with torch.inference_mode():
        logits = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
        ).logits[0, -1]
        probabilities = torch.softmax(logits, dim=-1)
        first_token_ids = torch.tensor(
            [sequence[0] for sequence in sequences],
            dtype=torch.long,
            device=probabilities.device,
        )
        sequence_probabilities = probabilities.index_select(0, first_token_ids)

        for depth in range(1, max_length):
            active_indexes = [index for index, sequence in enumerate(sequences) if depth < len(sequence)]
            if not active_indexes:
                continue
            branch_prefixes = sorted({sequences[index][:depth] for index in active_indexes})
            branch_index = {branch: index for index, branch in enumerate(branch_prefixes)}
            input_ids = model_inputs["input_ids"].repeat(len(branch_prefixes), 1)
            attention_mask = model_inputs["attention_mask"].repeat(len(branch_prefixes), 1)
            prefix_tokens = torch.tensor(
                branch_prefixes,
                dtype=torch.long,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, prefix_tokens], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(prefix_tokens)], dim=-1)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1]
            probabilities = torch.softmax(logits, dim=-1)
            active_tensor = torch.tensor(
                active_indexes,
                dtype=torch.long,
                device=sequence_probabilities.device,
            )
            branch_rows = torch.tensor(
                [branch_index[sequences[index][:depth]] for index in active_indexes],
                dtype=torch.long,
                device=probabilities.device,
            )
            target_ids = torch.tensor(
                [sequences[index][depth] for index in active_indexes],
                dtype=torch.long,
                device=probabilities.device,
            )
            token_probabilities = probabilities[branch_rows, target_ids]
            sequence_probabilities[active_tensor] *= token_probabilities
        return float(sequence_probabilities.sum().item())


def _sequence_prob_mass_batch(
    model: Any,
    model_inputs: dict[str, Any],
    sequences: tuple[tuple[int, ...], ...],
) -> list[float]:
    import torch

    batch_size = int(model_inputs["input_ids"].shape[0])
    if not sequences:
        return [0.0 for _row in range(batch_size)]
    max_length = max(len(sequence) for sequence in sequences)
    with torch.inference_mode():
        logits = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
        ).logits[:, -1]
        probabilities = torch.softmax(logits, dim=-1)
        first_token_ids = torch.tensor(
            [sequence[0] for sequence in sequences],
            dtype=torch.long,
            device=probabilities.device,
        )
        sequence_probabilities = probabilities.index_select(1, first_token_ids)

        for depth in range(1, max_length):
            active_sequence_indexes = [
                index for index, sequence in enumerate(sequences) if depth < len(sequence)
            ]
            if not active_sequence_indexes:
                continue
            branch_prefixes = sorted({sequences[index][:depth] for index in active_sequence_indexes})
            branch_index = {branch: index for index, branch in enumerate(branch_prefixes)}
            branch_count = len(branch_prefixes)
            input_ids = model_inputs["input_ids"].repeat_interleave(branch_count, dim=0)
            attention_mask = model_inputs["attention_mask"].repeat_interleave(branch_count, dim=0)
            prefix_tokens = torch.tensor(
                branch_prefixes,
                dtype=torch.long,
                device=input_ids.device,
            ).repeat(batch_size, 1)
            input_ids = torch.cat([input_ids, prefix_tokens], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(prefix_tokens)], dim=-1)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1]
            probabilities = torch.softmax(logits, dim=-1).reshape(batch_size, branch_count, -1)
            active_tensor = torch.tensor(
                active_sequence_indexes,
                dtype=torch.long,
                device=sequence_probabilities.device,
            )
            branch_rows = torch.tensor(
                [branch_index[sequences[index][:depth]] for index in active_sequence_indexes],
                dtype=torch.long,
                device=probabilities.device,
            )
            target_ids = torch.tensor(
                [sequences[index][depth] for index in active_sequence_indexes],
                dtype=torch.long,
                device=probabilities.device,
            )
            token_probabilities = probabilities[:, branch_rows, target_ids]
            sequence_probabilities[:, active_tensor] *= token_probabilities
        return [float(item) for item in sequence_probabilities.sum(dim=1).detach().cpu()]


def _sequence_prob_mass_batch_cached(
    model: Any,
    model_inputs: dict[str, Any],
    sequences: tuple[tuple[int, ...], ...],
    *,
    cache_branch_batch_size: int,
) -> list[float]:
    return _sequence_prob_mass_batch_cached_multi(
        model,
        model_inputs,
        {"_feature": sequences},
        cache_branch_batch_size=cache_branch_batch_size,
    )["_feature"]


def _sequence_prob_mass_batch_cached_multi(
    model: Any,
    model_inputs: dict[str, Any],
    sequences_by_feature: dict[str, tuple[tuple[int, ...], ...]],
    *,
    cache_branch_batch_size: int,
) -> dict[str, list[float]]:
    import torch

    batch_size = int(model_inputs["input_ids"].shape[0])
    if cache_branch_batch_size < 1:
        raise ValueError("cache_branch_batch_size must be positive")
    feature_names = sorted(sequences_by_feature)
    feature_sequence_indexes: dict[str, list[int]] = {}
    all_sequences = sorted(
        {
            sequence
            for sequences in sequences_by_feature.values()
            for sequence in sequences
            if sequence
        }
    )
    sequence_index = {sequence: index for index, sequence in enumerate(all_sequences)}
    for feature, sequences in sequences_by_feature.items():
        feature_sequence_indexes[feature] = [
            sequence_index[sequence] for sequence in sequences if sequence in sequence_index
        ]
    if not all_sequences:
        return {feature: [0.0 for _row in range(batch_size)] for feature in feature_names}
    max_length = max(len(sequence) for sequence in all_sequences)
    with torch.inference_mode():
        outputs = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            use_cache=True,
        )
        probabilities = torch.softmax(outputs.logits[:, -1], dim=-1)
        first_token_ids = torch.tensor(
            [sequence[0] for sequence in all_sequences],
            dtype=torch.long,
            device=probabilities.device,
        )
        sequence_probabilities = probabilities.index_select(1, first_token_ids)
        past_key_values = outputs.past_key_values

        for depth in range(1, max_length):
            active_sequence_indexes = [
                index for index, sequence in enumerate(all_sequences) if depth < len(sequence)
            ]
            if not active_sequence_indexes:
                continue
            branch_prefixes = sorted(
                {all_sequences[index][:depth] for index in active_sequence_indexes}
            )
            for branch_start in range(0, len(branch_prefixes), cache_branch_batch_size):
                branch_chunk = branch_prefixes[branch_start : branch_start + cache_branch_batch_size]
                branch_index = {branch: index for index, branch in enumerate(branch_chunk)}
                chunk_active_indexes = [
                    index
                    for index in active_sequence_indexes
                    if all_sequences[index][:depth] in branch_index
                ]
                branch_count = len(branch_chunk)
                branch_cache = copy.deepcopy(past_key_values)
                branch_cache.batch_repeat_interleave(branch_count)
                branch_tokens = torch.tensor(
                    branch_chunk,
                    dtype=torch.long,
                    device=model_inputs["input_ids"].device,
                ).repeat(batch_size, 1)
                branch_attention = torch.cat(
                    [
                        model_inputs["attention_mask"].repeat_interleave(branch_count, dim=0),
                        torch.ones_like(branch_tokens),
                    ],
                    dim=-1,
                )
                logits = model(
                    input_ids=branch_tokens,
                    attention_mask=branch_attention,
                    past_key_values=branch_cache,
                    use_cache=True,
                ).logits[:, -1]
                probabilities = torch.softmax(logits, dim=-1).reshape(batch_size, branch_count, -1)
                active_tensor = torch.tensor(
                    chunk_active_indexes,
                    dtype=torch.long,
                    device=sequence_probabilities.device,
                )
                branch_rows = torch.tensor(
                    [branch_index[all_sequences[index][:depth]] for index in chunk_active_indexes],
                    dtype=torch.long,
                    device=probabilities.device,
                )
                target_ids = torch.tensor(
                    [all_sequences[index][depth] for index in chunk_active_indexes],
                    dtype=torch.long,
                    device=probabilities.device,
                )
                sequence_probabilities[:, active_tensor] *= probabilities[:, branch_rows, target_ids]
        masses_by_feature: dict[str, list[float]] = {}
        for feature in feature_names:
            indexes = feature_sequence_indexes[feature]
            if indexes:
                index_tensor = torch.tensor(
                    indexes,
                    dtype=torch.long,
                    device=sequence_probabilities.device,
                )
                masses = sequence_probabilities.index_select(1, index_tensor).sum(dim=1)
                masses_by_feature[feature] = [float(item) for item in masses.detach().cpu()]
            else:
                masses_by_feature[feature] = [0.0 for _row in range(batch_size)]
        return masses_by_feature


def _score_opportunities_cached_multi(
    *,
    tokenizer: Any,
    model: Any,
    device: Any,
    text: str,
    opportunities: list[Any],
    initiator_sequences: dict[str, tuple[tuple[int, ...], ...]],
    max_prefix_tokens: int,
    fixed_prefix_tokens: int,
    opportunity_batch_size: int,
    cache_branch_batch_size: int,
) -> tuple[list[float], list[int]]:
    prob_masses = [0.0 for _opportunity in opportunities]
    prefix_token_counts = [0 for _opportunity in opportunities]
    indexes_by_offset: dict[int, list[int]] = defaultdict(list)
    for index, opportunity in enumerate(opportunities):
        indexes_by_offset[opportunity.char_offset].append(index)

    offsets = sorted(indexes_by_offset)
    for start in range(0, len(offsets), opportunity_batch_size):
        chunk_offsets = offsets[start : start + opportunity_batch_size]
        representative_opportunities = [
            opportunities[indexes_by_offset[offset][0]] for offset in chunk_offsets
        ]
        model_inputs = _prefix_input_batch(
            tokenizer,
            text,
            representative_opportunities,
            max_prefix_tokens=max_prefix_tokens,
            fixed_prefix_tokens=fixed_prefix_tokens,
            device=device,
        )
        features = sorted(
            {
                opportunities[index].feature
                for offset in chunk_offsets
                for index in indexes_by_offset[offset]
            }
        )
        masses_by_feature = _sequence_prob_mass_batch_cached_multi(
            model,
            model_inputs,
            {feature: initiator_sequences.get(feature, ()) for feature in features},
            cache_branch_batch_size=cache_branch_batch_size,
        )
        for row_index, offset in enumerate(chunk_offsets):
            for opportunity_index in indexes_by_offset[offset]:
                feature = opportunities[opportunity_index].feature
                prob_masses[opportunity_index] = masses_by_feature[feature][row_index]
                prefix_token_counts[opportunity_index] = int(model_inputs["prefix_tokens"][row_index])
    return prob_masses, prefix_token_counts


def _score_opportunities(
    *,
    tokenizer: Any,
    model: Any,
    device: Any,
    text: str,
    opportunities: list[Any],
    initiator_sequences: dict[str, tuple[tuple[int, ...], ...]],
    mass_mode: str,
    max_prefix_tokens: int,
    fixed_prefix_tokens: int,
    opportunity_batch_size: int,
    sequence_cache: bool,
    cache_branch_batch_size: int,
) -> tuple[list[float], list[int]]:
    if mass_mode == "sequence" and sequence_cache:
        return _score_opportunities_cached_multi(
            tokenizer=tokenizer,
            model=model,
            device=device,
            text=text,
            opportunities=opportunities,
            initiator_sequences=initiator_sequences,
            max_prefix_tokens=max_prefix_tokens,
            fixed_prefix_tokens=fixed_prefix_tokens,
            opportunity_batch_size=opportunity_batch_size,
            cache_branch_batch_size=cache_branch_batch_size,
        )

    prob_masses = [0.0 for _opportunity in opportunities]
    prefix_token_counts = [0 for _opportunity in opportunities]
    if opportunity_batch_size == 1:
        for index, opportunity in enumerate(opportunities):
            sequences = initiator_sequences.get(opportunity.feature, ())
            model_inputs = _prefix_input_ids(
                tokenizer,
                text,
                opportunity.char_offset,
                max_prefix_tokens=max_prefix_tokens,
                fixed_prefix_tokens=fixed_prefix_tokens,
                device=device,
            )
            if mass_mode == "first_token":
                prob_masses[index] = _first_token_prob_mass(
                    model,
                    model_inputs,
                    _first_token_ids(sequences),
                )
            elif sequence_cache:
                prob_masses[index] = _sequence_prob_mass_batch_cached(
                    model,
                    model_inputs,
                    sequences,
                    cache_branch_batch_size=cache_branch_batch_size,
                )[0]
            else:
                prob_masses[index] = _sequence_prob_mass(model, model_inputs, sequences)
            prefix_token_counts[index] = int(model_inputs["prefix_tokens"])
        return prob_masses, prefix_token_counts

    indexes_by_feature: dict[str, list[int]] = defaultdict(list)
    for index, opportunity in enumerate(opportunities):
        indexes_by_feature[opportunity.feature].append(index)
    for feature, feature_indexes in indexes_by_feature.items():
        sequences = initiator_sequences.get(feature, ())
        for start in range(0, len(feature_indexes), opportunity_batch_size):
            chunk_indexes = feature_indexes[start : start + opportunity_batch_size]
            chunk_opportunities = [opportunities[index] for index in chunk_indexes]
            model_inputs = _prefix_input_batch(
                tokenizer,
                text,
                chunk_opportunities,
                max_prefix_tokens=max_prefix_tokens,
                fixed_prefix_tokens=fixed_prefix_tokens,
                device=device,
            )
            if mass_mode == "first_token":
                chunk_masses = _first_token_prob_mass_batch(
                    model,
                    model_inputs,
                    _first_token_ids(sequences),
                )
            elif sequence_cache:
                chunk_masses = _sequence_prob_mass_batch_cached(
                    model,
                    model_inputs,
                    sequences,
                    cache_branch_batch_size=cache_branch_batch_size,
                )
            else:
                chunk_masses = _sequence_prob_mass_batch(model, model_inputs, sequences)
            for index, prob_mass, prefix_tokens in zip(
                chunk_indexes,
                chunk_masses,
                model_inputs["prefix_tokens"],
            ):
                prob_masses[index] = prob_mass
                prefix_token_counts[index] = int(prefix_tokens)
    return prob_masses, prefix_token_counts


def _summary_rows(accumulators: dict[str, FeatureAccumulator]) -> list[dict[str, Any]]:
    rows = []
    for feature, accumulator in sorted(accumulators.items()):
        rows.append(
            {
                "feature": feature,
                "opportunities": accumulator.opportunities,
                "reference_initiations": accumulator.reference_initiations,
                "reference_rate": accumulator.reference_rate,
                "mean_prob_mass": accumulator.mean_prob_mass,
                "amplification_factor": accumulator.amplification_factor,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_teacher_forced_propensity(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.opportunity_batch_size < 1:
        raise ValueError("opportunity_batch_size must be positive")
    if args.cache_branch_batch_size < 1:
        raise ValueError("cache_branch_batch_size must be positive")
    SamplingConfig, iter_corpus_records, DEFAULT_ID_FIELDS = _load_corpus_components()
    tokenizer, model, device = _load_model(
        model_name=args.model,
        dtype_name=args.dtype,
        device_name=args.device,
        compile_model=args.torch_compile,
    )
    selected_features = list(dict.fromkeys(args.feature or ["contrastive_negation"]))
    initiator_sequences = {
        feature: _initiator_token_sequences(tokenizer, feature)
        for feature in selected_features
        if feature in PHASE2_OPPORTUNITY_SPECS
    }
    text_fields = [args.text_field] if args.text_field else None
    sampling = SamplingConfig(cap=args.sample_size, seed=args.seed, max_scan=args.max_scan)
    id_fields = _id_fields_with_pair_id(DEFAULT_ID_FIELDS)

    rows: list[dict[str, Any]] = []
    accumulators: dict[str, FeatureAccumulator] = defaultdict(FeatureAccumulator)
    doc_count = 0
    measurement_rows = 0
    started_at = time.perf_counter()

    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage2", "phase2", "teacher-forced", "smoke", *args.wandb_tag],
        config={
            "model": args.model,
            "inputs": args.input,
            "split": args.split,
            "hf_config": args.hf_config,
            "sample_size": args.sample_size,
            "max_scan": args.max_scan,
            "features": selected_features,
            "max_opportunities": args.max_opportunities,
            "max_token_start_opportunities": args.max_token_start_opportunities,
            "max_prefix_tokens": args.max_prefix_tokens,
            "mass_mode": args.mass_mode,
            "fixed_prefix_tokens": args.fixed_prefix_tokens,
            "opportunity_batch_size": args.opportunity_batch_size,
            "sequence_cache": args.sequence_cache,
            "cache_branch_batch_size": args.cache_branch_batch_size,
            "dtype": args.dtype,
            "device": str(device),
            "torch_compile": args.torch_compile,
            "output": str(args.output),
        },
    )

    try:
        for raw_source in args.input:
            source = _source_from_input(raw_source, split=args.split, hf_config=args.hf_config)
            records_kwargs = {"sampling": sampling, "id_fields": id_fields}
            if text_fields:
                records_kwargs["text_fields"] = text_fields
            for record in tqdm(
                iter_corpus_records(source, **records_kwargs),
                desc=f"teacher-forced:{source.name}",
                unit="doc",
            ):
                doc_count += 1
                for role, text, _pair_id in _measurement_texts(record):
                    measurement_rows += 1
                    opportunities = iter_feature_opportunities(
                        text,
                        features=selected_features,
                        max_token_start_opportunities=args.max_token_start_opportunities,
                    )[: args.max_opportunities]
                    prob_masses, prefix_token_counts = _score_opportunities(
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        text=text,
                        opportunities=opportunities,
                        initiator_sequences=initiator_sequences,
                        mass_mode=args.mass_mode,
                        max_prefix_tokens=args.max_prefix_tokens,
                        fixed_prefix_tokens=args.fixed_prefix_tokens,
                        opportunity_batch_size=args.opportunity_batch_size,
                        sequence_cache=args.sequence_cache,
                        cache_branch_batch_size=args.cache_branch_batch_size,
                    )
                    for opportunity, prob_mass, prefix_tokens in zip(
                        opportunities,
                        prob_masses,
                        prefix_token_counts,
                    ):
                        accumulator = accumulators[opportunity.feature]
                        accumulator.opportunities += 1
                        accumulator.reference_initiations += int(opportunity.reference_initiates)
                        accumulator.prob_mass_sum += prob_mass
                        rows.append(
                            {
                                "source": record.source,
                                "record_id": record.record_id,
                                "role": role,
                                "feature": opportunity.feature,
                                "opportunity_kind": opportunity.opportunity_kind,
                                "char_offset": opportunity.char_offset,
                                "prefix_tokens": prefix_tokens,
                                "reference_initiates": int(opportunity.reference_initiates),
                                "matched_subtype": opportunity.matched_subtype or "",
                                "prob_mass": prob_mass,
                                "initiator_sequences": len(
                                    initiator_sequences.get(opportunity.feature, ())
                                ),
                            }
                        )

        _write_csv(args.output, rows, OUTPUT_COLUMNS)
        summary_rows = _summary_rows(accumulators)
        if args.summary_output is not None:
            _write_csv(
                args.summary_output,
                summary_rows,
                [
                    "feature",
                    "opportunities",
                    "reference_initiations",
                    "reference_rate",
                    "mean_prob_mass",
                    "amplification_factor",
                ],
            )

        wall_seconds = time.perf_counter() - started_at
        run.log(
            {
                "propensity/docs": doc_count,
                "propensity/measurement_rows": measurement_rows,
                "propensity/opportunities": len(rows),
                "propensity/opportunities_per_sec": len(rows) / wall_seconds
                if wall_seconds > 0
                else 0.0,
                "propensity/wall_seconds": wall_seconds,
                "propensity/peak_rss_mb": _peak_rss_mb(),
            }
        )
        log_summary_table(run, "propensity_summary", summary_rows)
        return summary_rows
    finally:
        run.finish()


def main() -> None:
    args = build_parser().parse_args()
    summary_rows = run_teacher_forced_propensity(args)
    print(f"Wrote {len(summary_rows)} feature summaries to {args.summary_output or args.output}")


if __name__ == "__main__":
    main()
