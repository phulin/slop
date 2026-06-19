from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from slop_sftdiv.cli.hape_augment_batch import (
    OPENAI_ACTIVE_STATUSES,
    OPENAI_TERMINAL_STATUSES,
    _anthropic_headers,
    _anthropic_text_from_output,
    _download_anthropic_results,
    _download_gemini_file,
    _download_openai_file,
    _gemini_done,
    _gemini_headers,
    _gemini_response_file,
    _gemini_state,
    _gemini_text_from_output,
    _load_provider_env,
    _openai_headers,
    _openai_text_from_output,
    _request_json,
    _safe_model_name,
    _write_json,
    _write_jsonl,
)


DEFAULT_DATA_DIR = Path("data/external/no_robots")
DEFAULT_OUTPUT_DIR = Path("artifacts/no_robots_batch_generations")
DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_FIREWORKS_MODEL = "accounts/fireworks/models/glm-5p2"
DEFAULT_OPENAI_REASONING_EFFORT = "none"
DEFAULT_GEMINI_THINKING_LEVEL = "minimal"
DEFAULT_ANTHROPIC_THINKING = "disabled"
DEFAULT_FIREWORKS_REASONING_EFFORT = "none"
DEFAULT_GEMINI_SHARD_TOKEN_BUDGET = 2_400_000
DEFAULT_MIN_OUTPUT_TOKENS = 128
DEFAULT_MAX_OUTPUT_TOKENS = 2048
NO_ROBOTS_DATASET = "HuggingFaceH4/no_robots"

ANTHROPIC_TERMINAL_STATUSES = {"ended"}
FIREWORKS_TERMINAL_STATES = {
    "JOB_STATE_COMPLETED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_EARLY_STOPPED",
    "JOB_STATE_DELETED",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and manage batch generations for every no_robots assistant turn."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    prepare.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    prepare.add_argument("--anthropic-model", default=DEFAULT_ANTHROPIC_MODEL)
    prepare.add_argument("--fireworks-model", default=DEFAULT_FIREWORKS_MODEL)
    prepare.add_argument("--openai-reasoning-effort", default=DEFAULT_OPENAI_REASONING_EFFORT)
    prepare.add_argument("--gemini-thinking-level", default=DEFAULT_GEMINI_THINKING_LEVEL)
    prepare.add_argument("--anthropic-thinking", choices=["disabled"], default=DEFAULT_ANTHROPIC_THINKING)
    prepare.add_argument("--fireworks-reasoning-effort", default=DEFAULT_FIREWORKS_REASONING_EFFORT)
    prepare.add_argument("--min-output-tokens", type=int, default=DEFAULT_MIN_OUTPUT_TOKENS)
    prepare.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    prepare.add_argument("--max-rows-per-split", type=int, default=None)
    prepare.add_argument(
        "--gemini-shard-token-budget",
        type=int,
        default=DEFAULT_GEMINI_SHARD_TOKEN_BUDGET,
        help=(
            "Conservative estimated input+reserved-output token budget per Gemini "
            "batch shard. Paid Tier 1 Gemini 3.5 Flash has a 3,000,000 enqueued-token limit."
        ),
    )
    prepare.set_defaults(func=prepare_no_robots_batches)

    submit_openai = subparsers.add_parser("submit-openai")
    submit_openai.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    submit_openai.add_argument("--max-open-batches", type=int, default=2)
    submit_openai.add_argument("--force", action="store_true")
    submit_openai.set_defaults(func=submit_openai_batch)

    submit_gemini = subparsers.add_parser("submit-gemini")
    submit_gemini.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    submit_gemini.add_argument("--force", action="store_true")
    submit_gemini.set_defaults(func=submit_gemini_batch)

    submit_anthropic = subparsers.add_parser("submit-anthropic")
    submit_anthropic.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    submit_anthropic.add_argument("--max-active-batch-requests", type=int, default=200_000)
    submit_anthropic.add_argument("--force", action="store_true")
    submit_anthropic.set_defaults(func=submit_anthropic_batch)

    submit_fireworks = subparsers.add_parser("submit-fireworks")
    submit_fireworks.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    submit_fireworks.add_argument("--account-id", default=None)
    submit_fireworks.add_argument("--force", action="store_true")
    submit_fireworks.set_defaults(func=submit_fireworks_batch)

    status = subparsers.add_parser("status")
    status.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    status.add_argument("--fireworks-account-id", default=None)
    status.set_defaults(func=status_batches)

    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    fetch.add_argument("--fireworks-account-id", default=None)
    fetch.set_defaults(func=fetch_completed_batches)

    compile_categorical = subparsers.add_parser("compile-categorical")
    compile_categorical.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    compile_categorical.add_argument(
        "--dataset-output-dir",
        type=Path,
        default=None,
        help="Output directory for the compiled HAP-E-style categorical parquet.",
    )
    compile_categorical.add_argument(
        "--provider",
        action="append",
        default=None,
        help="Generated provider to include. Repeatable. Defaults to openai and gemini.",
    )
    compile_categorical.add_argument(
        "--max-triplet-token-length",
        type=int,
        default=0,
        help=(
            "If positive, keep only complete human+provider triplets where every "
            "version is at or below this token length with --tokenizer-model."
        ),
    )
    compile_categorical.add_argument(
        "--tokenizer-model",
        default="artifacts/phase4/modernbert_detector_combined_v2_clean/checkpoint",
    )
    compile_categorical.set_defaults(func=compile_categorical_dataset)

    drive_gemini = subparsers.add_parser("drive-gemini")
    drive_gemini.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    drive_gemini.add_argument("--poll-seconds", type=int, default=300)
    drive_gemini.add_argument("--max-runtime-seconds", type=int, default=172800)
    drive_gemini.set_defaults(func=drive_gemini_batches)

    drive_anthropic = subparsers.add_parser("drive-anthropic")
    drive_anthropic.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    drive_anthropic.add_argument("--poll-seconds", type=int, default=300)
    drive_anthropic.add_argument("--max-runtime-seconds", type=int, default=172800)
    drive_anthropic.set_defaults(func=drive_anthropic_batch)

    drive_fireworks = subparsers.add_parser("drive-fireworks")
    drive_fireworks.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    drive_fireworks.add_argument("--fireworks-account-id", default=None)
    drive_fireworks.add_argument("--poll-seconds", type=int, default=300)
    drive_fireworks.add_argument("--max-runtime-seconds", type=int, default=172800)
    drive_fireworks.set_defaults(func=drive_fireworks_batch)

    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
    return rows


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, dict):
                pieces.append(str(part.get("text") or part.get("content") or ""))
            else:
                pieces.append(str(part))
        return "\n".join(piece for piece in pieces if piece)
    if content is None:
        return ""
    return str(content)


def _normalized_message(message: dict[str, Any]) -> dict[str, str]:
    role = str(message.get("role") or "").strip().lower()
    if role not in {"system", "user", "assistant"}:
        raise ValueError(f"unsupported no_robots message role: {role!r}")
    return {"role": role, "content": _message_text(message)}


def _target_max_output_tokens(
    target_text: str,
    *,
    min_output_tokens: int,
    max_output_tokens: int,
) -> int:
    if min_output_tokens <= 0:
        raise ValueError("--min-output-tokens must be positive")
    if max_output_tokens < min_output_tokens:
        raise ValueError("--max-output-tokens must be >= --min-output-tokens")
    word_count = len(target_text.split())
    estimated = int(word_count * 1.6) + 96
    return max(min_output_tokens, min(max_output_tokens, estimated))


def _non_system_roles(messages: list[dict[str, str]]) -> list[str]:
    return [message["role"] for message in messages if message["role"] != "system"]


def _can_use_native_messages(messages: list[dict[str, str]]) -> bool:
    roles = _non_system_roles(messages)
    if not roles or roles[0] != "user" or roles[-1] != "user":
        return False
    expected = "user"
    for role in roles:
        if role != expected:
            return False
        expected = "assistant" if expected == "user" else "user"
    return True


def _render_transcript_prompt(messages: list[dict[str, str]]) -> str:
    lines = [
        "The following is a conversation transcript. Write the next assistant message only.",
        "Do not include labels, markdown fences, commentary, or alternatives.",
        "",
        "Transcript:",
    ]
    role_names = {"system": "System", "user": "User", "assistant": "Assistant"}
    for message in messages:
        lines.append(f"{role_names[message['role']]}: {message['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _messages_json(messages: list[dict[str, str]]) -> str:
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":"))


def _assistant_turn_rows_for_frame(
    frame: pd.DataFrame,
    *,
    split: str,
    min_output_tokens: int,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_row_index, row in frame.reset_index(drop=True).iterrows():
        messages = [_normalized_message(message) for message in row["messages"]]
        assistant_ordinal = 0
        for message_index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            assistant_ordinal += 1
            input_messages = messages[:message_index]
            target_text = message["content"]
            native_messages = _can_use_native_messages(input_messages)
            turn_key = (
                f"{split}:{source_row_index}:{row['prompt_id']}:"
                f"{message_index}:{assistant_ordinal}"
            )
            turn_id = f"nr_{split}_{source_row_index:05d}_{assistant_ordinal:02d}_{_short_hash(turn_key)}"
            rows.append(
                {
                    "turn_id": turn_id,
                    "source_dataset": NO_ROBOTS_DATASET,
                    "split": split,
                    "source_row_index": int(source_row_index),
                    "prompt_id": str(row["prompt_id"]),
                    "category": str(row["category"]),
                    "assistant_message_index": int(message_index),
                    "assistant_turn_ordinal": int(assistant_ordinal),
                    "input_message_count": int(len(input_messages)),
                    "input_messages_json": _messages_json(input_messages),
                    "target_text": target_text,
                    "target_word_count": len(target_text.split()),
                    "target_char_count": len(target_text),
                    "target_sha256": hashlib.sha256(target_text.encode("utf-8")).hexdigest(),
                    "max_output_tokens": _target_max_output_tokens(
                        target_text,
                        min_output_tokens=min_output_tokens,
                        max_output_tokens=max_output_tokens,
                    ),
                    "uses_transcript_prompt": not native_messages,
                }
            )
    return rows


def _load_assistant_turn_rows(
    data_dir: Path,
    *,
    max_rows_per_split: int | None,
    min_output_tokens: int,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["train", "test"]:
        path = data_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        frame = pd.read_parquet(path)
        if max_rows_per_split is not None:
            frame = frame.head(max_rows_per_split)
        rows.extend(
            _assistant_turn_rows_for_frame(
                frame,
                split=split,
                min_output_tokens=min_output_tokens,
                max_output_tokens=max_output_tokens,
            )
        )
    return rows


def _turn_input_messages(turn: dict[str, Any]) -> list[dict[str, str]]:
    return json.loads(str(turn["input_messages_json"]))


def _turn_prompt_or_messages(turn: dict[str, Any]) -> tuple[bool, list[dict[str, str]] | str]:
    messages = _turn_input_messages(turn)
    if bool(turn.get("uses_transcript_prompt")):
        return True, _render_transcript_prompt(messages)
    return False, messages


def _provider_manifest_rows(
    turn_rows: list[dict[str, Any]],
    *,
    provider: str,
    model: str,
    openai_reasoning_effort: str | None = None,
    gemini_thinking_level: str | None = None,
    anthropic_thinking: str | None = None,
    fireworks_reasoning_effort: str | None = None,
) -> list[dict[str, Any]]:
    safe_model = _safe_model_name(model)
    rows = []
    for turn in turn_rows:
        if provider == "openai":
            custom_id = f"{provider}_{safe_model}_{turn['turn_id']}"
        else:
            split_letter = str(turn["split"])[0]
            provider_prefix = {
                "anthropic": "an",
                "gemini": "ge",
                "fireworks": "fw",
            }.get(provider, provider[:2])
            custom_id = (
                f"nr_{provider_prefix}_{split_letter}{int(turn['source_row_index']):05d}_"
                f"{int(turn['assistant_turn_ordinal']):02d}_{_short_hash(str(turn['turn_id']))}"
            )
        rows.append(
            {
                "custom_id": custom_id,
                "turn_id": turn["turn_id"],
                "provider": provider,
                "model": model,
                "split": turn["split"],
                "source_row_index": turn["source_row_index"],
                "prompt_id": turn["prompt_id"],
                "category": turn["category"],
                "assistant_message_index": turn["assistant_message_index"],
                "assistant_turn_ordinal": turn["assistant_turn_ordinal"],
                "max_output_tokens": turn["max_output_tokens"],
                "uses_transcript_prompt": turn["uses_transcript_prompt"],
                "openai_reasoning_effort": openai_reasoning_effort,
                "gemini_thinking_level": gemini_thinking_level,
                "anthropic_thinking": anthropic_thinking,
                "fireworks_reasoning_effort": fireworks_reasoning_effort,
            }
        )
    return rows


def _openai_request_rows(
    turn_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for turn, manifest in zip(turn_rows, manifest_rows, strict=True):
        is_prompt, payload = _turn_prompt_or_messages(turn)
        body: dict[str, Any] = {
            "model": manifest["model"],
            "input": payload if is_prompt else payload,
            "max_output_tokens": int(manifest["max_output_tokens"]),
            "reasoning": {
                "effort": manifest.get("openai_reasoning_effort") or DEFAULT_OPENAI_REASONING_EFFORT
            },
            "store": False,
        }
        rows.append(
            {
                "custom_id": manifest["custom_id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
        )
    return rows


def _split_system_messages(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    system_texts = [message["content"] for message in messages if message["role"] == "system"]
    non_system = [message for message in messages if message["role"] != "system"]
    system_text = "\n\n".join(text for text in system_texts if text).strip()
    return system_text or None, non_system


def _gemini_contents(messages: list[dict[str, str]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    system_text, non_system = _split_system_messages(messages)
    system_instruction = (
        {"parts": [{"text": system_text}]}
        if system_text
        else None
    )
    contents = []
    for message in non_system:
        contents.append(
            {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [{"text": message["content"]}],
            }
        )
    return system_instruction, contents


def _gemini_request_rows(
    turn_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for turn, manifest in zip(turn_rows, manifest_rows, strict=True):
        is_prompt, payload = _turn_prompt_or_messages(turn)
        if is_prompt:
            request: dict[str, Any] = {
                "contents": [{"role": "user", "parts": [{"text": str(payload)}]}],
            }
        else:
            system_instruction, contents = _gemini_contents(payload)  # type: ignore[arg-type]
            request = {"contents": contents}
            if system_instruction:
                request["systemInstruction"] = system_instruction
        request["generationConfig"] = {
            "maxOutputTokens": int(manifest["max_output_tokens"]),
            "thinkingConfig": {
                "thinkingLevel": manifest.get("gemini_thinking_level")
                or DEFAULT_GEMINI_THINKING_LEVEL
            },
        }
        rows.append({"key": manifest["custom_id"], "request": request})
    return rows


def _anthropic_messages(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    system_text, non_system = _split_system_messages(messages)
    return system_text, [
        {"role": message["role"], "content": message["content"]}
        for message in non_system
    ]


def _anthropic_request_rows(
    turn_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for turn, manifest in zip(turn_rows, manifest_rows, strict=True):
        thinking = manifest.get("anthropic_thinking") or DEFAULT_ANTHROPIC_THINKING
        if thinking != "disabled":
            raise ValueError("Anthropic no_robots generation only supports disabled thinking")
        is_prompt, payload = _turn_prompt_or_messages(turn)
        if is_prompt:
            system_text = None
            messages = [{"role": "user", "content": str(payload)}]
        else:
            system_text, messages = _anthropic_messages(payload)  # type: ignore[arg-type]
        params: dict[str, Any] = {
            "model": manifest["model"],
            "max_tokens": int(manifest["max_output_tokens"]),
            "messages": messages,
        }
        if system_text:
            params["system"] = system_text
        rows.append({"custom_id": manifest["custom_id"], "params": params})
    return rows


def _fireworks_request_rows(
    turn_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for turn, manifest in zip(turn_rows, manifest_rows, strict=True):
        is_prompt, payload = _turn_prompt_or_messages(turn)
        if is_prompt:
            messages = [{"role": "user", "content": str(payload)}]
        else:
            messages = payload  # type: ignore[assignment]
        body = {
            "messages": messages,
            "max_tokens": int(manifest["max_output_tokens"]),
            "reasoning_effort": (
                manifest.get("fireworks_reasoning_effort") or DEFAULT_FIREWORKS_REASONING_EFFORT
            ),
        }
        rows.append({"custom_id": manifest["custom_id"], "body": body})
    return rows


def _estimated_enqueued_tokens(turn: dict[str, Any]) -> int:
    messages = _turn_input_messages(turn)
    if bool(turn.get("uses_transcript_prompt")):
        prompt_text = _render_transcript_prompt(messages)
    else:
        prompt_text = "\n".join(message["content"] for message in messages)
    return int(len(prompt_text) / 3) + int(turn["max_output_tokens"]) + (16 * len(messages))


def _write_gemini_shards(
    *,
    output_dir: Path,
    turn_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    token_budget: int,
) -> list[dict[str, Any]]:
    if token_budget <= 0:
        raise ValueError("--gemini-shard-token-budget must be positive")
    shard_dir = output_dir / "requests" / "gemini_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    request_rows = _gemini_request_rows(turn_rows, manifest_rows)
    shard_summaries: list[dict[str, Any]] = []
    current_rows: list[dict[str, Any]] = []
    current_tokens = 0
    current_start_index = 0

    def flush(end_index: int) -> None:
        nonlocal current_rows, current_tokens, current_start_index
        if not current_rows:
            return
        shard_id = len(shard_summaries)
        path = shard_dir / f"gemini_requests_{shard_id:04d}.jsonl"
        _write_jsonl(path, current_rows)
        shard_summaries.append(
            {
                "shard_id": shard_id,
                "path": str(path),
                "request_count": len(current_rows),
                "estimated_enqueued_tokens": current_tokens,
                "start_index": current_start_index,
                "end_index_exclusive": end_index,
            }
        )
        current_rows = []
        current_tokens = 0
        current_start_index = end_index

    for index, (turn, request_row) in enumerate(zip(turn_rows, request_rows, strict=True)):
        estimated_tokens = _estimated_enqueued_tokens(turn)
        if current_rows and current_tokens + estimated_tokens > token_budget:
            flush(index)
        current_rows.append(request_row)
        current_tokens += estimated_tokens
    flush(len(request_rows))
    return shard_summaries


def prepare_no_robots_batches(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    requests_dir = output_dir / "requests"
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_dir.mkdir(parents=True, exist_ok=True)

    turn_rows = _load_assistant_turn_rows(
        args.data_dir,
        max_rows_per_split=args.max_rows_per_split,
        min_output_tokens=args.min_output_tokens,
        max_output_tokens=args.max_output_tokens,
    )
    openai_manifest = _provider_manifest_rows(
        turn_rows,
        provider="openai",
        model=args.openai_model,
        openai_reasoning_effort=args.openai_reasoning_effort,
    )
    gemini_manifest = _provider_manifest_rows(
        turn_rows,
        provider="gemini",
        model=args.gemini_model,
        gemini_thinking_level=args.gemini_thinking_level,
    )
    anthropic_manifest = _provider_manifest_rows(
        turn_rows,
        provider="anthropic",
        model=args.anthropic_model,
        anthropic_thinking=args.anthropic_thinking,
    )
    fireworks_manifest = _provider_manifest_rows(
        turn_rows,
        provider="fireworks",
        model=args.fireworks_model,
        fireworks_reasoning_effort=args.fireworks_reasoning_effort,
    )
    request_manifest = [
        *openai_manifest,
        *gemini_manifest,
        *anthropic_manifest,
        *fireworks_manifest,
    ]

    pd.DataFrame(turn_rows).to_parquet(output_dir / "assistant_turns.parquet", index=False)
    pd.DataFrame(request_manifest).to_parquet(output_dir / "request_manifest.parquet", index=False)
    _write_jsonl(requests_dir / "openai_requests.jsonl", _openai_request_rows(turn_rows, openai_manifest))
    _write_jsonl(requests_dir / "gemini_requests.jsonl", _gemini_request_rows(turn_rows, gemini_manifest))
    _write_jsonl(
        requests_dir / "anthropic_requests.jsonl",
        _anthropic_request_rows(turn_rows, anthropic_manifest),
    )
    _write_jsonl(
        requests_dir / "fireworks_requests.jsonl",
        _fireworks_request_rows(turn_rows, fireworks_manifest),
    )
    gemini_shards = _write_gemini_shards(
        output_dir=output_dir,
        turn_rows=turn_rows,
        manifest_rows=gemini_manifest,
        token_budget=args.gemini_shard_token_budget,
    )
    pd.DataFrame(gemini_shards).to_parquet(output_dir / "gemini_shard_manifest.parquet", index=False)

    turn_frame = pd.DataFrame(turn_rows)
    provider_counts = Counter(row["provider"] for row in request_manifest)
    summary = {
        "created_at": int(time.time()),
        "source_dataset": NO_ROBOTS_DATASET,
        "data_dir": str(args.data_dir),
        "assistant_turns": len(turn_rows),
        "provider_request_counts": dict(sorted(provider_counts.items())),
        "total_provider_requests": len(request_manifest),
        "fallback_transcript_prompt_turns": int(turn_frame["uses_transcript_prompt"].sum()),
        "max_output_tokens_min": int(turn_frame["max_output_tokens"].min()),
        "max_output_tokens_max": int(turn_frame["max_output_tokens"].max()),
        "max_output_tokens_mean": float(turn_frame["max_output_tokens"].mean()),
        "openai_model": args.openai_model,
        "openai_reasoning_effort": args.openai_reasoning_effort,
        "gemini_model": args.gemini_model,
        "gemini_thinking_level": args.gemini_thinking_level,
        "gemini_shard_token_budget": args.gemini_shard_token_budget,
        "gemini_shards": len(gemini_shards),
        "gemini_shard_max_estimated_enqueued_tokens": max(
            row["estimated_enqueued_tokens"] for row in gemini_shards
        )
        if gemini_shards
        else 0,
        "anthropic_model": args.anthropic_model,
        "anthropic_thinking": args.anthropic_thinking,
        "fireworks_model": args.fireworks_model,
        "fireworks_reasoning_effort": args.fireworks_reasoning_effort,
        "outputs": {
            "assistant_turns": str(output_dir / "assistant_turns.parquet"),
            "request_manifest": str(output_dir / "request_manifest.parquet"),
            "openai_requests": str(requests_dir / "openai_requests.jsonl"),
            "gemini_requests": str(requests_dir / "gemini_requests.jsonl"),
            "anthropic_requests": str(requests_dir / "anthropic_requests.jsonl"),
            "fireworks_requests": str(requests_dir / "fireworks_requests.jsonl"),
            "gemini_shard_manifest": str(output_dir / "gemini_shard_manifest.parquet"),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    print(
        f"Wrote {len(turn_rows)} no_robots assistant-turn requests per provider "
        f"({len(request_manifest)} total provider requests) under {output_dir}"
    )


def _load_openai_status(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "status" / "openai_batch.json"
    return _read_json(path) if path.exists() else None


def _save_openai_status(output_dir: Path, payload: dict[str, Any]) -> None:
    _write_json(output_dir / "status" / "openai_batch.json", payload)


def _load_anthropic_status(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "status" / "anthropic_batch.json"
    return _read_json(path) if path.exists() else None


def _save_anthropic_status(output_dir: Path, payload: dict[str, Any]) -> None:
    _write_json(output_dir / "status" / "anthropic_batch.json", payload)


def _gemini_shard_status_path(output_dir: Path, shard_id: int) -> Path:
    return output_dir / "status" / "gemini_batches" / f"gemini_batch_{shard_id:04d}.json"


def _load_gemini_shard_statuses(output_dir: Path) -> list[dict[str, Any]]:
    status_dir = output_dir / "status" / "gemini_batches"
    if not status_dir.exists():
        return []
    return [_read_json(path) for path in sorted(status_dir.glob("gemini_batch_*.json"))]


def _load_fireworks_status(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "status" / "fireworks_batch.json"
    return _read_json(path) if path.exists() else None


def _save_fireworks_status(output_dir: Path, payload: dict[str, Any]) -> None:
    _write_json(output_dir / "status" / "fireworks_batch.json", _sanitize_fireworks_payload(payload))


def _sanitize_fireworks_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(payload)
    sanitized.pop("createdBy", None)
    return sanitized


def _anthropic_batch_request_count(batch: dict[str, Any]) -> int:
    counts = batch.get("request_counts") or {}
    if not counts:
        return int(batch.get("request_count") or 0)
    return sum(int(counts.get(key) or 0) for key in counts)


def submit_openai_batch(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    request_path = output_dir / "requests" / "openai_requests.jsonl"
    if not request_path.exists():
        raise FileNotFoundError(f"missing {request_path}; run prepare first")
    existing = _load_openai_status(output_dir)
    if existing and existing.get("status") not in OPENAI_TERMINAL_STATUSES and not args.force:
        print(f"OpenAI batch already active: {existing.get('id')} status={existing.get('status')}")
        return

    api_key = _load_provider_env("OPENAI_API_KEY")
    headers = _openai_headers(api_key)
    batch_list = _request_json(
        "GET",
        "https://api.openai.com/v1/batches",
        headers={**headers, "Content-Type": "application/json"},
        params={"limit": "100"},
    )
    open_batches = [
        batch
        for batch in batch_list.get("data", [])
        if batch.get("status") in OPENAI_ACTIVE_STATUSES
    ]
    if len(open_batches) >= args.max_open_batches and not args.force:
        ids = ", ".join(str(batch.get("id")) for batch in open_batches[:5])
        raise RuntimeError(
            f"OpenAI has {len(open_batches)} active batches, at or above "
            f"--max-open-batches={args.max_open_batches}. Active: {ids}"
        )

    with request_path.open("rb") as handle:
        file_payload = _request_json(
            "POST",
            "https://api.openai.com/v1/files",
            headers=headers,
            files={"file": (request_path.name, handle, "application/jsonl")},
            data={"purpose": "batch"},
        )
    batch_payload = _request_json(
        "POST",
        "https://api.openai.com/v1/batches",
        headers={**headers, "Content-Type": "application/json"},
        json_payload={
            "input_file_id": file_payload["id"],
            "endpoint": "/v1/responses",
            "completion_window": "24h",
            "metadata": {"dataset": "no_robots", "source": "slop_sftdiv", "turns": "all_assistant"},
        },
    )
    status = {
        **batch_payload,
        "submitted_at": int(time.time()),
        "input_file": file_payload,
        "request_path": str(request_path),
    }
    _save_openai_status(output_dir, status)
    print(f"Submitted OpenAI batch {batch_payload.get('id')} status={batch_payload.get('status')}")


def _submit_gemini_request_file(
    *,
    api_key: str,
    model: str,
    request_path: Path,
    display_name: str,
) -> dict[str, Any]:
    file_bytes = request_path.read_bytes()
    start_response = requests.post(
        "https://generativelanguage.googleapis.com/upload/v1beta/files",
        headers={
            "x-goog-api-key": api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(file_bytes)),
            "X-Goog-Upload-Header-Content-Type": "application/jsonl",
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": request_path.name}},
        timeout=120,
    )
    if start_response.status_code >= 400:
        raise RuntimeError(
            f"Gemini file upload start failed with {start_response.status_code}: "
            f"{start_response.text[:2000]}"
        )
    upload_url = start_response.headers.get("x-goog-upload-url")
    if not upload_url:
        raise RuntimeError("Gemini upload start did not return x-goog-upload-url")

    upload_response = _request_json(
        "POST",
        upload_url,
        headers={
            "Content-Length": str(len(file_bytes)),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
            "Content-Type": "application/jsonl",
        },
        data=file_bytes,
        timeout=300,
    )
    file_name = upload_response["file"]["name"]
    batch_payload = _request_json(
        "POST",
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchGenerateContent",
        headers=_gemini_headers(api_key),
        json_payload={
            "batch": {
                "displayName": display_name,
                "inputConfig": {"fileName": file_name},
            }
        },
    )
    return {
        **batch_payload,
        "submitted_at": int(time.time()),
        "input_file": upload_response,
    }


def _refresh_gemini_status(status: dict[str, Any], api_key: str) -> dict[str, Any]:
    refreshed = _request_json(
        "GET",
        f"https://generativelanguage.googleapis.com/v1beta/{status['name']}",
        headers=_gemini_headers(api_key),
    )
    return {**status, **refreshed, "last_checked_at": int(time.time())}


def _refresh_gemini_shards(output_dir: Path, api_key: str) -> list[dict[str, Any]]:
    refreshed_statuses = []
    for status in _load_gemini_shard_statuses(output_dir):
        refreshed = status
        if status.get("name"):
            refreshed = _refresh_gemini_status(status, api_key)
            if status.get("shard_id") is not None:
                _write_json(_gemini_shard_status_path(output_dir, int(status["shard_id"])), refreshed)
        refreshed_statuses.append(refreshed)
    return refreshed_statuses


def _gemini_shard_manifest(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "gemini_shard_manifest.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run prepare first")
    return pd.read_parquet(path)


def _gemini_pending_shards(output_dir: Path) -> list[int]:
    shard_manifest = _gemini_shard_manifest(output_dir)
    statuses = {
        int(status["shard_id"]): status
        for status in _load_gemini_shard_statuses(output_dir)
        if status.get("shard_id") is not None
    }
    pending = []
    for row in shard_manifest.itertuples(index=False):
        status = statuses.get(int(row.shard_id))
        if status is None or _gemini_state(status) in {
            "BATCH_STATE_FAILED",
            "BATCH_STATE_CANCELLED",
            "BATCH_STATE_EXPIRED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }:
            pending.append(int(row.shard_id))
    return pending


def submit_gemini_batch(args: argparse.Namespace) -> None:
    _submit_next_gemini_shard(args)


def _submit_next_gemini_shard(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    api_key = _load_provider_env("GEMINI_API_KEY")
    summary = _read_json(output_dir / "summary.json")
    model = summary["gemini_model"]
    active_statuses = []
    for status in _load_gemini_shard_statuses(output_dir):
        refreshed = status
        if status.get("name"):
            refreshed = _refresh_gemini_status(status, api_key)
            if status.get("shard_id") is not None:
                _write_json(_gemini_shard_status_path(output_dir, int(status["shard_id"])), refreshed)
        if not _gemini_done(refreshed):
            active_statuses.append(refreshed)
    if active_statuses and not args.force:
        active = active_statuses[0]
        print(f"Gemini shard already active: {active.get('name')} state={_gemini_state(active)}")
        return

    shard_manifest = _gemini_shard_manifest(output_dir)
    submitted_by_shard = {
        int(status["shard_id"]): status
        for status in _load_gemini_shard_statuses(output_dir)
        if status.get("shard_id") is not None
    }
    next_row = None
    for row in shard_manifest.itertuples(index=False):
        status = submitted_by_shard.get(int(row.shard_id))
        if status is None or _gemini_state(status) in {
            "BATCH_STATE_FAILED",
            "BATCH_STATE_CANCELLED",
            "BATCH_STATE_EXPIRED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }:
            next_row = row
            break
    if next_row is None:
        print("No pending Gemini shards to submit")
        return

    request_path = Path(str(next_row.path))
    status = _submit_gemini_request_file(
        api_key=api_key,
        model=model,
        request_path=request_path,
        display_name=f"no-robots-{_safe_model_name(model)}-all-assistant-{int(next_row.shard_id):04d}",
    )
    status = {
        **status,
        "shard_id": int(next_row.shard_id),
        "request_path": str(request_path),
        "request_count": int(next_row.request_count),
        "estimated_enqueued_tokens": int(next_row.estimated_enqueued_tokens),
    }
    _write_json(_gemini_shard_status_path(output_dir, int(next_row.shard_id)), status)
    print(
        f"Submitted Gemini shard {int(next_row.shard_id):04d} "
        f"({int(next_row.request_count)} requests, "
        f"{int(next_row.estimated_enqueued_tokens)} estimated enqueued tokens) "
        f"as {status.get('name')} state={_gemini_state(status)}"
    )


def submit_anthropic_batch(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    request_path = output_dir / "requests" / "anthropic_requests.jsonl"
    if not request_path.exists():
        raise FileNotFoundError(f"missing {request_path}; run prepare first")
    existing = _load_anthropic_status(output_dir)
    if (
        existing
        and existing.get("processing_status") not in ANTHROPIC_TERMINAL_STATUSES
        and not args.force
    ):
        print(
            "Anthropic batch already active: "
            f"{existing.get('id')} status={existing.get('processing_status')}"
        )
        return

    request_rows = _iter_jsonl_rows(request_path)
    if not request_rows:
        raise RuntimeError(f"{request_path} has no requests")
    if len(request_rows) > 100_000:
        raise RuntimeError("Anthropic Message Batches support at most 100,000 requests per batch")

    api_key = _load_provider_env("ANTHROPIC_API_KEY")
    headers = _anthropic_headers(api_key)
    batch_list = _request_json(
        "GET",
        "https://api.anthropic.com/v1/messages/batches",
        headers=headers,
        params={"limit": "100"},
    )
    active_batches = [
        batch
        for batch in batch_list.get("data", [])
        if batch.get("processing_status") not in ANTHROPIC_TERMINAL_STATUSES
    ]
    active_requests = sum(_anthropic_batch_request_count(batch) for batch in active_batches)
    if active_requests + len(request_rows) > args.max_active_batch_requests and not args.force:
        ids = ", ".join(str(batch.get("id")) for batch in active_batches[:5])
        raise RuntimeError(
            f"Anthropic has about {active_requests} active batch requests; submitting "
            f"{len(request_rows)} would exceed --max-active-batch-requests="
            f"{args.max_active_batch_requests}. Active batches: {ids}"
        )

    batch_payload = _request_json(
        "POST",
        "https://api.anthropic.com/v1/messages/batches",
        headers=headers,
        json_payload={"requests": request_rows},
        timeout=300,
    )
    status = {
        **batch_payload,
        "submitted_at": int(time.time()),
        "request_path": str(request_path),
        "request_count": len(request_rows),
        "thinking": DEFAULT_ANTHROPIC_THINKING,
    }
    _save_anthropic_status(output_dir, status)
    print(
        f"Submitted Anthropic batch {batch_payload.get('id')} "
        f"status={batch_payload.get('processing_status')} requests={len(request_rows)} "
        f"thinking={DEFAULT_ANTHROPIC_THINKING}"
    )


def _fireworks_headers(api_key: str, *, content_type: str | None = "application/json") -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _fireworks_account_id(api_key: str, explicit_account_id: str | None = None) -> str:
    load_dotenv()
    account_id = explicit_account_id or os.environ.get("FIREWORKS_ACCOUNT_ID", "")
    if account_id:
        return account_id.removeprefix("accounts/")
    payload = _request_json(
        "GET",
        "https://api.fireworks.ai/v1/accounts",
        headers=_fireworks_headers(api_key),
        params={"pageSize": "200"},
    )
    accounts = payload.get("accounts") or []
    ready_accounts = [account for account in accounts if account.get("state") in {None, "READY"}]
    candidates = ready_accounts or accounts
    if len(candidates) != 1:
        raise RuntimeError(
            "FIREWORKS_ACCOUNT_ID is not set and the Fireworks account list did not "
            f"resolve to exactly one account (found {len(candidates)})"
        )
    name = str(candidates[0].get("name") or "")
    if not name:
        raise RuntimeError("Fireworks account response did not include an account name")
    return name.rsplit("/", 1)[-1]


def _fireworks_job_id() -> str:
    return f"no-robots-all-assistant-glm-5p2-{int(time.time())}"


def submit_fireworks_batch(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    request_path = output_dir / "requests" / "fireworks_requests.jsonl"
    if not request_path.exists():
        raise FileNotFoundError(f"missing {request_path}; run prepare first")
    existing = _load_fireworks_status(output_dir)
    if existing and existing.get("state") not in FIREWORKS_TERMINAL_STATES and not args.force:
        print(f"Fireworks batch already active: {existing.get('name')} state={existing.get('state')}")
        return

    summary = _read_json(output_dir / "summary.json")
    request_count = sum(1 for line in request_path.open(encoding="utf-8") if line.strip())
    api_key = _load_provider_env("FIREWORKS_API_KEY")
    account_id = _fireworks_account_id(api_key, args.account_id)
    job_id = _fireworks_job_id()
    input_dataset_id = f"{job_id}-input"
    output_dataset_id = f"{job_id}-output"

    dataset_payload = _request_json(
        "POST",
        f"https://api.fireworks.ai/v1/accounts/{account_id}/datasets",
        headers=_fireworks_headers(api_key),
        json_payload={
            "datasetId": input_dataset_id,
            "dataset": {
                "displayName": "no_robots all assistant turns batch input",
                "userUploaded": {},
                "format": "CHAT",
                "exampleCount": request_count,
            },
        },
    )
    with request_path.open("rb") as handle:
        upload_payload = _request_json(
            "POST",
            f"https://api.fireworks.ai/v1/accounts/{account_id}/datasets/{input_dataset_id}:upload",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (request_path.name, handle, "application/jsonl")},
            timeout=300,
        )
    job_payload = _request_json(
        "POST",
        f"https://api.fireworks.ai/v1/accounts/{account_id}/batchInferenceJobs",
        headers=_fireworks_headers(api_key),
        params={"batchInferenceJobId": job_id},
        json_payload={
            "displayName": "no_robots all assistant turns glm-5p2",
            "model": summary["fireworks_model"],
            "inputDatasetId": f"accounts/{account_id}/datasets/{input_dataset_id}",
            "outputDatasetId": f"accounts/{account_id}/datasets/{output_dataset_id}",
        },
    )
    status = {
        **job_payload,
        "submitted_at": int(time.time()),
        "account_id": account_id,
        "job_id": job_id,
        "input_dataset_id": input_dataset_id,
        "output_dataset_id": output_dataset_id,
        "request_path": str(request_path),
        "dataset": dataset_payload,
        "upload": upload_payload,
    }
    _save_fireworks_status(output_dir, status)
    print(f"Submitted Fireworks batch {job_payload.get('name')} state={job_payload.get('state')}")


def _refresh_openai_status(output_dir: Path, api_key: str) -> dict[str, Any] | None:
    status = _load_openai_status(output_dir)
    if not status or not status.get("id"):
        return status
    refreshed = _request_json(
        "GET",
        f"https://api.openai.com/v1/batches/{status['id']}",
        headers={**_openai_headers(api_key), "Content-Type": "application/json"},
    )
    merged = {**status, **refreshed, "last_checked_at": int(time.time())}
    _save_openai_status(output_dir, merged)
    return merged


def _refresh_anthropic_status(output_dir: Path, api_key: str) -> dict[str, Any] | None:
    status = _load_anthropic_status(output_dir)
    if not status or not status.get("id"):
        return status
    refreshed = _request_json(
        "GET",
        f"https://api.anthropic.com/v1/messages/batches/{status['id']}",
        headers=_anthropic_headers(api_key),
    )
    merged = {**status, **refreshed, "last_checked_at": int(time.time())}
    _save_anthropic_status(output_dir, merged)
    return merged


def _refresh_fireworks_status(
    output_dir: Path,
    api_key: str,
    explicit_account_id: str | None = None,
) -> dict[str, Any] | None:
    status = _load_fireworks_status(output_dir)
    if not status or not status.get("job_id"):
        return status
    account_id = str(status.get("account_id") or _fireworks_account_id(api_key, explicit_account_id))
    refreshed = _request_json(
        "GET",
        f"https://api.fireworks.ai/v1/accounts/{account_id}/batchInferenceJobs/{status['job_id']}",
        headers=_fireworks_headers(api_key),
    )
    merged = {**status, **refreshed, "last_checked_at": int(time.time())}
    _save_fireworks_status(output_dir, merged)
    return merged


def _status_summary(status: dict[str, Any] | None, *, provider: str) -> dict[str, Any] | None:
    if not status:
        return None
    if provider == "openai":
        return {
            "id": status.get("id"),
            "status": status.get("status"),
            "request_counts": status.get("request_counts"),
            "output_file_id": status.get("output_file_id"),
            "error_file_id": status.get("error_file_id"),
        }
    if provider == "gemini":
        return {
            "name": status.get("name"),
            "state": _gemini_state(status),
            "shard_id": status.get("shard_id"),
            "request_count": status.get("request_count"),
            "estimated_enqueued_tokens": status.get("estimated_enqueued_tokens"),
            "batch_stats": (status.get("metadata") or {}).get("batchStats") or status.get("batchStats"),
            "response_file": _gemini_response_file(status),
        }
    if provider == "anthropic":
        return {
            "id": status.get("id"),
            "processing_status": status.get("processing_status"),
            "request_counts": status.get("request_counts"),
            "request_count": status.get("request_count"),
            "thinking": status.get("thinking"),
        }
    return {
        "name": status.get("name"),
        "state": status.get("state"),
        "status": status.get("status"),
        "job_progress": status.get("jobProgress"),
        "input_dataset_id": status.get("input_dataset_id"),
        "output_dataset_id": status.get("output_dataset_id"),
    }


def status_batches(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    load_dotenv()
    openai = None
    gemini = _load_gemini_shard_statuses(output_dir)
    anthropic = None
    fireworks = None
    if os.environ.get("OPENAI_API_KEY"):
        openai = _refresh_openai_status(output_dir, os.environ["OPENAI_API_KEY"])
    else:
        openai = _load_openai_status(output_dir)
    if os.environ.get("GEMINI_API_KEY"):
        gemini = _refresh_gemini_shards(output_dir, os.environ["GEMINI_API_KEY"])
    if os.environ.get("ANTHROPIC_API_KEY"):
        anthropic = _refresh_anthropic_status(output_dir, os.environ["ANTHROPIC_API_KEY"])
    else:
        anthropic = _load_anthropic_status(output_dir)
    if os.environ.get("FIREWORKS_API_KEY"):
        fireworks = _refresh_fireworks_status(
            output_dir,
            os.environ["FIREWORKS_API_KEY"],
            args.fireworks_account_id,
        )
    else:
        fireworks = _load_fireworks_status(output_dir)

    print(
        json.dumps(
            {
                "openai": _status_summary(openai, provider="openai"),
                "gemini": [_status_summary(status, provider="gemini") for status in gemini],
                "anthropic": _status_summary(anthropic, provider="anthropic"),
                "fireworks": _status_summary(fireworks, provider="fireworks"),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _download_completed_gemini_shards(
    *,
    output_dir: Path,
    api_key: str,
    statuses: list[dict[str, Any]],
) -> int:
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for status in statuses:
        state = _gemini_state(status)
        if state not in {"JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"}:
            continue
        shard_id = status.get("shard_id")
        if shard_id is None:
            continue
        file_name = _gemini_response_file(status)
        if not file_name:
            continue
        output_path = results_dir / f"gemini_output_{int(shard_id):04d}.jsonl"
        before = output_path.exists() and output_path.stat().st_size > 0
        _download_gemini_file(api_key, file_name, output_path)
        after = output_path.exists() and output_path.stat().st_size > 0
        if after and not before:
            downloaded += 1
    return downloaded


def _download_fireworks_results(
    *,
    output_dir: Path,
    api_key: str,
    status: dict[str, Any],
) -> int:
    account_id = str(status["account_id"])
    output_dataset_id = str(status["output_dataset_id"])
    payload = _request_json(
        "GET",
        f"https://api.fireworks.ai/v1/accounts/{account_id}/datasets/{output_dataset_id}:getDownloadEndpoint",
        headers=_fireworks_headers(api_key),
        json_payload={},
        timeout=120,
    )
    signed_urls = payload.get("filenameToSignedUrls") or {}
    if not signed_urls:
        return 0
    results_dir = output_dir / "results" / "fireworks"
    results_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for object_path, signed_url in signed_urls.items():
        filename = Path(str(object_path)).name or f"fireworks_result_{downloaded:04d}.jsonl"
        output_path = results_dir / filename
        if output_path.exists() and output_path.stat().st_size > 0:
            continue
        response = requests.get(str(signed_url), timeout=300)
        if response.status_code >= 400:
            raise RuntimeError(
                f"Fireworks result download failed with {response.status_code}: "
                f"{response.text[:2000]}"
            )
        output_path.write_bytes(response.content)
        downloaded += 1
        print(f"Wrote {output_path}")
    return downloaded


def fetch_completed_batches(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv()

    if os.environ.get("OPENAI_API_KEY"):
        openai_status = _refresh_openai_status(output_dir, os.environ["OPENAI_API_KEY"])
        if openai_status and openai_status.get("status") == "completed":
            _download_openai_file(
                os.environ["OPENAI_API_KEY"],
                openai_status.get("output_file_id"),
                results_dir / "openai_output.jsonl",
            )
            _download_openai_file(
                os.environ["OPENAI_API_KEY"],
                openai_status.get("error_file_id"),
                results_dir / "openai_errors.jsonl",
                required=False,
            )
    if os.environ.get("GEMINI_API_KEY"):
        statuses = _refresh_gemini_shards(output_dir, os.environ["GEMINI_API_KEY"])
        _download_completed_gemini_shards(
            output_dir=output_dir,
            api_key=os.environ["GEMINI_API_KEY"],
            statuses=statuses,
        )
    if os.environ.get("ANTHROPIC_API_KEY"):
        anthropic_status = _refresh_anthropic_status(output_dir, os.environ["ANTHROPIC_API_KEY"])
        if anthropic_status and anthropic_status.get("processing_status") == "ended":
            _download_anthropic_results(
                os.environ["ANTHROPIC_API_KEY"],
                str(anthropic_status["id"]),
                results_dir / "anthropic_output.jsonl",
            )
    if os.environ.get("FIREWORKS_API_KEY"):
        fireworks_status = _refresh_fireworks_status(
            output_dir,
            os.environ["FIREWORKS_API_KEY"],
            args.fireworks_account_id,
        )
        if fireworks_status and fireworks_status.get("state") in {
            "JOB_STATE_COMPLETED",
            "JOB_STATE_EXPIRED",
        }:
            _download_fireworks_results(
                output_dir=output_dir,
                api_key=os.environ["FIREWORKS_API_KEY"],
                status=fireworks_status,
            )


def drive_gemini_batches(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.max_runtime_seconds <= 0:
        raise ValueError("--max-runtime-seconds must be positive")
    api_key = _load_provider_env("GEMINI_API_KEY")
    start = time.time()
    while time.time() - start < args.max_runtime_seconds:
        statuses = _refresh_gemini_shards(output_dir, api_key)
        _download_completed_gemini_shards(output_dir=output_dir, api_key=api_key, statuses=statuses)
        active = [status for status in statuses if not _gemini_done(status)]
        pending = _gemini_pending_shards(output_dir)
        done_count = sum(
            1
            for status in statuses
            if _gemini_state(status) in {"BATCH_STATE_SUCCEEDED", "JOB_STATE_SUCCEEDED"}
        )
        print(
            json.dumps(
                {
                    "elapsed_seconds": int(time.time() - start),
                    "submitted_shards": len(statuses),
                    "succeeded_shards": done_count,
                    "active_shards": [
                        {
                            "shard_id": status.get("shard_id"),
                            "state": _gemini_state(status),
                            "batch_stats": (status.get("metadata") or {}).get("batchStats"),
                        }
                        for status in active
                    ],
                    "pending_shards": pending,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not active:
            if not pending:
                print("All Gemini shards are terminal", flush=True)
                return
            submit_args = argparse.Namespace(output_dir=output_dir, force=False)
            _submit_next_gemini_shard(submit_args)
            continue
        remaining = args.max_runtime_seconds - int(time.time() - start)
        if remaining <= 0:
            break
        time.sleep(min(args.poll_seconds, remaining))
    print("Gemini driver stopped at runtime limit", flush=True)


def drive_anthropic_batch(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.max_runtime_seconds <= 0:
        raise ValueError("--max-runtime-seconds must be positive")
    api_key = _load_provider_env("ANTHROPIC_API_KEY")
    start = time.time()
    while time.time() - start < args.max_runtime_seconds:
        status = _refresh_anthropic_status(output_dir, api_key)
        if not status:
            raise RuntimeError("no Anthropic batch status found; run submit-anthropic first")
        processing_status = str(status.get("processing_status") or "")
        print(
            json.dumps(
                {
                    "elapsed_seconds": int(time.time() - start),
                    "id": status.get("id"),
                    "processing_status": processing_status,
                    "request_counts": status.get("request_counts"),
                    "thinking": status.get("thinking"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if processing_status == "ended":
            results_dir = output_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            _download_anthropic_results(
                api_key,
                str(status["id"]),
                results_dir / "anthropic_output.jsonl",
            )
            print("Anthropic batch is ended and results are fetched", flush=True)
            return
        remaining = args.max_runtime_seconds - int(time.time() - start)
        if remaining <= 0:
            break
        time.sleep(min(args.poll_seconds, remaining))
    print("Anthropic driver stopped at runtime limit", flush=True)


def drive_fireworks_batch(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.max_runtime_seconds <= 0:
        raise ValueError("--max-runtime-seconds must be positive")
    api_key = _load_provider_env("FIREWORKS_API_KEY")
    start = time.time()
    while time.time() - start < args.max_runtime_seconds:
        status = _refresh_fireworks_status(output_dir, api_key, args.fireworks_account_id)
        if not status:
            raise RuntimeError("no Fireworks batch status found; run submit-fireworks first")
        state = str(status.get("state") or "")
        print(
            json.dumps(
                {
                    "elapsed_seconds": int(time.time() - start),
                    "name": status.get("name"),
                    "state": state,
                    "job_progress": status.get("jobProgress"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if state in {"JOB_STATE_COMPLETED", "JOB_STATE_EXPIRED"}:
            _download_fireworks_results(output_dir=output_dir, api_key=api_key, status=status)
            print("Fireworks batch is terminal and results are fetched", flush=True)
            return
        if state in FIREWORKS_TERMINAL_STATES:
            print("Fireworks batch is terminal without downloadable success results", flush=True)
            return
        remaining = args.max_runtime_seconds - int(time.time() - start)
        if remaining <= 0:
            break
        time.sleep(min(args.poll_seconds, remaining))
    print("Fireworks driver stopped at runtime limit", flush=True)


def _collect_generated_rows(output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_parquet(output_dir / "request_manifest.parquet")
    turns = pd.read_parquet(output_dir / "assistant_turns.parquet")
    turn_by_id = {str(row.turn_id): row._asdict() for row in turns.itertuples(index=False)}
    manifest_by_custom = {
        str(row.custom_id): row._asdict() for row in manifest.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    parse_counts: Counter[str] = Counter()
    results_dir = output_dir / "results"

    def append_generated(custom_id: str, text: str, *, batch_id: str | None) -> None:
        meta = manifest_by_custom.get(custom_id)
        if not meta:
            parse_counts["unknown_custom_id"] += 1
            return
        turn = turn_by_id[str(meta["turn_id"])]
        rows.append(
            {
                "custom_id": custom_id,
                "turn_id": meta["turn_id"],
                "provider": meta["provider"],
                "model": meta["model"],
                "split": meta["split"],
                "source_row_index": meta["source_row_index"],
                "prompt_id": meta["prompt_id"],
                "category": meta["category"],
                "assistant_message_index": meta["assistant_message_index"],
                "assistant_turn_ordinal": meta["assistant_turn_ordinal"],
                "uses_transcript_prompt": meta["uses_transcript_prompt"],
                "batch_id": batch_id,
                "generated_text": text,
                "generated_word_count": len(text.split()),
                "generated_char_count": len(text),
                "generated_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "target_text": turn["target_text"],
                "target_word_count": turn["target_word_count"],
                "target_char_count": turn["target_char_count"],
                "target_sha256": turn["target_sha256"],
            }
        )

    for path in sorted(results_dir.glob("openai_output*.jsonl")):
        for row in _iter_jsonl_rows(path):
            parse_counts["openai_rows"] += 1
            if row.get("error"):
                parse_counts["openai_error_rows"] += 1
                continue
            text = _openai_text_from_output(row)
            if not text:
                parse_counts["openai_empty_rows"] += 1
                continue
            append_generated(str(row.get("custom_id")), text, batch_id=row.get("id"))

    for path in sorted(results_dir.glob("gemini_output*.jsonl")):
        for row in _iter_jsonl_rows(path):
            parse_counts["gemini_rows"] += 1
            if row.get("error"):
                parse_counts["gemini_error_rows"] += 1
                continue
            text = _gemini_text_from_output(row)
            if not text:
                parse_counts["gemini_empty_rows"] += 1
                continue
            append_generated(
                str(row.get("key") or ((row.get("metadata") or {}).get("key"))),
                text,
                batch_id=((row.get("response") or {}).get("responseId")),
            )

    for path in sorted(results_dir.glob("anthropic_output*.jsonl")):
        for row in _iter_jsonl_rows(path):
            parse_counts["anthropic_rows"] += 1
            result = row.get("result") or {}
            if result.get("type") != "succeeded":
                parse_counts[f"anthropic_{result.get('type') or 'unknown'}_rows"] += 1
                continue
            text = _anthropic_text_from_output(row)
            if not text:
                parse_counts["anthropic_empty_rows"] += 1
                continue
            append_generated(
                str(row.get("custom_id")),
                text,
                batch_id=((result.get("message") or {}).get("id")),
            )

    for path in sorted((results_dir / "fireworks").glob("*.jsonl")):
        for row in _iter_jsonl_rows(path):
            parse_counts["fireworks_rows"] += 1
            custom_id = str(row.get("custom_id") or row.get("id") or "")
            if row.get("error"):
                parse_counts["fireworks_error_rows"] += 1
                continue
            choices = ((row.get("response") or {}).get("choices")) or row.get("choices") or []
            text = ""
            if choices:
                message = choices[0].get("message") or {}
                text = str(message.get("content") or "").strip()
            if not text:
                parse_counts["fireworks_empty_rows"] += 1
                continue
            append_generated(custom_id, text, batch_id=str(row.get("request_id") or ""))

    generated = pd.DataFrame(rows)
    return generated, dict(sorted(parse_counts.items()))


def _token_lengths(
    texts: list[str],
    *,
    tokenizer_model: str,
    batch_size: int = 512,
) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    lengths: list[int] = []
    for index in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[index : index + batch_size],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])
    return lengths


def _complete_triplet_turn_ids(
    frame: pd.DataFrame,
    *,
    expected_models: list[str],
    max_token_length: int | None = None,
) -> tuple[set[str], dict[str, Any]]:
    if not expected_models:
        raise ValueError("expected_models must not be empty")
    missing = {"turn_id", "model", "record_id"} - set(frame.columns)
    if missing:
        raise ValueError(f"categorical frame missing required columns: {sorted(missing)}")
    presence = frame.pivot_table(
        index="turn_id",
        columns="model",
        values="record_id",
        aggfunc="first",
    )
    model_columns = [model for model in expected_models if model in presence.columns]
    complete = presence.dropna(subset=model_columns)
    missing_expected = sorted(set(expected_models) - set(presence.columns))
    if missing_expected:
        complete = complete.iloc[0:0]
    eligible_index = complete.index
    length_summary: dict[str, Any] = {}
    if max_token_length is not None:
        if max_token_length <= 0:
            raise ValueError("max_token_length must be positive when provided")
        if "token_len" not in frame.columns:
            raise ValueError("token_len is required when max_token_length is provided")
        lengths = frame.pivot_table(
            index="turn_id",
            columns="model",
            values="token_len",
            aggfunc="first",
        )
        lengths = lengths.reindex(complete.index)
        under_cap = (lengths[expected_models] <= max_token_length).all(axis=1)
        eligible_index = lengths.index[under_cap]
        length_summary = {
            "max_triplet_token_length": max_token_length,
            "complete_triplets_over_cap": int((~under_cap).sum()),
            "complete_triplets_under_or_equal_cap": int(under_cap.sum()),
        }
    eligible = {str(turn_id) for turn_id in eligible_index}
    summary = {
        "expected_models": expected_models,
        "candidate_turns": int(frame["turn_id"].nunique()),
        "complete_triplets": int(len(complete)),
        "eligible_triplets": int(len(eligible)),
        "incomplete_or_missing_triplets": int(frame["turn_id"].nunique() - len(complete)),
        "missing_expected_models": missing_expected,
        **length_summary,
    }
    return eligible, summary


def _default_categorical_dataset_dir(
    *,
    output_dir: Path,
    providers: list[str],
    max_triplet_token_length: int,
) -> Path:
    provider_slug = "_".join(provider.replace("-", "_") for provider in providers)
    if max_triplet_token_length > 0:
        return output_dir / f"categorical_{provider_slug}_triplet_cap_{max_triplet_token_length}"
    return output_dir / f"categorical_{provider_slug}_complete_triplets"


def _token_length_summary(frame: pd.DataFrame, *, prefix: str) -> dict[str, Any]:
    if "token_len" not in frame.columns or len(frame) == 0:
        return {}
    lengths = frame["token_len"]
    return {
        f"{prefix}_token_len_mean": float(lengths.mean()),
        f"{prefix}_token_len_p50": int(lengths.quantile(0.5)),
        f"{prefix}_token_len_p90": int(lengths.quantile(0.9)),
        f"{prefix}_token_len_p95": int(lengths.quantile(0.95)),
        f"{prefix}_token_len_p99": int(lengths.quantile(0.99)),
        f"{prefix}_token_len_max": int(lengths.max()),
    }


def compile_categorical_dataset(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    providers = args.provider or ["openai", "gemini"]
    providers = [str(provider).strip() for provider in providers if str(provider).strip()]
    if not providers:
        raise ValueError("at least one --provider is required")
    dataset_output_dir: Path = args.dataset_output_dir or _default_categorical_dataset_dir(
        output_dir=output_dir,
        providers=providers,
        max_triplet_token_length=args.max_triplet_token_length,
    )
    (dataset_output_dir / "data").mkdir(parents=True, exist_ok=True)
    (dataset_output_dir / "metadata").mkdir(parents=True, exist_ok=True)

    turns = pd.read_parquet(output_dir / "assistant_turns.parquet")
    generated, parse_counts = _collect_generated_rows(output_dir)
    if len(generated):
        generated = generated[generated["provider"].isin(providers)].copy()

    human_rows = []
    for row in turns.itertuples(index=False):
        text = str(row.target_text).strip()
        if not text:
            continue
        human_rows.append(
            {
                "record_id": f"{row.prompt_id}@turn_{int(row.assistant_turn_ordinal):02d}@human",
                "base_doc_id": str(row.prompt_id),
                "turn_id": str(row.turn_id),
                "source_dataset": str(row.source_dataset),
                "source_split": str(row.split),
                "source_row_index": int(row.source_row_index),
                "category": str(row.category),
                "assistant_message_index": int(row.assistant_message_index),
                "assistant_turn_ordinal": int(row.assistant_turn_ordinal),
                "role": "human_continuation",
                "provider": "human",
                "model": "human",
                "generated_by_project": False,
                "batch_custom_id": None,
                "batch_id": None,
                "text": text,
                "word_count": len(text.split()),
                "char_count": len(text),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )

    generated_rows = []
    for row in generated.itertuples(index=False):
        text = str(row.generated_text).strip()
        if not text:
            continue
        provider = str(row.provider)
        model = str(row.model)
        generated_rows.append(
            {
                "record_id": (
                    f"{row.prompt_id}@turn_{int(row.assistant_turn_ordinal):02d}@{model}"
                ),
                "base_doc_id": str(row.prompt_id),
                "turn_id": str(row.turn_id),
                "source_dataset": "HuggingFaceH4/no_robots",
                "source_split": str(row.split),
                "source_row_index": int(row.source_row_index),
                "category": str(row.category),
                "assistant_message_index": int(row.assistant_message_index),
                "assistant_turn_ordinal": int(row.assistant_turn_ordinal),
                "role": "llm_continuation",
                "provider": provider,
                "model": model,
                "generated_by_project": True,
                "batch_custom_id": str(row.custom_id),
                "batch_id": str(row.batch_id or ""),
                "text": text,
                "word_count": len(text.split()),
                "char_count": len(text),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )

    combined = pd.DataFrame([*human_rows, *generated_rows])
    if len(combined) == 0:
        raise RuntimeError("no categorical rows were available to compile")
    provider_models = sorted(
        combined.loc[combined["provider"].isin(providers), "model"].unique().tolist()
    )
    expected_models = ["human", *provider_models]
    token_summary: dict[str, Any] = {}
    if args.max_triplet_token_length > 0:
        combined["token_len"] = _token_lengths(
            combined["text"].fillna("").astype(str).tolist(),
            tokenizer_model=args.tokenizer_model,
        )
        token_summary = {
            "tokenizer_model": args.tokenizer_model,
            **_token_length_summary(combined, prefix="candidate"),
        }
    eligible_turn_ids, triplet_summary = _complete_triplet_turn_ids(
        combined,
        expected_models=expected_models,
        max_token_length=args.max_triplet_token_length
        if args.max_triplet_token_length > 0
        else None,
    )
    combined = combined[combined["turn_id"].isin(eligible_turn_ids)].copy()
    generated_filtered = generated[generated["turn_id"].isin(eligible_turn_ids)].copy()
    token_summary.update(_token_length_summary(combined, prefix="filtered"))

    combined_path = dataset_output_dir / "data" / "train.parquet"
    generated_path = dataset_output_dir / "metadata" / "generated_texts.parquet"
    turns_path = dataset_output_dir / "metadata" / "assistant_turns.parquet"
    combined.to_parquet(combined_path, index=False)
    generated_filtered.to_parquet(generated_path, index=False)
    turns[turns["turn_id"].isin(eligible_turn_ids)].to_parquet(turns_path, index=False)

    summary = {
        "created_at": int(time.time()),
        "source_dataset": NO_ROBOTS_DATASET,
        "dataset_path": str(combined_path),
        "assistant_turns": int(len(turns)),
        "eligible_turns": int(len(eligible_turn_ids)),
        "combined_rows": int(len(combined)),
        "providers": providers,
        "expected_models": expected_models,
        "rows_by_model": combined["model"].value_counts().sort_index().to_dict(),
        "rows_by_provider": combined["provider"].value_counts().sort_index().to_dict(),
        "rows_by_role": combined["role"].value_counts().sort_index().to_dict(),
        "parse_counts": parse_counts,
        "triplet_filter": triplet_summary,
        **token_summary,
        "metadata": {
            "generated_texts": str(generated_path),
            "assistant_turns": str(turns_path),
        },
    }
    _write_json(dataset_output_dir / "dataset_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
