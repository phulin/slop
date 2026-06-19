from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from datasets import load_dataset
from dotenv import load_dotenv


HAPE_DATASET = "browndw/human-ai-parallel-corpus"
DEFAULT_OUTPUT_DIR = Path("artifacts/hape_augmented_batch")
DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_OUTPUT_TOKENS = 900
DEFAULT_GEMINI_SHARD_TOKEN_BUDGET = 2_400_000
DEFAULT_OPENAI_REASONING_EFFORT = "none"
DEFAULT_GEMINI_THINKING_LEVEL = "minimal"
DEFAULT_ANTHROPIC_THINKING = "disabled"
HAPE_INSTRUCTION = (
    "In the same style, tone, and diction of the following text, complete the next "
    "500 words, generate exactly 500 words, and note that the text does not "
    "necessarily end after the generated words:"
)
SYSTEM_PROMPT = (
    "You are producing one continuation for a style-parallel corpus. Return only "
    "the continuation text, without labels, commentary, prefaces, or markdown."
)
OPENAI_ACTIVE_STATUSES = {"validating", "in_progress", "finalizing", "cancelling"}
OPENAI_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
ANTHROPIC_TERMINAL_STATUSES = {"ended"}
GEMINI_TERMINAL_STATES = {
    "BATCH_STATE_SUCCEEDED",
    "BATCH_STATE_FAILED",
    "BATCH_STATE_CANCELLED",
    "BATCH_STATE_EXPIRED",
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


@dataclass(frozen=True)
class HapeDoc:
    base_doc_id: str
    prompt_doc_id: str
    human_doc_id: str
    prompt_text: str
    human_text: str
    texts_by_author: dict[str, str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and manage Batch API jobs for an augmented HAP-E corpus."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare.add_argument("--dataset", default=HAPE_DATASET)
    prepare.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    prepare.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    prepare.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    prepare.add_argument("--openai-reasoning-effort", default=DEFAULT_OPENAI_REASONING_EFFORT)
    prepare.add_argument("--gemini-thinking-level", default=DEFAULT_GEMINI_THINKING_LEVEL)
    prepare.add_argument(
        "--gemini-shard-token-budget",
        type=int,
        default=DEFAULT_GEMINI_SHARD_TOKEN_BUDGET,
        help=(
            "Conservative estimated prompt+reserved-output token budget per Gemini "
            "batch shard. Tier 1 Gemini 3.5 Flash allows 3,000,000 enqueued tokens."
        ),
    )
    prepare.add_argument("--max-docs", type=int, default=None)
    prepare.set_defaults(func=prepare_hape_augmented_batch)

    prepare_anthropic = subparsers.add_parser("prepare-anthropic")
    prepare_anthropic.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare_anthropic.add_argument("--anthropic-model", default=DEFAULT_ANTHROPIC_MODEL)
    prepare_anthropic.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    prepare_anthropic.add_argument(
        "--anthropic-thinking",
        choices=["disabled"],
        default=DEFAULT_ANTHROPIC_THINKING,
        help="Lowest-thinking Claude mode. 'disabled' omits the thinking parameter.",
    )
    prepare_anthropic.set_defaults(func=prepare_anthropic_requests)

    reshard = subparsers.add_parser("reshard-gemini")
    reshard.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    reshard.add_argument(
        "--gemini-shard-token-budget",
        type=int,
        default=DEFAULT_GEMINI_SHARD_TOKEN_BUDGET,
    )
    reshard.set_defaults(func=reshard_gemini_requests)

    submit_openai = subparsers.add_parser("submit-openai")
    submit_openai.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    submit_openai.add_argument("--max-open-batches", type=int, default=2)
    submit_openai.add_argument("--force", action="store_true")
    submit_openai.set_defaults(func=submit_openai_batch)

    openai_retry = subparsers.add_parser("prepare-openai-retry")
    openai_retry.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    openai_retry.add_argument("--retry-name", default="openai_retry_0001")
    openai_retry.add_argument("--max-output-tokens", type=int, default=2000)
    openai_retry.add_argument("--reasoning-effort", default=DEFAULT_OPENAI_REASONING_EFFORT)
    openai_retry.set_defaults(func=prepare_openai_retry)

    submit_openai_retry = subparsers.add_parser("submit-openai-retry")
    submit_openai_retry.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    submit_openai_retry.add_argument("--retry-name", default="openai_retry_0001")
    submit_openai_retry.add_argument("--max-open-batches", type=int, default=2)
    submit_openai_retry.add_argument("--force", action="store_true")
    submit_openai_retry.set_defaults(func=submit_openai_retry_batch)

    submit_gemini = subparsers.add_parser("submit-gemini")
    submit_gemini.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    submit_gemini.add_argument("--force", action="store_true")
    submit_gemini.set_defaults(func=submit_gemini_batch)

    submit_anthropic = subparsers.add_parser("submit-anthropic")
    submit_anthropic.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    submit_anthropic.add_argument(
        "--max-active-batch-requests",
        type=int,
        default=200_000,
        help="Tier 2 Message Batches queue limit for batch requests in processing.",
    )
    submit_anthropic.add_argument("--force", action="store_true")
    submit_anthropic.set_defaults(func=submit_anthropic_batch)

    drive_gemini = subparsers.add_parser("drive-gemini")
    drive_gemini.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    drive_gemini.add_argument("--poll-seconds", type=int, default=60)
    drive_gemini.add_argument("--max-runtime-seconds", type=int, default=1800)
    drive_gemini.set_defaults(func=drive_gemini_batches)

    drive_anthropic = subparsers.add_parser("drive-anthropic")
    drive_anthropic.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    drive_anthropic.add_argument("--poll-seconds", type=int, default=300)
    drive_anthropic.add_argument("--max-runtime-seconds", type=int, default=86400)
    drive_anthropic.set_defaults(func=drive_anthropic_batch)

    status = subparsers.add_parser("status")
    status.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    status.set_defaults(func=status_batches)

    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    fetch.set_defaults(func=fetch_completed_batches)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    finalize.set_defaults(func=finalize_augmented_dataset)

    compile_hf = subparsers.add_parser("compile-hf")
    compile_hf.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    compile_hf.add_argument(
        "--hf-output-dir",
        type=Path,
        default=None,
        help="Output directory for a Hugging Face upload-ready dataset folder.",
    )
    compile_hf.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any provider request is missing a parsed generation.",
    )
    compile_hf.set_defaults(func=compile_hf_dataset)

    return parser


def _author_from_doc_id(doc_id: str) -> tuple[str, str]:
    if "@" in doc_id:
        base, author = doc_id.rsplit("@", 1)
        return base, author
    if doc_id.endswith("_chunk_1"):
        return doc_id.removesuffix("_chunk_1"), "chunk_1"
    if doc_id.endswith("_chunk_2"):
        return doc_id.removesuffix("_chunk_2"), "chunk_2"
    parts = doc_id.rsplit("_", 2)
    if len(parts) >= 2 and parts[-2] == "chunk":
        return "_".join(parts[:-2]), f"chunk_{parts[-1]}"
    return doc_id, "unknown"


def _safe_model_name(model: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in model).strip("_").lower()


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _word_count(text: str) -> int:
    return len(text.split())


def _option_string(value: Any, *, default: str) -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    text = str(value).strip()
    return text if text else default


def _prompt_for(prompt_text: str) -> str:
    return f"{HAPE_INSTRUCTION}\n\n{prompt_text}"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_hape_docs(dataset_name: str, *, max_docs: int | None = None) -> list[HapeDoc]:
    dataset = load_dataset(dataset_name, split="train")
    by_doc: dict[str, dict[str, str]] = defaultdict(dict)
    doc_ids: dict[tuple[str, str], str] = {}
    for row in dataset:
        doc_id = str(row["doc_id"])
        base_doc_id, author = _author_from_doc_id(doc_id)
        by_doc[base_doc_id][author] = str(row["text"])
        doc_ids[(base_doc_id, author)] = doc_id

    docs: list[HapeDoc] = []
    for base_doc_id in sorted(by_doc):
        texts = by_doc[base_doc_id]
        prompt_text = texts.get("chunk_1", "").strip()
        human_text = texts.get("chunk_2", "").strip()
        if not prompt_text or not human_text:
            continue
        docs.append(
            HapeDoc(
                base_doc_id=base_doc_id,
                prompt_doc_id=doc_ids.get((base_doc_id, "chunk_1"), f"{base_doc_id}_chunk_1"),
                human_doc_id=doc_ids.get((base_doc_id, "chunk_2"), f"{base_doc_id}_chunk_2"),
                prompt_text=prompt_text,
                human_text=human_text,
                texts_by_author=dict(texts),
            )
        )
        if max_docs is not None and len(docs) >= max_docs:
            break
    return docs


def _base_doc_rows(docs: list[HapeDoc], *, dataset_name: str) -> list[dict[str, Any]]:
    rows = []
    for doc in docs:
        rows.append(
            {
                "base_doc_id": doc.base_doc_id,
                "source_dataset": dataset_name,
                "prompt_doc_id": doc.prompt_doc_id,
                "human_doc_id": doc.human_doc_id,
                "prompt_text": doc.prompt_text,
                "human_text": doc.human_text,
                "prompt_word_count": _word_count(doc.prompt_text),
                "human_word_count": _word_count(doc.human_text),
                "prompt_sha256": hashlib.sha256(doc.prompt_text.encode("utf-8")).hexdigest(),
                "human_sha256": hashlib.sha256(doc.human_text.encode("utf-8")).hexdigest(),
            }
        )
    return rows


def _existing_text_rows(docs: list[HapeDoc], *, dataset_name: str) -> list[dict[str, Any]]:
    rows = []
    for doc in docs:
        for author, text in sorted(doc.texts_by_author.items()):
            text = text.strip()
            if not text:
                continue
            if author == "chunk_1":
                role = "prompt"
                provider = "human"
                model = "human_prompt"
                record_id = doc.prompt_doc_id
            elif author == "chunk_2":
                role = "human_continuation"
                provider = "human"
                model = "human"
                record_id = doc.human_doc_id
            else:
                role = "llm_continuation"
                provider = "original_hape"
                model = author
                record_id = f"{doc.base_doc_id}@{author}"
            rows.append(
                {
                    "record_id": record_id,
                    "base_doc_id": doc.base_doc_id,
                    "source_dataset": dataset_name,
                    "role": role,
                    "provider": provider,
                    "model": model,
                    "generated_by_project": False,
                    "batch_custom_id": None,
                    "batch_id": None,
                    "text": text,
                    "word_count": _word_count(text),
                    "char_count": len(text),
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
    return rows


def _manifest_rows(
    docs: list[HapeDoc],
    *,
    provider: str,
    model: str,
    max_output_tokens: int,
    openai_reasoning_effort: str | None = None,
    gemini_thinking_level: str | None = None,
    anthropic_thinking: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    safe_model = _safe_model_name(model)
    for index, doc in enumerate(docs):
        custom_id = f"hape_{provider}_{safe_model}_{index:05d}_{_short_hash(doc.base_doc_id)}"
        rows.append(
            {
                "custom_id": custom_id,
                "base_doc_id": doc.base_doc_id,
                "provider": provider,
                "model": model,
                "prompt_doc_id": doc.prompt_doc_id,
                "human_doc_id": doc.human_doc_id,
                "prompt_sha256": hashlib.sha256(doc.prompt_text.encode("utf-8")).hexdigest(),
                "max_output_tokens": max_output_tokens,
                "openai_reasoning_effort": openai_reasoning_effort,
                "gemini_thinking_level": gemini_thinking_level,
                "anthropic_thinking": anthropic_thinking,
            }
        )
    return rows


def _openai_request_rows(
    docs: list[HapeDoc],
    manifest_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for doc, manifest in zip(docs, manifest_rows, strict=True):
        rows.append(
            {
                "custom_id": manifest["custom_id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": manifest["model"],
                    "instructions": SYSTEM_PROMPT,
                    "input": _prompt_for(doc.prompt_text),
                    "max_output_tokens": manifest["max_output_tokens"],
                    "reasoning": {
                        "effort": _option_string(
                            manifest.get("openai_reasoning_effort"),
                            default=DEFAULT_OPENAI_REASONING_EFFORT,
                        )
                    },
                    "store": False,
                },
            }
        )
    return rows


def _gemini_request_rows(
    docs: list[HapeDoc],
    manifest_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for doc, manifest in zip(docs, manifest_rows, strict=True):
        rows.append(
            {
                "key": manifest["custom_id"],
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": (
                                        f"{SYSTEM_PROMPT}\n\n"
                                        f"{_prompt_for(doc.prompt_text)}"
                                    )
                                }
                            ],
                        }
                    ],
                    "generationConfig": {
                        "maxOutputTokens": manifest["max_output_tokens"],
                        "thinkingConfig": {
                            "thinkingLevel": _option_string(
                                manifest.get("gemini_thinking_level"),
                                default=DEFAULT_GEMINI_THINKING_LEVEL,
                            )
                        },
                    },
                },
            }
        )
    return rows


def _anthropic_request_rows(
    docs: list[HapeDoc],
    manifest_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for doc, manifest in zip(docs, manifest_rows, strict=True):
        params = {
            "model": manifest["model"],
            "max_tokens": manifest["max_output_tokens"],
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": _prompt_for(doc.prompt_text),
                }
            ],
        }
        thinking = _option_string(
            manifest.get("anthropic_thinking"),
            default=DEFAULT_ANTHROPIC_THINKING,
        )
        if thinking != "disabled":
            raise ValueError("Anthropic HAP-E generation only supports disabled thinking")
        rows.append(
            {
                "custom_id": manifest["custom_id"],
                "params": params,
            }
        )
    return rows


def _estimated_gemini_enqueued_tokens(doc: HapeDoc, *, max_output_tokens: int) -> int:
    request_text = f"{SYSTEM_PROMPT}\n\n{_prompt_for(doc.prompt_text)}"
    word_estimate = int(_word_count(request_text) * 1.45)
    char_estimate = int(len(request_text) / 3)
    return max(word_estimate, char_estimate) + max_output_tokens


def _write_gemini_shards(
    *,
    output_dir: Path,
    docs: list[HapeDoc],
    manifest_rows: list[dict[str, Any]],
    token_budget: int,
) -> list[dict[str, Any]]:
    if token_budget <= 0:
        raise ValueError("--gemini-shard-token-budget must be positive")
    shard_dir = output_dir / "requests" / "gemini_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    request_rows = _gemini_request_rows(docs, manifest_rows)
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

    for index, (doc, manifest, request_row) in enumerate(
        zip(docs, manifest_rows, request_rows, strict=True)
    ):
        estimated_tokens = _estimated_gemini_enqueued_tokens(
            doc, max_output_tokens=int(manifest["max_output_tokens"])
        )
        if current_rows and current_tokens + estimated_tokens > token_budget:
            flush(index)
        current_rows.append(request_row)
        current_tokens += estimated_tokens
    flush(len(request_rows))
    return shard_summaries


def prepare_hape_augmented_batch(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    requests_dir = output_dir / "requests"
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_dir.mkdir(parents=True, exist_ok=True)

    docs = _load_hape_docs(args.dataset, max_docs=args.max_docs)
    base_rows = _base_doc_rows(docs, dataset_name=args.dataset)
    existing_rows = _existing_text_rows(docs, dataset_name=args.dataset)
    openai_manifest = _manifest_rows(
        docs,
        provider="openai",
        model=args.openai_model,
        max_output_tokens=args.max_output_tokens,
        openai_reasoning_effort=args.openai_reasoning_effort,
    )
    gemini_manifest = _manifest_rows(
        docs,
        provider="gemini",
        model=args.gemini_model,
        max_output_tokens=args.max_output_tokens,
        gemini_thinking_level=args.gemini_thinking_level,
    )
    request_manifest = [*openai_manifest, *gemini_manifest]

    pd.DataFrame(base_rows).to_parquet(output_dir / "base_docs.parquet", index=False)
    pd.DataFrame(existing_rows).to_parquet(output_dir / "texts_existing.parquet", index=False)
    pd.DataFrame(request_manifest).to_parquet(output_dir / "request_manifest.parquet", index=False)
    _write_jsonl(requests_dir / "openai_requests.jsonl", _openai_request_rows(docs, openai_manifest))
    _write_jsonl(requests_dir / "gemini_requests.jsonl", _gemini_request_rows(docs, gemini_manifest))
    gemini_shards = _write_gemini_shards(
        output_dir=output_dir,
        docs=docs,
        manifest_rows=gemini_manifest,
        token_budget=args.gemini_shard_token_budget,
    )
    pd.DataFrame(gemini_shards).to_parquet(
        output_dir / "gemini_shard_manifest.parquet", index=False
    )

    summary = {
        "created_at": int(time.time()),
        "dataset": args.dataset,
        "base_docs": len(docs),
        "existing_text_rows": len(existing_rows),
        "openai_model": args.openai_model,
        "gemini_model": args.gemini_model,
        "max_output_tokens": args.max_output_tokens,
        "openai_reasoning_effort": args.openai_reasoning_effort,
        "gemini_thinking_level": args.gemini_thinking_level,
        "openai_requests": len(openai_manifest),
        "gemini_requests": len(gemini_manifest),
        "gemini_shard_token_budget": args.gemini_shard_token_budget,
        "gemini_shards": len(gemini_shards),
        "gemini_shard_max_estimated_enqueued_tokens": max(
            row["estimated_enqueued_tokens"] for row in gemini_shards
        ),
        "source_prompt": HAPE_INSTRUCTION,
        "system_prompt": SYSTEM_PROMPT,
        "outputs": {
            "base_docs": str(output_dir / "base_docs.parquet"),
            "texts_existing": str(output_dir / "texts_existing.parquet"),
            "request_manifest": str(output_dir / "request_manifest.parquet"),
            "openai_requests": str(requests_dir / "openai_requests.jsonl"),
            "gemini_requests": str(requests_dir / "gemini_requests.jsonl"),
            "gemini_shard_manifest": str(output_dir / "gemini_shard_manifest.parquet"),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    print(
        f"Wrote {len(docs)} HAP-E docs, {len(existing_rows)} existing text rows, "
        f"and {len(request_manifest)} provider requests under {output_dir}"
    )


def prepare_anthropic_requests(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    requests_dir = output_dir / "requests"
    base_docs_path = output_dir / "base_docs.parquet"
    manifest_path = output_dir / "request_manifest.parquet"
    if not base_docs_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("missing base_docs.parquet or request_manifest.parquet; run prepare first")
    requests_dir.mkdir(parents=True, exist_ok=True)

    base_docs = pd.read_parquet(base_docs_path)
    docs: list[HapeDoc] = []
    for row in base_docs.itertuples(index=False):
        docs.append(
            HapeDoc(
                base_doc_id=str(row.base_doc_id),
                prompt_doc_id=str(row.prompt_doc_id),
                human_doc_id=str(row.human_doc_id),
                prompt_text=str(row.prompt_text),
                human_text=str(row.human_text),
                texts_by_author={},
            )
        )

    anthropic_manifest = _manifest_rows(
        docs,
        provider="anthropic",
        model=args.anthropic_model,
        max_output_tokens=args.max_output_tokens,
        anthropic_thinking=args.anthropic_thinking,
    )
    request_manifest = pd.read_parquet(manifest_path)
    if len(request_manifest):
        request_manifest = request_manifest[
            ~(
                (request_manifest["provider"].astype(str) == "anthropic")
                & (request_manifest["model"].astype(str) == args.anthropic_model)
            )
        ].copy()
    request_manifest = pd.concat(
        [request_manifest, pd.DataFrame(anthropic_manifest)],
        ignore_index=True,
    )
    request_manifest.to_parquet(manifest_path, index=False)
    _write_jsonl(
        requests_dir / "anthropic_requests.jsonl",
        _anthropic_request_rows(docs, anthropic_manifest),
    )

    summary_path = output_dir / "summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}
    summary.update(
        {
            "anthropic_model": args.anthropic_model,
            "anthropic_requests": len(anthropic_manifest),
            "anthropic_thinking": args.anthropic_thinking,
            "anthropic_max_output_tokens": args.max_output_tokens,
            "outputs": {
                **summary.get("outputs", {}),
                "anthropic_requests": str(requests_dir / "anthropic_requests.jsonl"),
            },
        }
    )
    _write_json(summary_path, summary)
    print(
        f"Wrote {len(anthropic_manifest)} Anthropic requests to "
        f"{requests_dir / 'anthropic_requests.jsonl'} with thinking={args.anthropic_thinking}"
    )


def reshard_gemini_requests(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    base_docs_path = output_dir / "base_docs.parquet"
    manifest_path = output_dir / "request_manifest.parquet"
    if not base_docs_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("missing base_docs.parquet or request_manifest.parquet; run prepare first")

    base_docs = pd.read_parquet(base_docs_path)
    request_manifest = pd.read_parquet(manifest_path)
    gemini_manifest = request_manifest[request_manifest["provider"] == "gemini"]
    docs: list[HapeDoc] = []
    for row in base_docs.itertuples(index=False):
        docs.append(
            HapeDoc(
                base_doc_id=str(row.base_doc_id),
                prompt_doc_id=str(row.prompt_doc_id),
                human_doc_id=str(row.human_doc_id),
                prompt_text=str(row.prompt_text),
                human_text=str(row.human_text),
                texts_by_author={},
            )
        )
    manifest_rows = [dict(row._asdict()) for row in gemini_manifest.itertuples(index=False)]
    gemini_shards = _write_gemini_shards(
        output_dir=output_dir,
        docs=docs,
        manifest_rows=manifest_rows,
        token_budget=args.gemini_shard_token_budget,
    )
    pd.DataFrame(gemini_shards).to_parquet(
        output_dir / "gemini_shard_manifest.parquet", index=False
    )
    summary_path = output_dir / "summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}
    summary.update(
        {
            "gemini_shard_token_budget": args.gemini_shard_token_budget,
            "gemini_shards": len(gemini_shards),
            "gemini_shard_max_estimated_enqueued_tokens": max(
                row["estimated_enqueued_tokens"] for row in gemini_shards
            )
            if gemini_shards
            else 0,
            "outputs": {
                **summary.get("outputs", {}),
                "gemini_shard_manifest": str(output_dir / "gemini_shard_manifest.parquet"),
            },
        }
    )
    _write_json(summary_path, summary)
    print(
        f"Wrote {len(gemini_shards)} Gemini shards under {output_dir / 'requests' / 'gemini_shards'}"
    )


def _load_provider_env(key_name: str) -> str:
    load_dotenv()
    value = os.environ.get(key_name, "")
    if not value:
        raise RuntimeError(f"{key_name} is not set")
    return value


def _request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    json_payload: dict[str, Any] | None = None,
    data: bytes | None = None,
    files: dict[str, Any] | None = None,
    timeout: int = 120,
) -> Any:
    response = requests.request(
        method,
        url,
        headers=headers,
        params=params,
        json=json_payload,
        data=data,
        files=files,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"{method} {url} failed with {response.status_code}: {response.text[:2000]}"
        )
    if not response.content:
        return {}
    return response.json()


def _openai_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _anthropic_headers(api_key: str, *, content_type: str = "application/json") -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": content_type,
    }


def _load_openai_status(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "status" / "openai_batch.json"
    return _read_json(path) if path.exists() else None


def _load_openai_extra_statuses(output_dir: Path) -> list[dict[str, Any]]:
    status_dir = output_dir / "status" / "openai_batches"
    if not status_dir.exists():
        return []
    return [_read_json(path) for path in sorted(status_dir.glob("openai_batch_*.json"))]


def _openai_retry_status_path(output_dir: Path, retry_name: str) -> Path:
    return output_dir / "status" / "openai_batches" / f"openai_batch_{retry_name}.json"


def _load_gemini_status(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "status" / "gemini_batch.json"
    return _read_json(path) if path.exists() else None


def _load_gemini_shard_statuses(output_dir: Path) -> list[dict[str, Any]]:
    status_dir = output_dir / "status" / "gemini_batches"
    statuses = []
    if status_dir.exists():
        statuses.extend(_read_json(path) for path in sorted(status_dir.glob("gemini_batch_*.json")))
    legacy = _load_gemini_status(output_dir)
    if legacy:
        statuses.append(legacy)
    return statuses


def _load_anthropic_status(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "status" / "anthropic_batch.json"
    return _read_json(path) if path.exists() else None


def _gemini_shard_status_path(output_dir: Path, shard_id: int) -> Path:
    return output_dir / "status" / "gemini_batches" / f"gemini_batch_{shard_id:04d}.json"


def _gemini_state(status: dict[str, Any] | None) -> str | None:
    if not status:
        return None
    metadata = status.get("metadata") or {}
    return status.get("state") or metadata.get("state")


def _gemini_done(status: dict[str, Any] | None) -> bool:
    state = _gemini_state(status)
    return bool(state and state in GEMINI_TERMINAL_STATES)


def _gemini_response_file(status: dict[str, Any]) -> str | None:
    metadata = status.get("metadata") or {}
    output_config = metadata.get("outputConfig") or status.get("outputConfig") or {}
    dest = status.get("dest") or {}
    response = status.get("response") or {}
    return (
        output_config.get("responsesFile")
        or output_config.get("fileName")
        or dest.get("fileName")
        or response.get("responsesFile")
    )


def _save_openai_status(output_dir: Path, payload: dict[str, Any]) -> None:
    _write_json(output_dir / "status" / "openai_batch.json", payload)


def _save_gemini_status(output_dir: Path, payload: dict[str, Any]) -> None:
    _write_json(output_dir / "status" / "gemini_batch.json", payload)


def _save_anthropic_status(output_dir: Path, payload: dict[str, Any]) -> None:
    _write_json(output_dir / "status" / "anthropic_batch.json", payload)


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
            "metadata": {"dataset": "hape_augmented_batch", "source": "slop_sftdiv"},
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


def prepare_openai_retry(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    source_path = output_dir / "results" / "openai_output.jsonl"
    request_path = output_dir / "requests" / "openai_requests.jsonl"
    retry_path = output_dir / "requests" / f"{args.retry_name}.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"missing {source_path}; fetch the OpenAI output first")
    if not request_path.exists():
        raise FileNotFoundError(f"missing {request_path}; run prepare first")

    requests_by_custom: dict[str, dict[str, Any]] = {}
    with request_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            requests_by_custom[str(row["custom_id"])] = row

    retry_rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = defaultdict(int)
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = str(row.get("custom_id"))
            body = ((row.get("response") or {}).get("body")) or {}
            text = _openai_text_from_output(row)
            status_counts[str(body.get("status"))] += 1
            if text.strip():
                continue
            original = requests_by_custom.get(custom_id)
            if not original:
                continue
            retry_row = json.loads(json.dumps(original))
            retry_row["body"]["max_output_tokens"] = args.max_output_tokens
            retry_row["body"]["reasoning"] = {"effort": args.reasoning_effort}
            retry_row["body"]["text"] = {"verbosity": "low"}
            retry_rows.append(retry_row)

    _write_jsonl(retry_path, retry_rows)
    _write_json(
        output_dir / "requests" / f"{args.retry_name}_summary.json",
        {
            "retry_name": args.retry_name,
            "source_path": str(source_path),
            "original_request_path": str(request_path),
            "retry_path": str(retry_path),
            "retry_requests": len(retry_rows),
            "source_status_counts": dict(sorted(status_counts.items())),
            "max_output_tokens": args.max_output_tokens,
            "reasoning_effort": args.reasoning_effort,
        },
    )
    print(f"Wrote {len(retry_rows)} OpenAI retry requests to {retry_path}")


def submit_openai_retry_batch(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    request_path = output_dir / "requests" / f"{args.retry_name}.jsonl"
    if not request_path.exists():
        raise FileNotFoundError(f"missing {request_path}; run prepare-openai-retry first")
    status_path = _openai_retry_status_path(output_dir, args.retry_name)
    existing = _read_json(status_path) if status_path.exists() else None
    if existing and existing.get("status") not in OPENAI_TERMINAL_STATUSES and not args.force:
        print(f"OpenAI retry batch already active: {existing.get('id')} status={existing.get('status')}")
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
            "metadata": {
                "dataset": "hape_augmented_batch",
                "source": "slop_sftdiv",
                "retry_name": args.retry_name,
            },
        },
    )
    status = {
        **batch_payload,
        "submitted_at": int(time.time()),
        "input_file": file_payload,
        "request_path": str(request_path),
        "retry_name": args.retry_name,
    }
    _write_json(status_path, status)
    print(f"Submitted OpenAI retry batch {batch_payload.get('id')} status={batch_payload.get('status')}")


def submit_anthropic_batch(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    request_path = output_dir / "requests" / "anthropic_requests.jsonl"
    if not request_path.exists():
        raise FileNotFoundError(f"missing {request_path}; run prepare-anthropic first")
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


def _gemini_headers(api_key: str, *, content_type: str = "application/json") -> dict[str, str]:
    return {"x-goog-api-key": api_key, "Content-Type": content_type}


def submit_gemini_batch(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    shard_manifest_path = output_dir / "gemini_shard_manifest.parquet"
    if shard_manifest_path.exists():
        _submit_next_gemini_shard(args)
        return

    request_path = output_dir / "requests" / "gemini_requests.jsonl"
    if not request_path.exists():
        raise FileNotFoundError(f"missing {request_path}; run prepare first")
    existing = _load_gemini_status(output_dir)
    existing_state = _gemini_state(existing)
    if existing and not _gemini_done(existing) and not args.force:
        print(f"Gemini batch already active: {existing.get('name')} state={existing_state}")
        return
    api_key = _load_provider_env("GEMINI_API_KEY")
    summary = _read_json(output_dir / "summary.json")
    model = summary["gemini_model"]
    status = _submit_gemini_request_file(
        api_key=api_key,
        model=model,
        request_path=request_path,
        display_name=f"hape-augmented-{_safe_model_name(model)}",
    )
    status = {**status, "request_path": str(request_path)}
    _save_gemini_status(output_dir, status)
    print(f"Submitted Gemini batch {status.get('name')} state={_gemini_state(status)}")


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

    shard_manifest = pd.read_parquet(output_dir / "gemini_shard_manifest.parquet")
    submitted_by_shard = {
        int(status["shard_id"]): status
        for status in _load_gemini_shard_statuses(output_dir)
        if status.get("shard_id") is not None
    }
    next_row = None
    for row in shard_manifest.itertuples(index=False):
        status = submitted_by_shard.get(int(row.shard_id))
        if status is None:
            next_row = row
            break
        if _gemini_state(status) in {"BATCH_STATE_FAILED", "BATCH_STATE_CANCELLED", "BATCH_STATE_EXPIRED"}:
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
        display_name=f"hape-augmented-{_safe_model_name(model)}-shard-{int(next_row.shard_id):04d}",
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
                "displayName": f"hape-augmented-{_safe_model_name(model)}",
                "inputConfig": {"fileName": file_name},
            }
        },
    )
    return {
        **batch_payload,
        "submitted_at": int(time.time()),
        "input_file": upload_response,
    }


def _refresh_openai(output_dir: Path, api_key: str) -> dict[str, Any] | None:
    status = _load_openai_status(output_dir)
    if not status or not status.get("id"):
        return status
    merged = _refresh_openai_status(status, api_key)
    _save_openai_status(output_dir, merged)
    return merged


def _refresh_openai_status(status: dict[str, Any], api_key: str) -> dict[str, Any]:
    refreshed = _request_json(
        "GET",
        f"https://api.openai.com/v1/batches/{status['id']}",
        headers={**_openai_headers(api_key), "Content-Type": "application/json"},
    )
    return {**status, **refreshed, "last_checked_at": int(time.time())}


def _refresh_gemini(output_dir: Path, api_key: str) -> dict[str, Any] | None:
    status = _load_gemini_status(output_dir)
    if not status or not status.get("name"):
        return status
    refreshed = _refresh_gemini_status(status, api_key)
    _save_gemini_status(output_dir, refreshed)
    return refreshed


def _refresh_gemini_status(status: dict[str, Any], api_key: str) -> dict[str, Any]:
    refreshed = _request_json(
        "GET",
        f"https://generativelanguage.googleapis.com/v1beta/{status['name']}",
        headers=_gemini_headers(api_key),
    )
    merged = {**status, **refreshed, "last_checked_at": int(time.time())}
    return merged


def _refresh_anthropic(output_dir: Path, api_key: str) -> dict[str, Any] | None:
    status = _load_anthropic_status(output_dir)
    if not status or not status.get("id"):
        return status
    refreshed = _refresh_anthropic_status(status, api_key)
    _save_anthropic_status(output_dir, refreshed)
    return refreshed


def _refresh_anthropic_status(status: dict[str, Any], api_key: str) -> dict[str, Any]:
    refreshed = _request_json(
        "GET",
        f"https://api.anthropic.com/v1/messages/batches/{status['id']}",
        headers=_anthropic_headers(api_key),
    )
    return {**status, **refreshed, "last_checked_at": int(time.time())}


def _anthropic_batch_request_count(batch: dict[str, Any]) -> int:
    counts = batch.get("request_counts") or {}
    if not counts:
        return int(batch.get("request_count") or 0)
    return sum(int(counts.get(key) or 0) for key in counts)


def status_batches(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    load_dotenv()
    openai: dict[str, Any] | list[dict[str, Any]] | None = None
    gemini: dict[str, Any] | list[dict[str, Any]] | None = None
    anthropic: dict[str, Any] | None = None
    if os.environ.get("OPENAI_API_KEY"):
        openai_statuses = []
        main_openai = _refresh_openai(output_dir, os.environ["OPENAI_API_KEY"])
        if main_openai:
            openai_statuses.append(main_openai)
        for status in _load_openai_extra_statuses(output_dir):
            refreshed = _refresh_openai_status(status, os.environ["OPENAI_API_KEY"])
            retry_name = str(refreshed.get("retry_name") or (refreshed.get("metadata") or {}).get("retry_name") or "")
            if retry_name:
                _write_json(_openai_retry_status_path(output_dir, retry_name), refreshed)
            openai_statuses.append(refreshed)
        openai = openai_statuses if len(openai_statuses) > 1 else (openai_statuses[0] if openai_statuses else None)
    elif _load_openai_status(output_dir):
        openai_statuses = [_load_openai_status(output_dir), *_load_openai_extra_statuses(output_dir)]
        openai = openai_statuses if len(openai_statuses) > 1 else openai_statuses[0]
    shard_status_dir = output_dir / "status" / "gemini_batches"
    if shard_status_dir.exists():
        if os.environ.get("GEMINI_API_KEY"):
            gemini = _refresh_gemini_shards(output_dir, os.environ["GEMINI_API_KEY"])
        else:
            gemini = _load_gemini_shard_statuses(output_dir)
    elif os.environ.get("GEMINI_API_KEY"):
        gemini = _refresh_gemini(output_dir, os.environ["GEMINI_API_KEY"])
    elif _load_gemini_status(output_dir):
        gemini = _load_gemini_status(output_dir)
    if os.environ.get("ANTHROPIC_API_KEY"):
        anthropic = _refresh_anthropic(output_dir, os.environ["ANTHROPIC_API_KEY"])
    elif _load_anthropic_status(output_dir):
        anthropic = _load_anthropic_status(output_dir)

    print(
        json.dumps(
            {
                "anthropic": _anthropic_status_payload(anthropic),
                "openai": _openai_status_payload(openai),
                "gemini": _gemini_status_payload(gemini),
            },
            indent=2,
            sort_keys=True,
        )
    )


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


def _gemini_status_payload(
    gemini: dict[str, Any] | list[dict[str, Any]] | None,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    if isinstance(gemini, list):
        return [_status_summary(status, provider="gemini") for status in gemini]
    return _status_summary(gemini, provider="gemini")


def _openai_status_payload(
    openai: dict[str, Any] | list[dict[str, Any]] | None,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    if isinstance(openai, list):
        return [_status_summary(status, provider="openai") for status in openai]
    return _status_summary(openai, provider="openai")


def _anthropic_status_payload(anthropic: dict[str, Any] | None) -> dict[str, Any] | None:
    return _status_summary(anthropic, provider="anthropic")


def _status_summary(status: dict[str, Any] | None, *, provider: str) -> dict[str, Any] | None:
    if not status:
        return None
    if provider == "openai":
        return {
            "id": status.get("id"),
            "retry_name": status.get("retry_name") or (status.get("metadata") or {}).get("retry_name"),
            "status": status.get("status"),
            "request_counts": status.get("request_counts"),
            "output_file_id": status.get("output_file_id"),
            "error_file_id": status.get("error_file_id"),
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
        "state": _gemini_state(status),
        "shard_id": status.get("shard_id"),
        "request_count": status.get("request_count"),
        "estimated_enqueued_tokens": status.get("estimated_enqueued_tokens"),
        "batch_stats": (status.get("metadata") or {}).get("batchStats") or status.get("batchStats"),
        "response_file": _gemini_response_file(status),
    }


def fetch_completed_batches(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv()

    if os.environ.get("OPENAI_API_KEY"):
        openai_status = _refresh_openai(output_dir, os.environ["OPENAI_API_KEY"])
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
        for status in _load_openai_extra_statuses(output_dir):
            refreshed = _refresh_openai_status(status, os.environ["OPENAI_API_KEY"])
            retry_name = str(refreshed.get("retry_name") or (refreshed.get("metadata") or {}).get("retry_name") or "")
            if retry_name:
                _write_json(_openai_retry_status_path(output_dir, retry_name), refreshed)
            if refreshed.get("status") == "completed" and retry_name:
                _download_openai_file(
                    os.environ["OPENAI_API_KEY"],
                    refreshed.get("output_file_id"),
                    results_dir / f"openai_output_{retry_name}.jsonl",
                    required=False,
                )
                _download_openai_file(
                    os.environ["OPENAI_API_KEY"],
                    refreshed.get("error_file_id"),
                    results_dir / f"openai_errors_{retry_name}.jsonl",
                    required=False,
                )

    if os.environ.get("GEMINI_API_KEY"):
        if (output_dir / "status" / "gemini_batches").exists():
            for status in _refresh_gemini_shards(output_dir, os.environ["GEMINI_API_KEY"]):
                state = _gemini_state(status)
                if state in {"JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"}:
                    shard_id = status.get("shard_id")
                    if shard_id is None:
                        continue
                    file_name = _gemini_response_file(status)
                    if file_name:
                        _download_gemini_file(
                            os.environ["GEMINI_API_KEY"],
                            file_name,
                            results_dir / f"gemini_output_{int(shard_id):04d}.jsonl",
                        )
        else:
            gemini_status = _refresh_gemini(output_dir, os.environ["GEMINI_API_KEY"])
            state = _gemini_state(gemini_status)
            if gemini_status and state in {"JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"}:
                file_name = _gemini_response_file(gemini_status)
                if file_name:
                    _download_gemini_file(
                        os.environ["GEMINI_API_KEY"],
                        file_name,
                        results_dir / "gemini_output.jsonl",
                    )

    if os.environ.get("ANTHROPIC_API_KEY"):
        anthropic_status = _refresh_anthropic(output_dir, os.environ["ANTHROPIC_API_KEY"])
        if anthropic_status and anthropic_status.get("processing_status") == "ended":
            _download_anthropic_results(
                os.environ["ANTHROPIC_API_KEY"],
                str(anthropic_status["id"]),
                results_dir / "anthropic_output.jsonl",
            )


def _download_openai_file(
    api_key: str,
    file_id: str | None,
    output_path: Path,
    *,
    required: bool = True,
) -> None:
    if not file_id:
        if required:
            raise RuntimeError(f"missing OpenAI file id for {output_path}")
        return
    response = requests.get(
        f"https://api.openai.com/v1/files/{file_id}/content",
        headers=_openai_headers(api_key),
        timeout=300,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"OpenAI file download failed with {response.status_code}: {response.text[:2000]}"
        )
    output_path.write_bytes(response.content)
    print(f"Wrote {output_path}")


def _download_gemini_file(api_key: str, file_name: str, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    response = requests.get(
        f"https://generativelanguage.googleapis.com/download/v1beta/{file_name}:download",
        params={"alt": "media"},
        headers={"x-goog-api-key": api_key},
        timeout=300,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Gemini result download failed with {response.status_code}: {response.text[:2000]}"
        )
    output_path.write_bytes(response.content)
    print(f"Wrote {output_path}")


def _download_anthropic_results(api_key: str, batch_id: str, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    response = requests.get(
        f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results",
        headers=_anthropic_headers(api_key),
        timeout=300,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Anthropic result download failed with {response.status_code}: {response.text[:2000]}"
        )
    output_path.write_bytes(response.content)
    print(f"Wrote {output_path}")


def _gemini_shard_manifest(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "gemini_shard_manifest.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run prepare or reshard-gemini first")
    return pd.read_parquet(path)


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
        if status is None:
            pending.append(int(row.shard_id))
            continue
        if _gemini_state(status) in {
            "BATCH_STATE_FAILED",
            "BATCH_STATE_CANCELLED",
            "BATCH_STATE_EXPIRED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }:
            pending.append(int(row.shard_id))
    return pending


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
        done_count = sum(1 for status in statuses if _gemini_state(status) == "BATCH_STATE_SUCCEEDED")
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
        status = _refresh_anthropic(output_dir, api_key)
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


def _openai_text_from_output(row: dict[str, Any]) -> str:
    body = (((row.get("response") or {}).get("body")) or {})
    choices = body.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(str(part.get("text", "")) for part in content).strip()
    output = body.get("output") or []
    pieces = []
    for item in output:
        for part in item.get("content", []) or []:
            if part.get("type") in {"output_text", "text"}:
                pieces.append(str(part.get("text", "")))
    return "\n".join(pieces).strip()


def _gemini_text_from_output(row: dict[str, Any]) -> str:
    response = row.get("response") or {}
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    return "\n".join(str(part.get("text", "")) for part in parts if part.get("text")).strip()


def _anthropic_text_from_output(row: dict[str, Any]) -> str:
    result = row.get("result") or {}
    if result.get("type") != "succeeded":
        return ""
    message = result.get("message") or {}
    pieces = [
        str(part.get("text", ""))
        for part in message.get("content", []) or []
        if part.get("type") == "text" and part.get("text")
    ]
    return "\n".join(pieces).strip()


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


def _collect_generated_text_rows(
    *,
    output_dir: Path,
    manifest: pd.DataFrame,
    source_dataset: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results_dir = output_dir / "results"
    generated_rows: list[dict[str, Any]] = []
    manifest_by_custom = {
        str(row.custom_id): row._asdict() for row in manifest.itertuples(index=False)
    }
    parse_counts: Counter[str] = Counter()

    seen_openai_custom_ids: set[str] = set()
    for openai_path in sorted(results_dir.glob("openai_output*.jsonl"), reverse=True):
        parse_counts["openai_result_files"] += 1
        for row in _iter_jsonl_rows(openai_path):
            parse_counts["openai_result_rows"] += 1
            custom_id = str(row.get("custom_id"))
            if custom_id in seen_openai_custom_ids:
                parse_counts["openai_duplicate_rows"] += 1
                continue
            meta = manifest_by_custom.get(custom_id)
            if not meta or meta.get("provider") != "openai":
                parse_counts["openai_unknown_custom_id"] += 1
                continue
            if row.get("error"):
                parse_counts["openai_error_rows"] += 1
                continue
            text = _openai_text_from_output(row)
            if not text:
                parse_counts["openai_empty_text_rows"] += 1
                continue
            generated_rows.append(
                _generated_text_row(
                    meta,
                    text,
                    batch_id=row.get("id"),
                    source_dataset=source_dataset,
                )
            )
            seen_openai_custom_ids.add(custom_id)

    seen_gemini_custom_ids: set[str] = set()
    for gemini_path in sorted(results_dir.glob("gemini_output*.jsonl")):
        parse_counts["gemini_result_files"] += 1
        for row in _iter_jsonl_rows(gemini_path):
            parse_counts["gemini_result_rows"] += 1
            custom_id = str(row.get("key") or ((row.get("metadata") or {}).get("key")))
            if custom_id in seen_gemini_custom_ids:
                parse_counts["gemini_duplicate_rows"] += 1
                continue
            meta = manifest_by_custom.get(custom_id)
            if not meta or meta.get("provider") != "gemini":
                parse_counts["gemini_unknown_custom_id"] += 1
                continue
            if row.get("error"):
                parse_counts["gemini_error_rows"] += 1
                continue
            text = _gemini_text_from_output(row)
            if not text:
                parse_counts["gemini_empty_text_rows"] += 1
                continue
            generated_rows.append(
                _generated_text_row(
                    meta,
                    text,
                    batch_id=((row.get("response") or {}).get("responseId")),
                    source_dataset=source_dataset,
                )
            )
            seen_gemini_custom_ids.add(custom_id)

    seen_anthropic_custom_ids: set[str] = set()
    for anthropic_path in sorted(results_dir.glob("anthropic_output*.jsonl"), reverse=True):
        parse_counts["anthropic_result_files"] += 1
        for row in _iter_jsonl_rows(anthropic_path):
            parse_counts["anthropic_result_rows"] += 1
            custom_id = str(row.get("custom_id"))
            if custom_id in seen_anthropic_custom_ids:
                parse_counts["anthropic_duplicate_rows"] += 1
                continue
            meta = manifest_by_custom.get(custom_id)
            if not meta or meta.get("provider") != "anthropic":
                parse_counts["anthropic_unknown_custom_id"] += 1
                continue
            result = row.get("result") or {}
            if result.get("type") != "succeeded":
                parse_counts[f"anthropic_{result.get('type') or 'unknown'}_rows"] += 1
                continue
            text = _anthropic_text_from_output(row)
            if not text:
                parse_counts["anthropic_empty_text_rows"] += 1
                continue
            generated_rows.append(
                _generated_text_row(
                    meta,
                    text,
                    batch_id=((result.get("message") or {}).get("id")),
                    source_dataset=source_dataset,
                )
            )
            seen_anthropic_custom_ids.add(custom_id)

    return generated_rows, dict(sorted(parse_counts.items()))


def _group_counts(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if len(frame) == 0:
        return []
    return frame.groupby(columns, dropna=False).size().reset_index(name="rows").to_dict("records")


def _build_augmented_frames(
    *,
    output_dir: Path,
    require_complete: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    existing_path = output_dir / "texts_existing.parquet"
    manifest_path = output_dir / "request_manifest.parquet"
    if not existing_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("missing prepared parquet files; run prepare first")

    existing = pd.read_parquet(existing_path)
    manifest = pd.read_parquet(manifest_path)
    summary_path = output_dir / "summary.json"
    prepared_summary = _read_json(summary_path) if summary_path.exists() else {}
    source_dataset = str(prepared_summary.get("dataset") or HAPE_DATASET)

    generated_rows, parse_summary = _collect_generated_text_rows(
        output_dir=output_dir,
        manifest=manifest,
        source_dataset=source_dataset,
    )
    output_columns = list(existing.columns)
    generated = pd.DataFrame(generated_rows)
    if len(generated):
        generated = generated.reindex(columns=output_columns)
    else:
        generated = pd.DataFrame(columns=output_columns)
    combined = pd.concat([existing, generated], ignore_index=True)
    combined = combined.drop_duplicates(subset=["record_id"], keep="last")
    completed_custom_ids = set(generated["batch_custom_id"].dropna().astype(str))
    manifest_custom_ids = manifest["custom_id"].astype(str)
    missing_manifest = manifest[~manifest_custom_ids.isin(completed_custom_ids)]
    compile_summary = {
        "source_dataset": source_dataset,
        "base_docs": int(existing["base_doc_id"].nunique()) if "base_doc_id" in existing else 0,
        "existing_rows": int(len(existing)),
        "expected_generation_requests": int(len(manifest)),
        "generated_rows": int(len(generated)),
        "missing_generation_requests": int(len(missing_manifest)),
        "combined_rows": int(len(combined)),
        "expected_generation_requests_by_provider": _group_counts(
            manifest, ["provider", "model"]
        ),
        "generated_by_provider": _group_counts(generated, ["provider", "model"]),
        "missing_generation_requests_by_provider": _group_counts(
            missing_manifest, ["provider", "model"]
        ),
        "result_parse_summary": parse_summary,
    }
    if require_complete and len(missing_manifest):
        raise RuntimeError(
            "compiled dataset is incomplete: "
            f"{len(missing_manifest)} of {len(manifest)} provider requests are missing "
            "a parsed generation"
        )
    return existing, generated, combined, compile_summary


def finalize_augmented_dataset(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    _, generated, combined, summary = _build_augmented_frames(
        output_dir=output_dir,
        require_complete=False,
    )
    generated_path = output_dir / "texts_generated.parquet"
    dataset_path = output_dir / "hape_augmented_texts.parquet"
    generated.to_parquet(generated_path, index=False)
    combined.to_parquet(dataset_path, index=False)
    _write_json(
        output_dir / "finalize_summary.json",
        {**summary, "generated_path": str(generated_path), "dataset_path": str(dataset_path)},
    )
    print(f"Wrote {len(combined)} rows to {dataset_path}")


def compile_hf_dataset(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    hf_output_dir: Path = args.hf_output_dir or output_dir / "hf_dataset"
    data_dir = hf_output_dir / "data"
    metadata_dir = hf_output_dir / "metadata"
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    _, generated, combined, summary = _build_augmented_frames(
        output_dir=output_dir,
        require_complete=args.require_complete,
    )
    base_docs_path = output_dir / "base_docs.parquet"
    if not base_docs_path.exists():
        raise FileNotFoundError(f"missing {base_docs_path}; run prepare first")
    base_docs = pd.read_parquet(base_docs_path)

    dataset_path = data_dir / "train.parquet"
    base_docs_output = metadata_dir / "base_docs.parquet"
    generated_output = metadata_dir / "generated_texts.parquet"
    combined.to_parquet(dataset_path, index=False)
    base_docs.to_parquet(base_docs_output, index=False)
    generated.to_parquet(generated_output, index=False)

    dataset_summary = {
        **summary,
        "created_at": int(time.time()),
        "hf_output_dir": str(hf_output_dir),
        "dataset_path": str(dataset_path),
        "base_docs_path": str(base_docs_output),
        "generated_path": str(generated_output),
        "columns": list(combined.columns),
    }
    _write_json(hf_output_dir / "dataset_summary.json", dataset_summary)
    _write_hf_readme(hf_output_dir / "README.md", dataset_summary)
    print(
        f"Wrote HF-ready HAP-E dataset to {hf_output_dir} "
        f"({len(combined)} rows, {summary['missing_generation_requests']} missing generations)"
    )


def _write_hf_readme(path: Path, summary: dict[str, Any]) -> None:
    provider_rows = summary.get("generated_by_provider") or []
    provider_table = ["| provider | model | rows |", "| --- | --- | --- |"]
    if provider_rows:
        for row in provider_rows:
            provider_table.append(
                f"| {row.get('provider', '')} | {row.get('model', '')} | {row.get('rows', 0)} |"
            )
    else:
        provider_table.append("|  |  | 0 |")
    lines = [
        "---",
        "configs:",
        "- config_name: default",
        "  data_files:",
        "  - split: train",
        "    path: data/train.parquet",
        "---",
        "",
        "# Augmented HAP-E",
        "",
        "Long-table version of HAP-E with original prompt and continuation rows plus "
        "project-generated model continuations.",
        "",
        "## Main File",
        "",
        "`data/train.parquet` contains one text row per prompt, human continuation, "
        "or LLM continuation.",
        "",
        "Key columns: `record_id`, `base_doc_id`, `role`, `provider`, `model`, "
        "`generated_by_project`, `text`, `word_count`, `char_count`, `text_sha256`.",
        "",
        "Roles: `prompt`, `human_continuation`, and `llm_continuation`.",
        "",
        "## Generation Coverage",
        "",
        f"- Existing rows: {summary.get('existing_rows', 0)}",
        f"- Expected provider generations: {summary.get('expected_generation_requests', 0)}",
        f"- Parsed provider generations: {summary.get('generated_rows', 0)}",
        f"- Missing provider generations: {summary.get('missing_generation_requests', 0)}",
        "",
        "## Generated Rows By Provider",
        "",
        *provider_table,
        "",
        "## Metadata",
        "",
        "`metadata/base_docs.parquet` stores prompt and human-continuation text by base "
        "document. `metadata/generated_texts.parquet` stores only project-generated "
        "continuations. `dataset_summary.json` records coverage and paths.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generated_text_row(
    meta: dict[str, Any],
    text: str,
    *,
    batch_id: str | None,
    source_dataset: str = HAPE_DATASET,
) -> dict[str, Any]:
    return {
        "record_id": f"{meta['base_doc_id']}@{meta['model']}",
        "base_doc_id": meta["base_doc_id"],
        "source_dataset": source_dataset,
        "role": "llm_continuation",
        "provider": meta["provider"],
        "model": meta["model"],
        "generated_by_project": True,
        "batch_custom_id": meta["custom_id"],
        "batch_id": batch_id,
        "text": text,
        "word_count": _word_count(text),
        "char_count": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
