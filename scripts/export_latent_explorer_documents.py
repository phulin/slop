from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export prompt and full completion text for latent explorer examples."
    )
    parser.add_argument("--examples-csv", type=Path, required=True)
    parser.add_argument("--dataset-parquet", type=Path, required=True)
    parser.add_argument("--assistant-turns-parquet", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _string(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def _prompt_text(messages_json: str) -> tuple[str, list[dict[str, str]]]:
    try:
        raw_messages = json.loads(messages_json)
    except json.JSONDecodeError:
        return messages_json, []
    messages: list[dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        role = _string(message.get("role")).strip() or "message"
        content = _string(message.get("content")).strip()
        messages.append({"role": role, "content": content})
    if len(messages) == 1 and messages[0]["role"] == "user":
        return messages[0]["content"], messages
    rendered = "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in messages
    )
    return rendered, messages


def export_documents(args: argparse.Namespace) -> dict[str, Any]:
    examples = pd.read_csv(args.examples_csv, usecols=["doc_id"])
    wanted = set(examples["doc_id"].fillna("").astype(str))
    dataset = pd.read_parquet(args.dataset_parquet)
    turns = pd.read_parquet(args.assistant_turns_parquet)
    turns_by_id = {str(row.turn_id): row._asdict() for row in turns.itertuples(index=False)}
    example_rows = dataset[dataset["record_id"].fillna("").astype(str).isin(wanted)].copy()
    wanted_turns = set(example_rows["turn_id"].fillna("").astype(str))
    dataset = dataset[dataset["turn_id"].fillna("").astype(str).isin(wanted_turns)].copy()
    documents: dict[str, dict[str, Any]] = {}
    prompt_groups: dict[str, list[str]] = {}
    for row in dataset.itertuples(index=False):
        record_id = _string(getattr(row, "record_id")).strip()
        turn_id = _string(getattr(row, "turn_id")).strip()
        turn = turns_by_id.get(turn_id, {})
        prompt, messages = _prompt_text(_string(turn.get("input_messages_json", "")))
        documents[record_id] = {
            "doc_id": record_id,
            "turn_id": turn_id,
            "base_doc_id": _string(getattr(row, "base_doc_id", "")),
            "source_dataset": _string(getattr(row, "source_dataset", "")),
            "source_split": _string(getattr(row, "source_split", "")),
            "source_row_index": int(getattr(row, "source_row_index", -1)),
            "category": _string(getattr(row, "category", turn.get("category", ""))),
            "role": _string(getattr(row, "role", "")),
            "provider": _string(getattr(row, "provider", "")),
            "model": _string(getattr(row, "model", "")),
            "prompt": prompt,
            "messages": messages,
            "text": _string(getattr(row, "text", "")),
            "word_count": int(getattr(row, "word_count", 0)),
            "char_count": int(getattr(row, "char_count", 0)),
            "token_len": int(getattr(row, "token_len", 0)),
        }
        prompt_groups.setdefault(turn_id, []).append(record_id)
    model_order = {"human": 0, "gpt-5.5": 1, "gemini-3.5-flash": 2, "accounts/fireworks/models/glm-5p2": 3}
    for turn_id, doc_ids in prompt_groups.items():
        prompt_groups[turn_id] = sorted(
            doc_ids,
            key=lambda doc_id: (
                model_order.get(str(documents[doc_id]["model"]), 999),
                str(documents[doc_id]["model"]),
            ),
        )
    missing = sorted(wanted - set(documents))
    payload = {"documents": documents, "promptGroups": prompt_groups}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {
        "documents": len(documents),
        "prompt_groups": len(prompt_groups),
        "missing": len(missing),
        "output": str(args.output_json),
    }


def main() -> None:
    summary = export_documents(build_parser().parse_args())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
