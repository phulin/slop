from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from slop_sftdiv.cli.no_robots_batch import (
    DEFAULT_DATA_DIR,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MIN_OUTPUT_TOKENS,
    _load_assistant_turn_rows,
    _strip_markdown_formatting,
    _turn_prompt_or_messages,
)


BASE_URL = "https://api.neuralwatt.com/v1"
WORD_RE = re.compile(r"\b[\w'’-]+\b")
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _messages(turn: dict[str, Any]) -> list[dict[str, str]]:
    is_prompt, payload = _turn_prompt_or_messages(turn)
    if is_prompt:
        return [{"role": "user", "content": str(payload)}]
    return payload  # type: ignore[return-value]


def _content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    return str(((choices[0].get("message") or {}).get("content")) or "").strip()


def _get_json(path: str, api_key: str, *, max_attempts: int = 8) -> dict[str, Any]:
    for attempt in range(1, max_attempts + 1):
        response = requests.get(
            f"{BASE_URL}/{path}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        if response.status_code != 429 or attempt == max_attempts:
            response.raise_for_status()
            return response.json()
        retry_after = response.headers.get("retry-after")
        try:
            sleep_seconds = float(retry_after) if retry_after else 15.0
        except ValueError:
            sleep_seconds = 15.0
        time.sleep(sleep_seconds)
    raise RuntimeError(f"failed to fetch {path}")


def _usage_totals(summary: dict[str, Any]) -> dict[str, float]:
    totals = summary.get("totals") or {}
    return {
        "requests": float(totals.get("requests") or 0),
        "total_tokens": float(totals.get("total_tokens") or 0),
        "prompt_tokens": float(totals.get("prompt_tokens") or 0),
        "completion_tokens": float(totals.get("completion_tokens") or 0),
        "total_cost_usd": float(totals.get("total_cost_usd") or 0),
        "energy_kwh_consumed": float(
            totals.get("energy_kwh_consumed") or totals.get("energy_kwh") or 0
        ),
        "energy_kwh_charged": float(totals.get("energy_kwh_charged") or 0),
    }


def _request_turn(
    *,
    turn: dict[str, Any],
    index: int,
    api_key: str,
    model: str,
    max_attempts: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": _messages(turn),
        "max_tokens": int(turn["max_output_tokens"]),
        "temperature": 1.0,
        "reasoning_effort": "none",
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    attempt = 0
    while True:
        attempt += 1
        started = time.time()
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=headers,
            json=body,
            timeout=240,
        )
        elapsed = time.time() - started
        try:
            payload = response.json()
        except Exception:
            payload = {"raw_text": response.text}
        retryable = response.status_code in RETRYABLE_STATUS_CODES
        if not retryable or attempt >= max_attempts:
            break
        retry_after = response.headers.get("retry-after")
        try:
            sleep_seconds = float(retry_after) if retry_after else 20.0
        except ValueError:
            sleep_seconds = 20.0
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, dict) and detail.get("error") == "concurrent_request_limit_exceeded":
            sleep_seconds = max(sleep_seconds, 45.0)
        else:
            sleep_seconds = max(sleep_seconds, min(180.0, 10.0 * attempt))
        time.sleep(sleep_seconds)

    if response.status_code >= 400:
        return {
            "index": index,
            "turn_id": turn["turn_id"],
            "status_code": response.status_code,
            "attempts": attempt,
            "error": payload,
            "wall_seconds": elapsed,
        }

    raw_text = _content(payload)
    cleaned_text = _strip_markdown_formatting(raw_text)
    usage = payload.get("usage") or {}
    energy = payload.get("energy") or {}
    choice = (payload.get("choices") or [{}])[0]
    return {
        "index": index,
        "turn_id": turn["turn_id"],
        "prompt_id": turn["prompt_id"],
        "split": turn["split"],
        "source_row_index": turn["source_row_index"],
        "assistant_turn_ordinal": turn["assistant_turn_ordinal"],
        "target_words": int(turn["target_word_count"]),
        "status_code": response.status_code,
        "attempts": attempt,
        "id": payload.get("id"),
        "model": payload.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        ),
        "energy_joules": energy.get("energy_joules"),
        "energy_kwh": energy.get("energy_kwh"),
        "duration_seconds_reported": energy.get("duration_seconds"),
        "measurement_available": energy.get("measurement_available"),
        "attribution_method": energy.get("attribution_method"),
        "wall_seconds": elapsed,
        "raw_words": _word_count(raw_text),
        "cleaned_words": _word_count(cleaned_text),
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
    }


def _write_summary(
    *,
    output_path: Path,
    before_summary: dict[str, Any],
    before_quota: dict[str, Any],
    api_key: str,
    total_turns: int,
    started_at: float,
) -> None:
    rows = _jsonl_rows(output_path)
    successes = [row for row in rows if not row.get("error")]
    errors = [row for row in rows if row.get("error")]
    after_summary = _get_json("usage/summary", api_key)
    after_quota = _get_json("quota", api_key)
    before_totals = _usage_totals(before_summary)
    after_totals = _usage_totals(after_summary)
    delta = {key: after_totals[key] - before_totals.get(key, 0.0) for key in after_totals}
    summary = {
        "updated_at": int(time.time()),
        "total_turns": total_turns,
        "rows_written": len(rows),
        "requests_succeeded": len(successes),
        "requests_failed": len(errors),
        "remaining": total_turns - len({int(row["index"]) for row in rows}),
        "input_tokens_response_sum": sum(int(row.get("prompt_tokens") or 0) for row in successes),
        "output_tokens_response_sum": sum(
            int(row.get("completion_tokens") or 0) for row in successes
        ),
        "total_tokens_response_sum": sum(int(row.get("total_tokens") or 0) for row in successes),
        "energy_kwh_response_sum": sum(float(row.get("energy_kwh") or 0) for row in successes),
        "energy_joules_response_sum": sum(
            float(row.get("energy_joules") or 0) for row in successes
        ),
        "usage_delta": delta,
        "finish_reasons": {
            str(reason): sum(row.get("finish_reason") == reason for row in successes)
            for reason in sorted({row.get("finish_reason") for row in successes}, key=str)
        },
        "wall_seconds": time.time() - started_at,
        "before_quota": before_quota,
        "after_quota": after_quota,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--summary-every", type=int, default=25)
    args = parser.parse_args()

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    load_dotenv("/home/user/slop/.env")
    api_key = os.environ["NEURALWATT_API_KEY"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    before_summary = _get_json("usage/summary", api_key)
    before_quota = _get_json("quota", api_key)

    turns = _load_assistant_turn_rows(
        DEFAULT_DATA_DIR,
        max_rows_per_split=None,
        min_output_tokens=DEFAULT_MIN_OUTPUT_TOKENS,
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    indexed_turns = [(index, turn) for index, turn in enumerate(turns, start=1)]
    indexed_turns = [(index, turn) for index, turn in indexed_turns if index >= args.start_index]
    if args.max_requests > 0:
        indexed_turns = indexed_turns[: args.max_requests]

    completed = {int(row["index"]) for row in _jsonl_rows(args.output)}
    pending = [(index, turn) for index, turn in indexed_turns if index not in completed]
    lock = threading.Lock()
    print(
        json.dumps(
            {
                "event": "start",
                "output": str(args.output),
                "model": args.model,
                "concurrency": args.concurrency,
                "total_turns": len(turns),
                "eligible_after_start_and_limit": len(indexed_turns),
                "already_completed": len(completed),
                "pending": len(pending),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    submitted = 0
    completed_this_run = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {}
        while pending or futures:
            while pending and len(futures) < args.concurrency:
                index, turn = pending.pop(0)
                futures[
                    executor.submit(
                        _request_turn,
                        turn=turn,
                        index=index,
                        api_key=api_key,
                        model=args.model,
                        max_attempts=args.max_attempts,
                    )
                ] = index
                submitted += 1
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                row = future.result()
                _append_jsonl(args.output, row, lock)
                completed_this_run += 1
                if completed_this_run <= 5 or completed_this_run % args.summary_every == 0:
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "completed_this_run": completed_this_run,
                                "submitted_this_run": submitted,
                                "index": row.get("index"),
                                "status_code": row.get("status_code"),
                                "finish_reason": row.get("finish_reason"),
                                "target_words": row.get("target_words"),
                                "cleaned_words": row.get("cleaned_words"),
                                "error": bool(row.get("error")),
                                "pending": len(pending),
                                "active": len(futures),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    _write_summary(
                        output_path=args.output,
                        before_summary=before_summary,
                        before_quota=before_quota,
                        api_key=api_key,
                        total_turns=len(turns),
                        started_at=started_at,
                    )

    _write_summary(
        output_path=args.output,
        before_summary=before_summary,
        before_quota=before_quota,
        api_key=api_key,
        total_turns=len(turns),
        started_at=started_at,
    )
    print(json.dumps({"event": "done", "output": str(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
