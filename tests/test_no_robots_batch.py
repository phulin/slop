from __future__ import annotations

import json

import pandas as pd

from slop_sftdiv.cli.no_robots_batch import (
    _anthropic_request_rows,
    _assistant_turn_rows_for_frame,
    _complete_triplet_turn_ids,
    _fireworks_request_rows,
    _gemini_request_rows,
    _openai_request_rows,
    _provider_manifest_rows,
)


def test_assistant_turn_rows_include_consecutive_assistant_fallback() -> None:
    frame = pd.DataFrame(
        [
            {
                "prompt": "prompt",
                "prompt_id": "prompt-1",
                "category": "Chat",
                "messages": [
                    {"role": "system", "content": "Be terse."},
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First answer"},
                    {"role": "user", "content": "Second question"},
                    {"role": "assistant", "content": "Second answer"},
                    {"role": "assistant", "content": "Consecutive assistant answer"},
                ],
            }
        ]
    )

    rows = _assistant_turn_rows_for_frame(
        frame,
        split="train",
        min_output_tokens=32,
        max_output_tokens=256,
    )

    assert len(rows) == 3
    assert [row["assistant_turn_ordinal"] for row in rows] == [1, 2, 3]
    assert rows[0]["uses_transcript_prompt"] is False
    assert rows[1]["uses_transcript_prompt"] is False
    assert rows[2]["uses_transcript_prompt"] is True
    assert json.loads(rows[2]["input_messages_json"])[-1]["role"] == "assistant"


def test_provider_requests_set_lowest_reasoning_controls() -> None:
    frame = pd.DataFrame(
        [
            {
                "prompt": "prompt",
                "prompt_id": "prompt-1",
                "category": "Chat",
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ],
            }
        ]
    )
    turns = _assistant_turn_rows_for_frame(
        frame,
        split="test",
        min_output_tokens=32,
        max_output_tokens=256,
    )

    openai_manifest = _provider_manifest_rows(
        turns,
        provider="openai",
        model="gpt-5.5",
        openai_reasoning_effort="none",
    )
    gemini_manifest = _provider_manifest_rows(
        turns,
        provider="gemini",
        model="gemini-3.5-flash",
        gemini_thinking_level="minimal",
    )
    anthropic_manifest = _provider_manifest_rows(
        turns,
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        anthropic_thinking="disabled",
    )
    fireworks_manifest = _provider_manifest_rows(
        turns,
        provider="fireworks",
        model="accounts/fireworks/models/glm-5p2",
        fireworks_reasoning_effort="none",
    )

    openai = _openai_request_rows(turns, openai_manifest)[0]
    gemini = _gemini_request_rows(turns, gemini_manifest)[0]
    anthropic = _anthropic_request_rows(turns, anthropic_manifest)[0]
    fireworks = _fireworks_request_rows(turns, fireworks_manifest)[0]

    assert openai["body"]["reasoning"] == {"effort": "none"}
    assert gemini["request"]["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "minimal"
    }
    assert "thinking" not in anthropic["params"]
    assert fireworks["body"]["reasoning_effort"] == "none"
    assert len(anthropic["custom_id"]) <= 64


def test_complete_triplet_turn_ids_excludes_incomplete_and_long_examples() -> None:
    frame = pd.DataFrame(
        [
            {"turn_id": "ok", "model": "human", "record_id": "ok-h", "token_len": 100},
            {"turn_id": "ok", "model": "gpt-5.5", "record_id": "ok-g", "token_len": 640},
            {
                "turn_id": "ok",
                "model": "gemini-3.5-flash",
                "record_id": "ok-m",
                "token_len": 300,
            },
            {"turn_id": "long", "model": "human", "record_id": "long-h", "token_len": 100},
            {"turn_id": "long", "model": "gpt-5.5", "record_id": "long-g", "token_len": 641},
            {
                "turn_id": "long",
                "model": "gemini-3.5-flash",
                "record_id": "long-m",
                "token_len": 300,
            },
            {"turn_id": "missing", "model": "human", "record_id": "missing-h", "token_len": 100},
            {
                "turn_id": "missing",
                "model": "gpt-5.5",
                "record_id": "missing-g",
                "token_len": 100,
            },
        ]
    )

    eligible, summary = _complete_triplet_turn_ids(
        frame,
        expected_models=["human", "gpt-5.5", "gemini-3.5-flash"],
        max_token_length=640,
    )

    assert eligible == {"ok"}
    assert summary["complete_triplets"] == 2
    assert summary["eligible_triplets"] == 1
    assert summary["incomplete_or_missing_triplets"] == 1
    assert summary["complete_triplets_over_cap"] == 1
