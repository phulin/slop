from __future__ import annotations

import json

import pandas as pd

from slop_sftdiv.cli.hape_augment_batch import _anthropic_request_rows, _build_augmented_frames


def _write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_augmented_frames_compiles_existing_and_provider_results(tmp_path) -> None:
    output_dir = tmp_path
    (output_dir / "results").mkdir()
    (output_dir / "summary.json").write_text(
        json.dumps({"dataset": "unit/hape"}) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "record_id": "doc_1_chunk_1",
                "base_doc_id": "doc_1",
                "source_dataset": "unit/hape",
                "role": "prompt",
                "provider": "human",
                "model": "human_prompt",
                "generated_by_project": False,
                "batch_custom_id": None,
                "batch_id": None,
                "text": "prompt one",
                "word_count": 2,
                "char_count": 10,
                "text_sha256": "prompt-sha",
            },
            {
                "record_id": "doc_1_chunk_2",
                "base_doc_id": "doc_1",
                "source_dataset": "unit/hape",
                "role": "human_continuation",
                "provider": "human",
                "model": "human",
                "generated_by_project": False,
                "batch_custom_id": None,
                "batch_id": None,
                "text": "human one",
                "word_count": 2,
                "char_count": 9,
                "text_sha256": "human-sha",
            },
        ]
    ).to_parquet(output_dir / "texts_existing.parquet", index=False)
    pd.DataFrame(
        [
            {
                "custom_id": "openai-doc-1",
                "base_doc_id": "doc_1",
                "provider": "openai",
                "model": "gpt-5.5",
            },
            {
                "custom_id": "gemini-doc-1",
                "base_doc_id": "doc_1",
                "provider": "gemini",
                "model": "gemini-3.5-flash",
            },
            {
                "custom_id": "anthropic-doc-1",
                "base_doc_id": "doc_1",
                "provider": "anthropic",
                "model": "claude-haiku-4-5-20251001",
            },
        ]
    ).to_parquet(output_dir / "request_manifest.parquet", index=False)
    _write_jsonl(
        output_dir / "results" / "openai_output.jsonl",
        [
            {
                "custom_id": "openai-doc-1",
                "id": "resp_1",
                "response": {
                    "body": {
                        "output": [
                            {
                                "content": [
                                    {"type": "output_text", "text": "openai continuation"}
                                ]
                            }
                        ]
                    }
                },
            }
        ],
    )
    _write_jsonl(
        output_dir / "results" / "gemini_output_0000.jsonl",
        [
            {
                "key": "gemini-doc-1",
                "response": {
                    "responseId": "gemini_resp_1",
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "gemini continuation"},
                                ]
                            }
                        }
                    ],
                },
            }
        ],
    )
    _write_jsonl(
        output_dir / "results" / "anthropic_output.jsonl",
        [
            {
                "custom_id": "anthropic-doc-1",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "id": "msg_1",
                        "content": [
                            {"type": "text", "text": "anthropic continuation"},
                        ],
                    },
                },
            }
        ],
    )

    existing, generated, combined, summary = _build_augmented_frames(
        output_dir=output_dir,
        require_complete=True,
    )

    assert len(existing) == 2
    assert len(generated) == 3
    assert len(combined) == 5
    assert summary["missing_generation_requests"] == 0
    assert sorted(generated["model"].tolist()) == [
        "claude-haiku-4-5-20251001",
        "gemini-3.5-flash",
        "gpt-5.5",
    ]
    assert set(generated["source_dataset"]) == {"unit/hape"}


def test_anthropic_request_rows_disable_thinking_by_omission() -> None:
    doc = type(
        "Doc",
        (),
        {
            "prompt_text": "prompt text",
        },
    )()
    rows = _anthropic_request_rows(
        [doc],
        [
            {
                "custom_id": "anthropic-doc-1",
                "model": "claude-haiku-4-5-20251001",
                "max_output_tokens": 900,
                "anthropic_thinking": "disabled",
            }
        ],
    )

    assert len(rows) == 1
    assert rows[0]["params"]["model"] == "claude-haiku-4-5-20251001"
    assert rows[0]["params"]["max_tokens"] == 900
    assert "thinking" not in rows[0]["params"]
