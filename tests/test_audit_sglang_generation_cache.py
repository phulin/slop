from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from slop_sftdiv.cli.audit_sglang_generation_cache import audit_cache, run
from slop_sftdiv.cli.audit_sglang_generation_cache import AuditSpec


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "backend": "sglang",
        "completion_index": 0,
        "feature_text_tokens": 16,
        "features_json": json.dumps({"slop_lexicon": 1}),
        "generated_tokens": 16,
        "generation": "Here is a generated answer.",
        "model": "model/base",
        "prompt_id": "prompt-1",
        "repeated_features_json": json.dumps({"slop_lexicon": 0}),
        "source": "prompt_package",
        "source_row_index": 7,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    row.update(overrides)
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_plan(path: Path, cache: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "model",
                "temperature",
                "top_p",
                "expected_generations",
                "completions_per_prompt",
                "max_new_tokens",
                "generations_output",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "stage": "base",
                "model": "model/base",
                "temperature": "1.0",
                "top_p": "0.95",
                "expected_generations": "4",
                "completions_per_prompt": "2",
                "max_new_tokens": "16",
                "generations_output": cache,
            }
        )


def _args(tmp_path: Path, *, plan: Path | None = None, cache: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        generation_plan=plan,
        audit_existing_only=False,
        generation_cache=[] if cache is None else [cache],
        expected_records=None,
        expected_backend="sglang",
        expected_model=None,
        expected_temperature=None,
        expected_top_p=None,
        completions_per_prompt=None,
        max_new_tokens=None,
        require_max_new_tokens=True,
        max_examples=5,
        output=tmp_path / "audit.csv",
        summary_output=tmp_path / "audit.md",
    )


def test_audit_sglang_cache_passes_partial_valid_plan_cache(tmp_path: Path) -> None:
    cache = tmp_path / "generations.jsonl"
    _write_jsonl(
        cache,
        [
            _row(prompt_id="prompt-1", completion_index=0),
            _row(prompt_id="prompt-1", completion_index=1),
        ],
    )
    plan = tmp_path / "plan.csv"
    _write_plan(plan, cache)

    rows = run(_args(tmp_path, plan=plan))

    assert len(rows) == 1
    assert rows[0].records == 2
    assert rows[0].expected_records == 4
    assert rows[0].prompt_groups == 1
    assert rows[0].incomplete_prompt_groups == 0
    assert rows[0].complete is False
    assert rows[0].passed is True
    assert "Passed caches: `1/1`" in (tmp_path / "audit.md").read_text(encoding="utf-8")


def test_audit_sglang_cache_can_skip_missing_plan_caches(tmp_path: Path) -> None:
    cache = tmp_path / "generations.jsonl"
    _write_jsonl(
        cache,
        [
            _row(prompt_id="prompt-1", completion_index=0),
            _row(prompt_id="prompt-1", completion_index=1),
        ],
    )
    plan = tmp_path / "plan.csv"
    missing_cache = tmp_path / "missing.jsonl"
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "model",
                "temperature",
                "top_p",
                "expected_generations",
                "completions_per_prompt",
                "max_new_tokens",
                "generations_output",
            ],
        )
        writer.writeheader()
        for stage, path in (("base", cache), ("sft", missing_cache)):
            writer.writerow(
                {
                    "stage": stage,
                    "model": "model/base",
                    "temperature": "1.0",
                    "top_p": "0.95",
                    "expected_generations": "4",
                    "completions_per_prompt": "2",
                    "max_new_tokens": "16",
                    "generations_output": path,
                }
            )
    args = _args(tmp_path, plan=plan)
    args.audit_existing_only = True

    rows = run(args)

    assert len(rows) == 1
    assert rows[0].stage == "base"
    assert rows[0].passed is True


def test_audit_sglang_cache_allows_existing_only_plan_with_no_caches(tmp_path: Path) -> None:
    plan = tmp_path / "plan.csv"
    _write_plan(plan, tmp_path / "missing.jsonl")
    args = _args(tmp_path, plan=plan)
    args.audit_existing_only = True

    rows = run(args)

    assert rows == []
    assert "Passed caches: `0/0`" in (tmp_path / "audit.md").read_text(encoding="utf-8")


def test_audit_sglang_cache_rejects_duplicates_and_bad_feature_json(tmp_path: Path) -> None:
    cache = tmp_path / "generations.jsonl"
    duplicate = _row(prompt_id="prompt-1", completion_index=0)
    _write_jsonl(
        cache,
        [
            duplicate,
            duplicate,
            _row(prompt_id="prompt-2", completion_index=0, features_json="{not-json"),
        ],
    )
    spec = AuditSpec(
        generation_cache=cache,
        expected_records=3,
        expected_model="model/base",
        expected_temperature=1.0,
        expected_top_p=0.95,
        completions_per_prompt=2,
        max_new_tokens=16,
    )

    result = audit_cache(spec, require_max_new_tokens=True, max_examples=5)

    assert result.passed is False
    assert result.duplicate_generation_keys == 1
    assert result.bad_features_json == 1
    assert result.complete is True


def test_audit_sglang_cache_rejects_incomplete_prompt_groups(tmp_path: Path) -> None:
    cache = tmp_path / "generations.jsonl"
    _write_jsonl(cache, [_row(prompt_id="prompt-1", completion_index=0)])
    spec = AuditSpec(
        generation_cache=cache,
        expected_records=2,
        expected_model="model/base",
        expected_temperature=1.0,
        expected_top_p=0.95,
        completions_per_prompt=2,
        max_new_tokens=16,
    )

    result = audit_cache(spec, require_max_new_tokens=True, max_examples=5)

    assert result.passed is False
    assert result.prompt_groups == 1
    assert result.incomplete_prompt_groups == 1
    assert "1/2 completions" in (result.example_failures or [""])[0]


def test_audit_sglang_cache_rejects_wrong_metadata_and_short_rows(tmp_path: Path) -> None:
    cache = tmp_path / "generations.jsonl"
    _write_jsonl(
        cache,
        [
            _row(
                backend="torch",
                model="wrong/model",
                temperature=0.7,
                top_p=1.0,
                generated_tokens=8,
                feature_text_tokens=0,
                generation="",
            )
        ],
    )
    spec = AuditSpec(
        generation_cache=cache,
        expected_model="model/base",
        expected_temperature=1.0,
        expected_top_p=0.95,
        completions_per_prompt=2,
        max_new_tokens=16,
    )

    result = audit_cache(spec, require_max_new_tokens=True, max_examples=5)

    assert result.passed is False
    assert result.wrong_backend == 1
    assert result.wrong_model == 1
    assert result.wrong_temperature == 1
    assert result.wrong_top_p == 1
    assert result.short_generated_tokens == 1
    assert result.bad_feature_text_tokens == 1
    assert result.empty_generation == 1


def test_audit_sglang_cache_requires_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provide --generation-plan"):
        run(_args(tmp_path))
