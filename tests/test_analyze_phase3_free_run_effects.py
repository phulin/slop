import json

import pytest

from slop_sftdiv.cli.analyze_phase3_free_run_effects import (
    analyze_free_run_effects,
    build_parser,
    run_analyze_phase3_free_run_effects,
)


def _write_cache(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _row(record_id, completion_index, slop_count, stock_count=0, generated_tokens=100):
    return {
        "record_id": record_id,
        "completion_index": completion_index,
        "temperature": 1.0,
        "top_p": 0.95,
        "generated_tokens": generated_tokens,
        "features_json": json.dumps(
            {
                "slop_lexicon": slop_count,
                "stock_openers": stock_count,
            },
            sort_keys=True,
        ),
    }


def _sglang_row(prompt_id, completion_index, slop_count, generated_tokens=100):
    row = _row("", completion_index, slop_count, generated_tokens=generated_tokens)
    row.pop("record_id")
    row["prompt_id"] = prompt_id
    return row


def test_analyze_free_run_effects_uses_paired_sign_test_and_bh(tmp_path):
    left_path = tmp_path / "left.jsonl"
    right_path = tmp_path / "right.jsonl"
    _write_cache(
        left_path,
        [
            _row("a", 0, 0, stock_count=1),
            _row("b", 0, 0, stock_count=1),
            _row("c", 0, 1),
            _row("d", 0, 1),
        ],
    )
    _write_cache(
        right_path,
        [
            _row("a", 0, 1),
            _row("b", 0, 1),
            _row("c", 0, 1),
            _row("d", 0, 0),
            _row("unpaired", 0, 99),
        ],
    )
    args = build_parser().parse_args(
        [
            "--generation-cache",
            f"sft={left_path}",
            "--generation-cache",
            f"dpo={right_path}",
            "--comparison",
            "sft=dpo",
            "--feature",
            "slop_lexicon",
            "--feature",
            "stock_openers",
            "--output",
            str(tmp_path / "out.csv"),
            "--summary-output",
            str(tmp_path / "out.md"),
        ]
    )

    rows = analyze_free_run_effects(args)
    by_feature = {row["feature"]: row for row in rows}

    assert by_feature["slop_lexicon"]["paired_units"] == 4
    assert by_feature["slop_lexicon"]["positive_pairs"] == 2
    assert by_feature["slop_lexicon"]["negative_pairs"] == 1
    assert by_feature["slop_lexicon"]["zero_pairs"] == 1
    assert by_feature["slop_lexicon"]["sign_test_p"] == pytest.approx(1.0)
    assert by_feature["slop_lexicon"]["right_mean_per_1k_tokens"] == pytest.approx(7.5)
    assert by_feature["slop_lexicon"]["left_mean_per_1k_tokens"] == pytest.approx(5.0)
    assert by_feature["stock_openers"]["direction"] == "left_gt_right"
    assert all("bh_q" in row for row in rows)


def test_analyze_free_run_effects_pairs_sglang_prompt_ids(tmp_path):
    left_path = tmp_path / "left.jsonl"
    right_path = tmp_path / "right.jsonl"
    _write_cache(
        left_path,
        [
            _sglang_row("prompt-a", 0, 0),
            _sglang_row("prompt-a", 1, 0),
            _sglang_row("prompt-b", 0, 1),
            _sglang_row("prompt-b", 1, 1),
        ],
    )
    _write_cache(
        right_path,
        [
            _sglang_row("prompt-a", 0, 1),
            _sglang_row("prompt-a", 1, 1),
            _sglang_row("prompt-b", 0, 1),
            _sglang_row("prompt-b", 1, 0),
        ],
    )
    args = build_parser().parse_args(
        [
            "--generation-cache",
            f"base={left_path}",
            "--generation-cache",
            f"sft={right_path}",
            "--comparison",
            "base=sft",
            "--feature",
            "slop_lexicon",
            "--output",
            str(tmp_path / "out.csv"),
            "--summary-output",
            str(tmp_path / "out.md"),
        ]
    )

    rows = analyze_free_run_effects(args)

    assert len(rows) == 1
    assert rows[0]["paired_units"] == 4
    assert rows[0]["positive_pairs"] == 2
    assert rows[0]["negative_pairs"] == 1
    assert rows[0]["zero_pairs"] == 1


def test_run_analyze_phase3_free_run_effects_writes_outputs_and_logs(tmp_path, monkeypatch):
    base_path = tmp_path / "base.jsonl"
    dpo_path = tmp_path / "dpo.jsonl"
    output_path = tmp_path / "effects.csv"
    summary_path = tmp_path / "effects.md"
    _write_cache(base_path, [_row("a", 0, 0), _row("b", 0, 0)])
    _write_cache(dpo_path, [_row("a", 0, 1), _row("b", 0, 1)])
    payloads = []
    tables = {}

    class FakeRun:
        def log(self, payload):
            payloads.append(payload)

        def finish(self):
            pass

    monkeypatch.setattr(
        "slop_sftdiv.cli.analyze_phase3_free_run_effects.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.analyze_phase3_free_run_effects.log_summary_table",
        lambda _run, table_name, rows: tables.setdefault(table_name, rows),
    )
    args = build_parser().parse_args(
        [
            "--generation-cache",
            f"base={base_path}",
            "--generation-cache",
            f"dpo={dpo_path}",
            "--comparison",
            "base=dpo",
            "--feature",
            "slop_lexicon",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    rows = run_analyze_phase3_free_run_effects(args)

    assert output_path.exists()
    assert summary_path.exists()
    assert len(rows) == 1
    assert payloads[-1]["free_run_effects/rows"] == 1
    assert tables["phase3_free_run_stage_effects"] == rows
