import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_sglang_generation.py"
SPEC = importlib.util.spec_from_file_location("benchmark_sglang_generation", SCRIPT_PATH)
assert SPEC is not None
bench = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bench)


def test_sglang_benchmark_writes_contract_and_redacts_wandb(tmp_path, monkeypatch):
    input_path = tmp_path / "phase2_prompts.jsonl"
    generations_path = tmp_path / "generations.jsonl"
    summary_path = tmp_path / "summary.csv"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "phase2_prompt_id": "pkg-p1",
                        "prompt": "Give a short answer.",
                        "source_row_index": 7,
                    }
                ),
                json.dumps(
                    {
                        "phase2_prompt_id": "pkg-p2",
                        "prompt": "Give another short answer.",
                        "source_row_index": 8,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    logged_payloads = []
    logged_tables = {}
    observed_sampling_params = []

    class FakeRun:
        def log(self, payload):
            logged_payloads.append(payload)

        def finish(self):
            pass

    class FakeTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [max(3, ord(char) % 101) for char in text]

    class FakeEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.shutdown_called = False

        def generate(self, *, input_ids, sampling_params):
            observed_sampling_params.append(sampling_params)
            return [
                {
                    "text": [
                        f"Certainly, delve delve. Prompt={prompt_ids}"
                        for _index in range(sampling_params["n"])
                    ],
                    "meta_info": [
                        {
                            "completion_tokens": 4,
                            "finish_reason": "length",
                        }
                        for _index in range(sampling_params["n"])
                    ],
                }
                for prompt_ids in input_ids
            ]

        def shutdown(self):
            self.shutdown_called = True

    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=FakeEngine))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda _model: FakeTokenizer())
        ),
    )
    monkeypatch.setattr(bench, "init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr(
        bench,
        "log_summary_table",
        lambda _run, table_name, rows: logged_tables.setdefault(table_name, rows),
    )

    args = bench.build_parser().parse_args(
        [
            "--model",
            "fake-model",
            "--input",
            str(input_path),
            "--sample-size",
            "2",
            "--temperature",
            "1.0",
            "--top-p",
            "0.95",
            "--completions-per-prompt",
            "2",
            "--prompt-batch-size",
            "1",
            "--max-new-tokens",
            "4",
            "--ignore-eos",
            "--generations-output",
            str(generations_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    summary = bench.run_benchmark(args)

    generation_rows = [json.loads(line) for line in generations_path.read_text().splitlines()]
    assert len(generation_rows) == 4
    assert generation_rows[0]["backend"] == "sglang"
    assert generation_rows[0]["source"] == "phase2_prompts"
    assert generation_rows[0]["generated_tokens"] == 4
    assert "generation" in generation_rows[0]
    assert generation_rows[0]["feature_text_mode"] == "raw"
    assert generation_rows[0]["feature_text_tokens"] == 4

    with summary_path.open(encoding="utf-8", newline="") as handle:
        summary_csv = list(csv.DictReader(handle))
    assert len(summary) == len(summary_csv)
    assert {row["feature"] for row in summary} == {
        "contrastive_negation",
        "rule_of_three_approx",
        "slop_lexicon",
        "stock_closers",
        "stock_openers",
        "stock_openers_closers",
    }
    assert {row["source"] for row in summary} == {"phase2_prompts"}
    assert logged_payloads[-1]["generation/prompts_seen"] == 2
    assert logged_payloads[-1]["generation/generations"] == 4
    assert observed_sampling_params[0]["ignore_eos"] is True
    assert "Certainly, delve" not in json.dumps(logged_tables["generation_samples_redacted"])


def test_sglang_benchmark_accepts_flattened_multi_completion_responses(tmp_path, monkeypatch):
    input_path = tmp_path / "phase2_prompts.jsonl"
    generations_path = tmp_path / "generations.jsonl"
    summary_path = tmp_path / "summary.csv"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"phase2_prompt_id": "pkg-p1", "prompt": "Prompt one."}),
                json.dumps({"phase2_prompt_id": "pkg-p2", "prompt": "Prompt two."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    class FakeTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [max(3, ord(char) % 101) for char in text]

    class FakeEngine:
        def __init__(self, **_kwargs):
            pass

        def generate(self, *, input_ids, sampling_params):
            rows = []
            for prompt_index, _prompt_ids in enumerate(input_ids):
                for completion_index in range(sampling_params["n"]):
                    rows.append(
                        {
                            "text": f"completion {prompt_index}.{completion_index}",
                            "meta_info": {
                                "completion_tokens": 3,
                                "finish_reason": "length",
                            },
                        }
                    )
            return rows

        def shutdown(self):
            pass

    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=FakeEngine))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda _model: FakeTokenizer())
        ),
    )
    monkeypatch.setattr(bench, "init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr(bench, "log_summary_table", lambda *_args, **_kwargs: None)

    args = bench.build_parser().parse_args(
        [
            "--model",
            "fake-model",
            "--input",
            str(input_path),
            "--sample-size",
            "2",
            "--sampling-strategy",
            "first",
            "--temperature",
            "1.0",
            "--top-p",
            "0.95",
            "--completions-per-prompt",
            "3",
            "--max-new-tokens",
            "4",
            "--generations-output",
            str(generations_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    bench.run_benchmark(args)

    generation_rows = [json.loads(line) for line in generations_path.read_text().splitlines()]
    assert len(generation_rows) == 6
    assert [row["completion_index"] for row in generation_rows] == [0, 1, 2, 0, 1, 2]
    assert {row["generated_tokens"] for row in generation_rows} == {3}


def test_sglang_benchmark_can_count_final_answer_text_only(tmp_path, monkeypatch):
    input_path = tmp_path / "phase2_prompts.jsonl"
    generations_path = tmp_path / "generations.jsonl"
    summary_path = tmp_path / "summary.csv"
    input_path.write_text(
        json.dumps({"phase2_prompt_id": "pkg-p1", "prompt": "Prompt one."}) + "\n",
        encoding="utf-8",
    )

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    class FakeTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [max(3, ord(char) % 101) for char in text]

    class FakeEngine:
        def __init__(self, **_kwargs):
            pass

        def generate(self, *, input_ids, sampling_params):
            del input_ids, sampling_params
            return [
                {
                    "text": "<think>Certainly, delve hidden.</think>\nPlain answer.",
                    "meta_info": {
                        "completion_tokens": 8,
                        "finish_reason": "length",
                    },
                }
            ]

        def shutdown(self):
            pass

    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=FakeEngine))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda _model: FakeTokenizer())
        ),
    )
    monkeypatch.setattr(bench, "init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr(bench, "log_summary_table", lambda *_args, **_kwargs: None)

    args = bench.build_parser().parse_args(
        [
            "--model",
            "fake-model",
            "--input",
            str(input_path),
            "--sample-size",
            "1",
            "--sampling-strategy",
            "first",
            "--feature",
            "slop_lexicon",
            "--feature",
            "stock_openers",
            "--temperature",
            "1.0",
            "--top-p",
            "0.95",
            "--max-new-tokens",
            "8",
            "--feature-text-mode",
            "final_answer",
            "--generations-output",
            str(generations_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    summary = bench.run_benchmark(args)

    generation_rows = [json.loads(line) for line in generations_path.read_text().splitlines()]
    assert generation_rows[0]["generation"].startswith("<think>")
    assert generation_rows[0]["feature_text_mode"] == "final_answer"
    assert generation_rows[0]["feature_text_chars"] == len("Plain answer.")
    by_feature = {row["feature"]: row for row in summary}
    assert by_feature["slop_lexicon"]["count"] == 0
    assert by_feature["stock_openers"]["count"] == 0
    assert by_feature["slop_lexicon"]["generated_tokens"] == 2


def test_sglang_benchmark_resume_skips_completed_generation_rows(tmp_path, monkeypatch):
    input_path = tmp_path / "phase2_prompts.jsonl"
    generations_path = tmp_path / "generations.jsonl"
    summary_path = tmp_path / "summary.csv"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"phase2_prompt_id": "pkg-p1", "prompt": "Prompt one."}),
                json.dumps({"phase2_prompt_id": "pkg-p2", "prompt": "Prompt two."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    existing_row = {
        "backend": "sglang",
        "model": "fake-model",
        "source": "phase2_prompts",
        "prompt_id": "pkg-p1",
        "source_row_index": 0,
        "completion_index": 0,
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": 1729,
        "prompt_tokens": 2,
        "generated_tokens": 4,
        "generation_chars": 16,
        "feature_text_mode": "raw",
        "feature_text_chars": 16,
        "feature_text_tokens": 4,
        "features_json": json.dumps({"slop_lexicon": 1}),
        "repeated_features_json": json.dumps({"slop_lexicon": 0}),
        "feature_count_total": 1,
        "repeated_feature_count_total": 0,
        "finish_reason": "length",
        "generation": "Existing delve.",
    }
    generations_path.write_text(json.dumps(existing_row) + "\n", encoding="utf-8")
    generated_prompt_batches = []

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    class FakeTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [max(3, ord(char) % 101) for char in text]

    class FakeEngine:
        def __init__(self, **_kwargs):
            pass

        def generate(self, *, input_ids, sampling_params):
            generated_prompt_batches.append(len(input_ids))
            rows = []
            for prompt_index, _prompt_ids in enumerate(input_ids):
                for completion_index in range(sampling_params["n"]):
                    rows.append(
                        {
                            "text": f"new completion {prompt_index}.{completion_index}",
                            "meta_info": {
                                "completion_tokens": 3,
                                "finish_reason": "length",
                            },
                        }
                    )
            return rows

        def shutdown(self):
            pass

    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=FakeEngine))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda _model: FakeTokenizer())
        ),
    )
    monkeypatch.setattr(bench, "init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr(bench, "log_summary_table", lambda *_args, **_kwargs: None)

    args = bench.build_parser().parse_args(
        [
            "--model",
            "fake-model",
            "--input",
            str(input_path),
            "--sample-size",
            "2",
            "--sampling-strategy",
            "first",
            "--feature",
            "slop_lexicon",
            "--temperature",
            "1.0",
            "--top-p",
            "0.95",
            "--completions-per-prompt",
            "2",
            "--prompt-batch-size",
            "1",
            "--max-new-tokens",
            "4",
            "--resume",
            "--generations-output",
            str(generations_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    summary = bench.run_benchmark(args)

    generation_rows = [json.loads(line) for line in generations_path.read_text().splitlines()]
    assert len(generation_rows) == 4
    assert generation_rows[0]["generation"] == "Existing delve."
    assert generated_prompt_batches == [1, 1]
    keys = {
        (
            row["prompt_id"],
            row["source_row_index"],
            row["completion_index"],
            row["temperature"],
            row["top_p"],
        )
        for row in generation_rows
    }
    assert keys == {
        ("pkg-p1", 0, 0, 1.0, 0.95),
        ("pkg-p1", 0, 1, 1.0, 0.95),
        ("pkg-p2", 1, 0, 1.0, 0.95),
        ("pkg-p2", 1, 1, 1.0, 0.95),
    }
    by_feature = {row["feature"]: row for row in summary}
    assert by_feature["slop_lexicon"]["generations"] == 4
    assert by_feature["slop_lexicon"]["count"] == 1
