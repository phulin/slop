import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_vllm_generation.py"
SPEC = importlib.util.spec_from_file_location("benchmark_vllm_generation", SCRIPT_PATH)
assert SPEC is not None
bench = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bench)


def test_vllm_benchmark_writes_contract_and_redacts_wandb(tmp_path, monkeypatch):
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

    class FakeRun:
        def log(self, payload):
            logged_payloads.append(payload)

        def finish(self):
            pass

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [max(3, ord(char) % 101) for char in text]

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, prompts, sampling_params):
            return [
                SimpleNamespace(
                    prompt_token_ids=[1, 2, 3],
                    outputs=[
                        SimpleNamespace(
                            text=f"Certainly, delve delve. Prompt={prompt}",
                            token_ids=[10, 11, 12, 13],
                            finish_reason="length",
                        )
                        for _index in range(sampling_params.kwargs["n"])
                    ],
                )
                for prompt in prompts
            ]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )
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

    summary = bench.run_benchmark(args)

    generation_rows = [json.loads(line) for line in generations_path.read_text().splitlines()]
    assert len(generation_rows) == 4
    assert generation_rows[0]["backend"] == "vllm"
    assert generation_rows[0]["source"] == "phase2_prompts"
    assert generation_rows[0]["prompt_tokens"] == 3
    assert generation_rows[0]["generated_tokens"] == 4
    assert "generation" in generation_rows[0]

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
    assert "Certainly, delve" not in json.dumps(logged_tables["generation_samples_redacted"])
