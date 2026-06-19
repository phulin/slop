import csv
import json
import sys
from types import SimpleNamespace

from slop_sftdiv.cli.free_running_emission import (
    _chat_template_kwargs,
    _load_model,
    _prompt_ids,
    build_parser,
    run_free_running_emission,
)


class FakeTokenizer:
    chat_template = "fake-template"
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [max(1, ord(char) % 97) for char in text[:5]]

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, **kwargs):
        assert tokenize is True
        assert add_generation_prompt is True
        assert kwargs == {"enable_thinking": False}
        rendered = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        rendered += "|assistant:"
        return [max(1, ord(char) % 97) for char in rendered[:20]]


class StringChatTemplateTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, **kwargs):
        del tokenize, kwargs
        assert add_generation_prompt is True
        return "|".join(f"{message['role']}:{message['content']}" for message in messages)


class MappingChatTemplateTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, **kwargs):
        del messages, tokenize, kwargs
        assert add_generation_prompt is True
        return {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9], "attention_mask": [1] * 9}


class NoChatTemplateTokenizer(FakeTokenizer):
    chat_template = None


def test_free_running_emission_writes_generation_cache_and_redacted_wandb(tmp_path, monkeypatch):
    input_path = tmp_path / "prompts.jsonl"
    generations_path = tmp_path / "generations.jsonl"
    summary_path = tmp_path / "summary.csv"
    input_path.write_text(
        json.dumps(
            {
                "phase2_prompt_id": "pkg-p1",
                "prompt_id": "p1",
                "prompt": "Give a short answer.",
                "text": "Reference.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    logged_tables = {}
    logged_payloads = []
    load_kwargs = {}
    init_kwargs = {}

    class FakeRun:
        def log(self, payload):
            logged_payloads.append(payload)

        def finish(self):
            pass

    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission._load_model",
        lambda **kwargs: load_kwargs.update(kwargs) or (FakeTokenizer(), object(), "cpu"),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission._generate_batch",
        lambda prompts, **_kwargs: [
            ("Certainly, delve into robust tests.", 5, 5) for _prompt in prompts
        ],
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission.init_wandb",
        lambda **kwargs: init_kwargs.update(kwargs) or FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission.log_summary_table",
        lambda _run, table_name, rows: logged_tables.setdefault(table_name, rows),
    )
    args = build_parser().parse_args(
        [
            "--model",
            "fake",
            "--model-revision",
            "it-SFT",
            "--input",
            str(input_path),
            "--sample-size",
            "1",
            "--feature",
            "slop_lexicon",
            "--feature",
            "stock_openers",
            "--temperature",
            "0.0",
            "--generations-output",
            str(generations_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    summary = run_free_running_emission(args)

    assert generations_path.exists()
    generation_rows = [json.loads(line) for line in generations_path.read_text().splitlines()]
    assert generation_rows[0]["generation"] == "Certainly, delve into robust tests."
    assert set(generation_rows[0]) == {
        "completion_index",
        "feature_count_total",
        "feature_text_chars",
        "feature_text_mode",
        "feature_text_tokens",
        "features_json",
        "generated_tokens",
        "generation",
        "generation_chars",
        "prompt_tokens",
        "record_id",
        "repeated_feature_count_total",
        "repeated_features_json",
        "row_index",
        "seed",
        "source",
        "source_kind",
        "split",
        "temperature",
        "top_p",
    }
    with summary_path.open(encoding="utf-8", newline="") as handle:
        summary_csv = list(csv.DictReader(handle))
    assert summary == summary_csv or len(summary) == len(summary_csv)
    assert {row["feature"] for row in summary} == {"slop_lexicon", "stock_openers"}
    assert load_kwargs["model_revision"] == "it-SFT"
    assert init_kwargs["config"]["model_revision"] == "it-SFT"
    assert logged_payloads[-1]["generation/generations"] == 1
    assert "Certainly, delve" not in json.dumps(logged_tables["generation_samples_redacted"])


def test_free_running_emission_can_count_final_answer_text_only(tmp_path, monkeypatch):
    input_path = tmp_path / "prompts.jsonl"
    generations_path = tmp_path / "generations.jsonl"
    summary_path = tmp_path / "summary.csv"
    input_path.write_text(
        json.dumps({"id": "p1", "prompt": "Answer.", "text": "Reference."}) + "\n",
        encoding="utf-8",
    )

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission._load_model",
        lambda **_kwargs: (FakeTokenizer(), object(), "cpu"),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission._generate_batch",
        lambda prompts, **_kwargs: [
            ("<think>Certainly, delve hidden.</think>\nPlain answer.", 5, 8)
            for _prompt in prompts
        ],
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission.log_summary_table",
        lambda *_args: None,
    )
    args = build_parser().parse_args(
        [
            "--model",
            "fake",
            "--input",
            str(input_path),
            "--sample-size",
            "1",
            "--feature",
            "slop_lexicon",
            "--feature",
            "stock_openers",
            "--temperature",
            "0.0",
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

    summary = run_free_running_emission(args)

    rows = [json.loads(line) for line in generations_path.read_text().splitlines()]
    assert rows[0]["generation"].startswith("<think>")
    assert rows[0]["feature_text_mode"] == "final_answer"
    assert rows[0]["feature_text_chars"] == len("Plain answer.")
    by_feature = {row["feature"]: row for row in summary}
    assert by_feature["slop_lexicon"]["count"] == 0
    assert by_feature["stock_openers"]["count"] == 0
    assert by_feature["slop_lexicon"]["generated_tokens"] == 2


def test_free_running_emission_resume_preserves_existing_rows_and_skips_cells(tmp_path, monkeypatch):
    input_path = tmp_path / "prompts.jsonl"
    generations_path = tmp_path / "generations.jsonl"
    summary_path = tmp_path / "summary.csv"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "p1", "prompt": "One", "text": "Reference."}),
                json.dumps({"id": "p2", "prompt": "Two", "text": "Reference."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    generations_path.write_text(
        json.dumps(
            {
                "source": input_path.stem,
                "record_id": "p1",
                "source_kind": "jsonl",
                "split": "train",
                "row_index": 0,
                "completion_index": 0,
                "temperature": 0.0,
                "top_p": 0.95,
                "seed": 1729,
                "prompt_tokens": 1,
                "generated_tokens": 1,
                "generation_chars": 5,
                "features_json": json.dumps({"slop_lexicon": 1}),
                "repeated_features_json": json.dumps({"slop_lexicon": 0}),
                "feature_count_total": 1,
                "repeated_feature_count_total": 0,
                "generation": "delve",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    generated_prompts = []
    logged_payloads = []

    class FakeRun:
        def log(self, payload):
            logged_payloads.append(payload)

        def finish(self):
            pass

    def fake_generate_batch(prompts, **_kwargs):
        generated_prompts.extend(prompts)
        return [("plain text", 3, 2) for _prompt in prompts]

    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission._load_model",
        lambda **_kwargs: (FakeTokenizer(), object(), "cpu"),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission._generate_batch",
        fake_generate_batch,
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission.log_summary_table",
        lambda *_args: None,
    )
    args = build_parser().parse_args(
        [
            "--model",
            "fake",
            "--input",
            str(input_path),
            "--sample-size",
            "2",
            "--feature",
            "slop_lexicon",
            "--temperature",
            "0.0",
            "--generations-output",
            str(generations_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
            "--resume",
        ]
    )

    summary = run_free_running_emission(args)

    assert generated_prompts == ["Two"]
    rows = [json.loads(line) for line in generations_path.read_text(encoding="utf-8").splitlines()]
    assert [row["record_id"] for row in rows] == ["p1", "p2"]
    assert logged_payloads[-1]["generation/generations"] == 2
    assert summary[0]["generations"] == 2
    assert summary[0]["count"] == 1


def test_prompt_ids_can_apply_chat_template_with_kwargs():
    kwargs = _chat_template_kwargs('{"enable_thinking": false}')

    ids = _prompt_ids(
        FakeTokenizer(),
        "Answer the user.",
        max_prompt_tokens=8,
        apply_chat_template=True,
        chat_template_kwargs=kwargs,
        missing_chat_template="error",
    )

    assert ids
    assert len(ids) <= 8


def test_prompt_ids_handles_string_chat_template_result():
    ids = _prompt_ids(
        StringChatTemplateTokenizer(),
        "Answer the user.",
        max_prompt_tokens=8,
        apply_chat_template=True,
        chat_template_kwargs={},
        missing_chat_template="error",
    )

    assert ids
    assert len(ids) <= 8
    assert all(isinstance(token_id, int) for token_id in ids)


def test_prompt_ids_handles_mapping_chat_template_result():
    ids = _prompt_ids(
        MappingChatTemplateTokenizer(),
        "Answer the user.",
        max_prompt_tokens=8,
        apply_chat_template=True,
        chat_template_kwargs={},
        missing_chat_template="error",
    )

    assert ids == [2, 3, 4, 5, 6, 7, 8, 9]


def test_prompt_ids_can_fallback_to_plain_when_chat_template_missing():
    ids = _prompt_ids(
        NoChatTemplateTokenizer(),
        "Answer the user.",
        max_prompt_tokens=8,
        apply_chat_template=True,
        chat_template_kwargs={},
        missing_chat_template="plain",
    )

    assert ids
    assert len(ids) <= 8


def test_load_model_enables_generation_cache(monkeypatch):
    class FakeDevice:
        type = "cpu"

        def __str__(self):
            return "cpu"

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()
        bfloat16 = object()
        float16 = object()
        float32 = object()

        @staticmethod
        def device(_device_name):
            return FakeDevice()

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=False)
            self.generation_config = SimpleNamespace(use_cache=False)

        def to(self, _device):
            return self

        def eval(self):
            return self

    fake_model = FakeModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_model_name, **_kwargs):
            return FakeTokenizer()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(_model_name, **_kwargs):
            return fake_model

    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
        AutoTokenizer=FakeAutoTokenizer,
    )
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    _tokenizer, model, _device = _load_model(
        model_name="fake",
        model_revision="it-soup-APO",
        dtype_name="bfloat16",
        device_name="auto",
        compile_model=False,
    )

    assert model.config.use_cache is True
    assert model.generation_config.use_cache is True


def test_free_running_emission_batches_generation_and_tracks_repeats(tmp_path, monkeypatch):
    input_path = tmp_path / "prompts.jsonl"
    generations_path = tmp_path / "generations.jsonl"
    summary_path = tmp_path / "summary.csv"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "p1", "prompt": "One", "text": "Reference."}),
                json.dumps({"id": "p2", "prompt": "Two", "text": "Reference."}),
                json.dumps({"id": "p3", "prompt": "Three", "text": "Reference."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    batch_sizes = []

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    def fake_generate_batch(prompts, **_kwargs):
        batch_sizes.append(len(prompts))
        return [("delve delve", 3, 2) for _prompt in prompts]

    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission._load_model",
        lambda **_kwargs: (FakeTokenizer(), object(), "cpu"),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission._generate_batch",
        fake_generate_batch,
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission.log_summary_table",
        lambda *_args: None,
    )
    args = build_parser().parse_args(
        [
            "--model",
            "fake",
            "--input",
            str(input_path),
            "--sample-size",
            "3",
            "--feature",
            "slop_lexicon",
            "--temperature",
            "0.0",
            "--top-p",
            "0.90",
            "--top-p",
            "0.95",
            "--generation-batch-size",
            "2",
            "--generations-output",
            str(generations_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    summary = run_free_running_emission(args)

    assert batch_sizes == [2, 2, 1, 1]
    assert len(generations_path.read_text().splitlines()) == 6
    assert {row["top_p"] for row in summary} == {0.9, 0.95}
    for row in summary:
        assert row["count"] == 6
        assert row["repeated_count"] == 3
        assert row["generations_with_feature"] == 3
        assert row["repeat_generations"] == 3
        assert row["repeat_share_after_first"] == 0.5


def test_free_running_emission_streams_completed_batches_before_failure(tmp_path, monkeypatch):
    input_path = tmp_path / "prompts.jsonl"
    generations_path = tmp_path / "generations.jsonl"
    summary_path = tmp_path / "summary.csv"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "p1", "prompt": "One", "text": "Reference."}),
                json.dumps({"id": "p2", "prompt": "Two", "text": "Reference."}),
                json.dumps({"id": "p3", "prompt": "Three", "text": "Reference."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    calls = 0

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    def fake_generate_batch(prompts, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated generation failure")
        return [("delve", 3, 1) for _prompt in prompts]

    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission._load_model",
        lambda **_kwargs: (FakeTokenizer(), object(), "cpu"),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission._generate_batch",
        fake_generate_batch,
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission.log_summary_table",
        lambda *_args: None,
    )
    args = build_parser().parse_args(
        [
            "--model",
            "fake",
            "--input",
            str(input_path),
            "--sample-size",
            "3",
            "--feature",
            "slop_lexicon",
            "--temperature",
            "0.0",
            "--generation-batch-size",
            "2",
            "--generations-output",
            str(generations_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    try:
        run_free_running_emission(args)
    except RuntimeError as exc:
        assert str(exc) == "simulated generation failure"
    else:
        raise AssertionError("expected simulated generation failure")

    rows = [json.loads(line) for line in generations_path.read_text().splitlines()]
    assert len(rows) == 2
    assert {row["record_id"] for row in rows} == {"p1", "p2"}
