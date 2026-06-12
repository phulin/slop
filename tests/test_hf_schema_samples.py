import json

from slop_sftdiv.data import CorpusSource, SamplingConfig, iter_corpus_records


def test_dolci_sft_chat_messages_extract_assistant_text_and_user_prompt(tmp_path):
    row = {
        "id": "personas_math_26ts6mqd3ipqzum75d8qf56e",
        "messages": [
            {"role": "user", "content": "Solve the mortgage probability problem."},
            {"role": "assistant", "content": "Use the normal CDF for both sub-problems."},
        ],
        "source_dataset": "Tulu 3 Persona MATH",
        "domain": "Math",
    }
    path = tmp_path / "dolci_sft.jsonl"
    path.write_text(f"{json.dumps(row)}\n", encoding="utf-8")
    source = CorpusSource.jsonl(path, name="allenai/Dolci-Instruct-SFT")

    records = list(iter_corpus_records(source, SamplingConfig(cap=None), keep_raw=True))

    assert len(records) == 1
    assert records[0].record_id == row["id"]
    assert records[0].text == "Use the normal CDF for both sub-problems."
    assert records[0].prompt == "Solve the mortgage probability problem."
    assert records[0].fields["text"] == "messages"
    assert records[0].fields["prompt"] == "messages"
    assert records[0].raw == row


def test_dolci_dpo_chosen_rejected_chat_transcripts_are_preserved(tmp_path):
    row = {
        "prompt_id": "correct-python-sft-187k-request-70-606",
        "preference_type": "delta_learning",
        "chosen_model": "qwen3-no_reasoning-32b",
        "rejected_model": "qwen3-no_reasoning-0.6b",
        "chosen": [
            {"role": "user", "content": "Implement default_user_agent(version, url)."},
            {"role": "assistant", "content": "Return f'airslate/{version} ({url})'."},
        ],
        "rejected": [
            {"role": "user", "content": "Implement default_user_agent(version, url)."},
            {"role": "assistant", "content": "Concatenate the app name, version, and URL."},
        ],
    }
    path = tmp_path / "dolci_dpo.jsonl"
    path.write_text(f"{json.dumps(row)}\n", encoding="utf-8")
    source = CorpusSource.jsonl(path, name="allenai/Dolci-Instruct-DPO")

    records = list(
        iter_corpus_records(source, SamplingConfig(cap=None), id_fields=("prompt_id",), keep_raw=True)
    )

    assert len(records) == 1
    assert records[0].record_id == row["prompt_id"]
    assert records[0].chosen == "Return f'airslate/{version} ({url})'."
    assert records[0].rejected == "Concatenate the app name, version, and URL."
    assert records[0].text == records[0].chosen
    assert records[0].fields == {
        "text": "chosen",
        "prompt": None,
        "chosen": "chosen",
        "rejected": "rejected",
    }
    assert records[0].metadata["sampling_strategy"] == "first"
    assert records[0].raw == row


def test_dolci_rl_prompt_is_default_text_and_solution_can_be_selected(tmp_path):
    row = {
        "id": "43743c43fab0835ee6cc7ab1af9d745535a4d424be0b611e6f3e23abdaabaa9e",
        "dataset": ["code"],
        "prompt": "user: Write delete_record(records, target).",
        "solution": "def delete_record(records, target):\n    return []",
        "ground_truth": [["assert delete_record([], {}) == []"]],
        "outputs": None,
    }
    path = tmp_path / "dolci_rl.jsonl"
    path.write_text(f"{json.dumps(row)}\n", encoding="utf-8")
    source = CorpusSource.jsonl(path, name="allenai/Dolci-Instruct-RL")

    default_records = list(iter_corpus_records(source, SamplingConfig(cap=None)))
    solution_records = list(
        iter_corpus_records(source, SamplingConfig(cap=None), text_fields=("solution",))
    )

    assert len(default_records) == 1
    assert default_records[0].text == row["solution"]
    assert default_records[0].prompt == row["prompt"]
    assert default_records[0].fields["text"] == "solution"
    assert default_records[0].fields["prompt"] == "prompt"
    assert solution_records[0].text == row["solution"]
    assert solution_records[0].prompt == row["prompt"]
    assert solution_records[0].fields["text"] == "solution"
