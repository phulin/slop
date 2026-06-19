import json

from slop_sftdiv.data import CorpusSource, SamplingConfig, iter_corpus_records, iter_jsonl_rows


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8")


def test_iter_jsonl_rows_skips_blank_lines_and_preserves_objects(tmp_path):
    path = tmp_path / "sample.jsonl"
    path.write_text('{"id": "a", "text": "One"}\n\n{"id": "b", "text": "Two"}\n', encoding="utf-8")

    assert list(iter_jsonl_rows(path)) == [
        {"id": "a", "text": "One"},
        {"id": "b", "text": "Two"},
    ]


def test_iter_corpus_records_extracts_common_fields_and_metadata(tmp_path):
    path = tmp_path / "prefs.jsonl"
    _write_jsonl(
        path,
        [
            {
                "uid": "row-1",
                "instruction": "Explain testing.",
                "chosen": {"content": "Good answer"},
                "rejected": ["Bad", "answer"],
            },
            {"id": "row-2", "chosen": "Fallback text"},
            {"id": "row-3", "metadata_only": True},
        ],
    )
    source = CorpusSource.jsonl(path, name="prefs", stratum="gold")

    records = list(iter_corpus_records(source, SamplingConfig(cap=None), keep_raw=True))

    assert [record.record_id for record in records] == ["row-1", "row-2"]
    assert records[0].text == "Good answer"
    assert records[0].prompt == "Explain testing."
    assert records[0].rejected == "Bad\nanswer"
    assert records[0].fields == {
        "text": "chosen",
        "prompt": "instruction",
        "chosen": "chosen",
        "rejected": "rejected",
    }
    assert records[0].metadata["stratum"] == "gold"
    assert records[0].raw is not None
    assert records[1].text == "Fallback text"


def test_sampling_first_honors_cap_and_max_scan(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [{"id": str(i), "text": f"row {i}"} for i in range(5)])

    records = list(
        iter_corpus_records(
            CorpusSource.jsonl(path, name="rows"),
            SamplingConfig(cap=3, max_scan=2, strategy="first"),
        )
    )

    assert [record.row_index for record in records] == [0, 1]
    assert [record.text for record in records] == ["row 0", "row 1"]
    assert all(record.metadata["sampling_strategy"] == "first" for record in records)


def test_hash_reservoir_sampling_is_deterministic_and_seeded(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [{"id": str(i), "text": f"row {i}"} for i in range(20)])
    source = CorpusSource.jsonl(path, name="rows")

    first = list(iter_corpus_records(source, SamplingConfig(cap=5, seed=7, strategy="hash_reservoir")))
    second = list(iter_corpus_records(source, SamplingConfig(cap=5, seed=7, strategy="hash_reservoir")))
    different_seed = list(
        iter_corpus_records(source, SamplingConfig(cap=5, seed=8, strategy="hash_reservoir"))
    )

    assert [record.record_id for record in first] == [record.record_id for record in second]
    assert [record.record_id for record in first] != [record.record_id for record in different_seed]
    assert len(first) == 5
    assert all(len(record.metadata["sample_key"]) == 16 for record in first)


def test_chat_messages_extract_prompt_and_assistant_target(tmp_path):
    path = tmp_path / "sft.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "sft-1",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Explain tests."},
                    {"role": "assistant", "content": "Tests check behavior."},
                ],
            }
        ],
    )

    [record] = list(iter_corpus_records(CorpusSource.jsonl(path, name="sft")))

    assert record.prompt == "Be concise.\nExplain tests."
    assert record.text == "Tests check behavior."
    assert record.fields["prompt"] == "messages"
    assert record.fields["text"] == "messages"


def test_preference_chat_lists_extract_assistant_responses(tmp_path):
    path = tmp_path / "dpo.jsonl"
    _write_jsonl(
        path,
        [
            {
                "prompt_id": "pair-1",
                "chosen": [
                    {"role": "user", "content": "Explain tests."},
                    {"role": "assistant", "content": "Great question. Use assertions."},
                ],
                "rejected": [
                    {"role": "user", "content": "Explain tests."},
                    {"role": "assistant", "content": "No."},
                ],
            }
        ],
    )

    [record] = list(iter_corpus_records(CorpusSource.jsonl(path, name="dpo")))

    assert record.text == "Great question. Use assertions."
    assert record.chosen == "Great question. Use assertions."
    assert record.rejected == "No."


def test_response_mapping_lists_extract_as_text(tmp_path):
    path = tmp_path / "responses.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "reasoning-1",
                "responses": [
                    {"response_model": "model-a", "response": "First response."},
                    {"response_model": "model-b", "response": "Second response."},
                ],
            }
        ],
    )

    [record] = list(iter_corpus_records(CorpusSource.jsonl(path, name="responses"), text_fields=("responses",)))

    assert record.text == "First response.\nSecond response."
    assert record.fields["text"] == "responses"


def test_row_metadata_promotes_common_corpus_fields_and_json_metadata(tmp_path):
    path = tmp_path / "dolma.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "doc-1",
                "text": "Document text.",
                "source": "common_crawl",
                "version": "v1",
                "metadata": json.dumps({"domain": "web", "quality": 0.8, "nested": {"skip": True}}),
                "attributes": {"language": "en", "score": 0.9},
            }
        ],
    )

    [record] = list(iter_corpus_records(CorpusSource.jsonl(path, name="dolma")))

    assert record.metadata["source"] == "common_crawl"
    assert record.metadata["version"] == "v1"
    assert record.metadata["inferred_stratum"] == "web_cc"
    assert record.metadata["metadata.domain"] == "web"
    assert record.metadata["metadata.quality"] == 0.8
    assert "metadata.nested" not in record.metadata
    assert record.metadata["attributes.language"] == "en"
    assert record.metadata["attributes.score"] == 0.9


def test_source_metadata_overrides_promoted_row_metadata(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [{"id": "row-1", "text": "Text.", "domain": "row-domain"}])

    [record] = list(iter_corpus_records(CorpusSource.jsonl(path, name="rows", domain="source-domain")))

    assert record.metadata["domain"] == "source-domain"


def test_row_metadata_promotes_retained_sample_fields(tmp_path):
    path = tmp_path / "retained.jsonl"
    _write_jsonl(
        path,
        [
            {
                "doc_id": "doc-1",
                "text": "Text.",
                "stratum": "forums_qa",
                "provenance": "forum_source",
                "text_role": "pretrain_document",
            }
        ],
    )

    [record] = list(iter_corpus_records(CorpusSource.jsonl(path, name="retained")))

    assert record.metadata["stratum"] == "forums_qa"
    assert record.metadata["provenance"] == "forum_source"
    assert record.metadata["text_role"] == "pretrain_document"


def test_row_metadata_infers_scientific_stratum(tmp_path):
    path = tmp_path / "science.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "doc-1",
                "text": "Scientific PDF text.",
                "source": "olmocr_science_pdfs",
                "metadata": json.dumps({"collection": "arxiv"}),
            }
        ],
    )

    [record] = list(iter_corpus_records(CorpusSource.jsonl(path, name="dolma")))

    assert record.metadata["inferred_stratum"] == "scientific"


def test_row_metadata_infers_full_cc_as_web_cc(tmp_path):
    path = tmp_path / "full_cc.jsonl"
    _write_jsonl(path, [{"id": "doc-1", "text": "Web text.", "source": "full_CC"}])

    [record] = list(iter_corpus_records(CorpusSource.jsonl(path, name="dolma")))

    assert record.metadata["inferred_stratum"] == "web_cc"
