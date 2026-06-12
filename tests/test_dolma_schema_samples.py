import json

from slop_sftdiv.data import CorpusSource, SamplingConfig, iter_corpus_records


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_dolma3_mix_row_extracts_text_id_and_json_metadata(tmp_path):
    row = {
        "id": "7becd6f8-d2dc-4e9b-b6e7-b7956dc26c34",
        "text": "Short representative Dolma 3 document text.",
        "metadata": json.dumps(
            {
                "cc_dump": "CC-MAIN-2018-26",
                "quality_score": 0.9998,
                "nested_quality": {"kept": False},
            }
        ),
        "source": "dclm-hero-run-fasttext_for_HF",
        "version": "1.0",
        "created": "",
        "added": "",
        "doc": None,
        "attributes": None,
    }
    path = tmp_path / "dolma3_mix.jsonl"
    _write_jsonl(path, [row])

    [record] = list(
        iter_corpus_records(
            CorpusSource.jsonl(path, name="allenai/dolma3_mix-6T-1025-7B"),
            SamplingConfig(cap=None),
            keep_raw=True,
        )
    )

    assert record.record_id == row["id"]
    assert record.text == row["text"]
    assert record.prompt is None
    assert record.fields["text"] == "text"
    assert record.metadata["source"] == row["source"]
    assert record.metadata["version"] == row["version"]
    assert record.metadata["inferred_stratum"] == "web_cc"
    assert record.metadata["created"] == ""
    assert record.metadata["added"] == ""
    assert record.metadata["metadata.cc_dump"] == "CC-MAIN-2018-26"
    assert record.metadata["metadata.quality_score"] == 0.9998
    assert "metadata.nested_quality" not in record.metadata
    assert record.raw == row


def test_dolma3_mix_preserves_optional_doc_and_attributes_scalars(tmp_path):
    row = {
        "id": "51e52e10-0ef1-4e9e-8dfe-88036876fd50",
        "text": "Another representative Dolma 3 document.",
        "metadata": json.dumps({"cc_dump": "CC-MAIN-2024-18", "language": "en"}),
        "source": None,
        "version": None,
        "created": None,
        "added": None,
        "doc": json.dumps({"uri": "https://example.invalid/doc", "title": "Example", "labels": ["skip"]}),
        "attributes": json.dumps({"domain": "web", "score": 0.91, "flags": {"nested": True}}),
    }
    path = tmp_path / "dolma3_mix_full_schema.jsonl"
    _write_jsonl(path, [row])

    [record] = list(
        iter_corpus_records(
            CorpusSource.jsonl(path, name="allenai/dolma3_mix-6T"),
            SamplingConfig(cap=None),
            keep_raw=True,
        )
    )

    assert record.record_id == row["id"]
    assert record.text == row["text"]
    assert record.metadata["source"] is None
    assert record.metadata["version"] is None
    assert record.metadata["metadata.cc_dump"] == "CC-MAIN-2024-18"
    assert record.metadata["metadata.language"] == "en"
    assert record.metadata["inferred_stratum"] == "web_cc"
    assert record.metadata["doc.uri"] == "https://example.invalid/doc"
    assert record.metadata["doc.title"] == "Example"
    assert "doc.labels" not in record.metadata
    assert record.metadata["attributes.domain"] == "web"
    assert record.metadata["attributes.score"] == 0.91
    assert "attributes.flags" not in record.metadata
    assert record.raw == row
