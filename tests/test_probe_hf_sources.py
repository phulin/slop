import pytest

from slop_sftdiv.cli.probe_hf_sources import _parse_dataset_spec, _repo_url, _shape_value


def test_shape_value_does_not_include_raw_string_text():
    shape = _shape_value("Sensitive full text that should not be logged")

    assert shape == {"type": "str", "chars": 45}


def test_shape_value_summarizes_nested_rows_without_text_preview():
    shape = _shape_value(
        {
            "messages": [
                {"role": "user", "content": "Question text"},
                {"role": "assistant", "content": "Answer text"},
            ],
            "score": 0.9,
        }
    )

    assert shape["type"] == "dict"
    assert shape["keys"] == ["messages", "score"]
    assert "Question text" not in str(shape)
    assert "Answer text" not in str(shape)


def test_repo_url_formats_dataset_and_model_links():
    assert _repo_url("HuggingFaceTB/smoltalk2", repo_type="dataset") == (
        "https://huggingface.co/datasets/HuggingFaceTB/smoltalk2"
    )
    assert _repo_url("HuggingFaceTB/SmolLM3-3B", repo_type="model") == (
        "https://huggingface.co/HuggingFaceTB/SmolLM3-3B"
    )


def test_parse_dataset_spec_accepts_repo_config_and_split():
    repo_only = _parse_dataset_spec("HuggingFaceTB/smoltalk2")
    with_config = _parse_dataset_spec("HuggingFaceTB/smoltalk2::SFT")
    with_split = _parse_dataset_spec(
        "HuggingFaceTB/smoltalk2::Preference::llama_3.1_tulu_3_8b_preference_mixture_no_think"
    )

    assert repo_only.repo_id == "HuggingFaceTB/smoltalk2"
    assert repo_only.config is None
    assert repo_only.split is None
    assert with_config.config == "SFT"
    assert with_config.split is None
    assert with_split.config == "Preference"
    assert with_split.split == "llama_3.1_tulu_3_8b_preference_mixture_no_think"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "HuggingFaceTB/smoltalk2::",
        "HuggingFaceTB/smoltalk2::::train",
        "HuggingFaceTB/smoltalk2::SFT::split::extra",
    ],
)
def test_parse_dataset_spec_rejects_ambiguous_forms(raw):
    with pytest.raises(ValueError, match="dataset must be"):
        _parse_dataset_spec(raw)
