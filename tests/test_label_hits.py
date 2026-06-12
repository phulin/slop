import csv

from slop_sftdiv.cli.label_hits import (
    LabelState,
    _first_unlabeled,
    _load_rows,
    _next_unlabeled,
    _progress,
    _save,
    _set_label,
)


FIELDNAMES = [
    "label",
    "labeler",
    "notes",
    "source",
    "subset",
    "role",
    "record_id",
    "pair_id",
    "feature",
    "subtype",
    "start",
    "end",
    "hit_text",
    "context",
    "sample_key",
]


def _write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _row(feature, label=""):
    return {
        "label": label,
        "labeler": "",
        "notes": "",
        "source": "allenai/Dolci-Instruct-DPO",
        "subset": "unknown",
        "role": "chosen",
        "record_id": f"{feature}-1",
        "pair_id": f"{feature}-1",
        "feature": feature,
        "subtype": "subtype",
        "start": "0",
        "end": "4",
        "hit_text": "hit",
        "context": "context with hit",
        "sample_key": "abc",
    }


def test_load_rows_filters_features_and_resumes_from_output(tmp_path):
    input_path = tmp_path / "hits.csv"
    output_path = tmp_path / "labels.csv"
    _write_rows(
        input_path,
        [
            _row("contrastive_negation"),
            _row("slop_lexicon"),
        ],
    )
    _write_rows(
        output_path,
        [
            _row("contrastive_negation", label="exact"),
            _row("slop_lexicon", label="false_positive"),
        ],
    )

    rows, fieldnames = _load_rows(input_path, output_path, {"slop_lexicon"})

    assert fieldnames == FIELDNAMES
    assert len(rows) == 1
    assert rows[0]["feature"] == "slop_lexicon"
    assert rows[0]["label"] == "false_positive"


def test_progress_and_navigation_prefer_unlabeled_rows():
    rows = [
        _row("contrastive_negation", label="exact"),
        _row("contrastive_negation"),
        _row("contrastive_negation", label="ambiguous"),
    ]

    assert _progress(rows) == (2, 3)
    assert _first_unlabeled(rows) == 1
    assert _next_unlabeled(rows, 2) == 1
    assert _next_unlabeled(rows, 0, direction=-1) == 1


def test_set_label_saves_and_advances_to_next_unlabeled(tmp_path):
    output_path = tmp_path / "labels.csv"
    rows = [_row("contrastive_negation"), _row("contrastive_negation")]
    state = LabelState(rows=rows, fieldnames=FIELDNAMES, output=output_path, labeler="ph", index=0)

    _set_label(state, "exact")

    assert state.index == 1
    assert state.rows[0]["label"] == "exact"
    assert state.rows[0]["labeler"] == "ph"
    with output_path.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert saved[0]["label"] == "exact"
    assert saved[0]["labeler"] == "ph"


def test_save_preserves_existing_field_order(tmp_path):
    output_path = tmp_path / "labels.csv"
    state = LabelState(
        rows=[_row("slop_lexicon", label="partial")],
        fieldnames=FIELDNAMES,
        output=output_path,
        labeler="ph",
    )

    _save(state)

    with output_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == FIELDNAMES
