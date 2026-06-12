import json

from slop_sftdiv.cli.census import build_parser, run_census


def test_run_census_with_disabled_wandb_on_local_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "summary.csv"
    input_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"id": "a", "text": "Certainly, delve into this robust plan."},
                {"id": "b", "text": "It is not noisy but focused."},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--sample-size",
            "2",
            "--output",
            str(output_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    frame = run_census(args)

    assert output_path.exists()
    assert not frame.empty
    assert set(frame["feature"]) >= {
        "contrastive_negation",
        "slop_lexicon",
        "stock_openers_closers",
    }
    assert frame.loc[frame["feature"] == "slop_lexicon", "count"].sum() >= 2
    assert frame["docs"].max() == 2
