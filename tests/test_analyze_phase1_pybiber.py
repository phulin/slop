import pandas as pd

from slop_sftdiv.cli.analyze_phase1_pybiber import build_parser, run


def test_analyze_phase1_pybiber_writes_weighted_means_and_deltas(tmp_path):
    features = {
        "f_01_past_tense": [10.0, 30.0],
        "f_02_perfect_aspect": [0.0, 0.0],
        "f_03_present_tense": [1.0, 3.0],
    }
    # Fill the remaining pybiber feature columns with zeros.
    from slop_sftdiv.features.pybiber_full import PYBIBER_FEATURES

    def write_sample(name, doc_ids, tokens, values):
        csv = tmp_path / f"{name}.csv"
        manifest = tmp_path / f"{name}.parquet"
        rows = []
        for i, doc_id in enumerate(doc_ids):
            row = {"doc_id": doc_id}
            for feature in PYBIBER_FEATURES:
                row[feature] = values.get(feature, [0.0] * len(doc_ids))[i]
            rows.append(row)
        pd.DataFrame(rows).to_csv(csv, index=False)
        pd.DataFrame(
            {
                "doc_id": [doc_id.split("#row")[0] for doc_id in doc_ids],
                "token_count": tokens,
                "text_role": ["role"] * len(doc_ids),
            }
        ).to_parquet(manifest, index=False)
        return csv, manifest

    dolma_csv, dolma_manifest = write_sample(
        "dolma",
        ["dup#row0", "dup#row1"],
        [1, 3],
        features,
    )
    sft_csv, sft_manifest = write_sample(
        "sft",
        ["s1", "s2"],
        [2, 2],
        {"f_01_past_tense": [20.0, 20.0], "f_03_present_tense": [5.0, 5.0]},
    )
    dpo_chosen_csv, dpo_chosen_manifest = write_sample(
        "chosen",
        ["c1", "c2"],
        [1, 1],
        {"f_01_past_tense": [40.0, 40.0], "f_03_present_tense": [6.0, 6.0]},
    )
    dpo_rejected_csv, dpo_rejected_manifest = write_sample(
        "rejected",
        ["r1", "r2"],
        [1, 1],
        {"f_01_past_tense": [10.0, 10.0], "f_03_present_tense": [2.0, 2.0]},
    )

    means = tmp_path / "means.csv"
    deltas = tmp_path / "deltas.csv"
    intervals = tmp_path / "intervals.csv"
    summary = tmp_path / "summary.md"
    args = build_parser().parse_args(
        [
            "--sample",
            f"dolma_pretrain,pretrain_document,{dolma_csv},{dolma_manifest},role",
            "--sample",
            f"sft_target,target_response,{sft_csv},{sft_manifest},role",
            "--sample",
            f"dpo_chosen,chosen,{dpo_chosen_csv},{dpo_chosen_manifest},role",
            "--sample",
            f"dpo_rejected,rejected,{dpo_rejected_csv},{dpo_rejected_manifest},role",
            "--output-means",
            str(means),
            "--output-deltas",
            str(deltas),
            "--output-intervals",
            str(intervals),
            "--bootstrap-samples",
            "25",
            "--bootstrap-chunk-size",
            "10",
            "--summary-output",
            str(summary),
        ]
    )

    result = run(args)

    mean_frame = pd.read_csv(means)
    delta_frame = pd.read_csv(deltas)
    interval_frame = pd.read_csv(intervals)
    dolma_past = mean_frame[
        (mean_frame["sample"] == "dolma_pretrain")
        & (mean_frame["feature"] == "f_01_past_tense")
    ]["token_weighted_mean"].item()
    chosen_delta = delta_frame[
        (delta_frame["comparison"] == "dpo_chosen_minus_rejected")
        & (delta_frame["feature"] == "f_01_past_tense")
    ]["delta"].item()

    assert result["samples"] == 4
    assert dolma_past == 25.0
    assert chosen_delta == 30.0
    assert result["interval_rows"] > 0
    assert "mean" in set(interval_frame["statistic"])
    assert "delta" in set(interval_frame["statistic"])
    summary_text = summary.read_text(encoding="utf-8")
    assert "Selected Bootstrap Intervals" in summary_text
    assert "Substantive Read" in summary_text
