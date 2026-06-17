import json

from slop_sftdiv.cli.eqbench_slop_score import build_parser, run_eqbench_slop_score
from slop_sftdiv.features import eqbench_prompt_manifest, score_eqbench_outputs
from slop_sftdiv.features.eqbench_slop import iter_eqbench_exact_contrast_matches


SLOPPY_OUTPUT = (
    "It is not just a checklist, but a paradigm shift for the workflow. "
    "The assistant took deep breath, showed unwavering resolve, and handled "
    "the meticulous transition with scalable insights. "
)


def test_score_eqbench_outputs_matches_aggregate_shape_and_filters_short_outputs():
    scorecard = score_eqbench_outputs(
        [
            SLOPPY_OUTPUT,
            "short",
            SLOPPY_OUTPUT.replace("paradigm", "transformative"),
        ],
        min_chars=80,
    )

    assert scorecard.sample_count == 3
    assert scorecard.valid_sample_count == 2
    assert scorecard.total_chars > 160
    assert scorecard.metrics.slop_score > 0
    assert scorecard.metrics.word_hit_count >= 5
    assert scorecard.metrics.trigram_hit_count >= 2
    assert scorecard.metrics.contrast_hit_count >= 2
    assert dict(scorecard.slop_trigram_hits)["took deep breath"] == 2
    assert any(pattern_name.startswith("RE_") for pattern_name, _ in scorecard.contrast_matches)


def test_eqbench_prompt_manifest_exposes_vendored_benchmark_prompts():
    prompts = eqbench_prompt_manifest()

    assert len(prompts) == 300
    assert all(prompt for prompt in prompts)


def test_eqbench_slop_score_cli_writes_json_csv_and_prompts(tmp_path):
    input_path = tmp_path / "outputs.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"output": SLOPPY_OUTPUT}),
                json.dumps({"custom_generation": SLOPPY_OUTPUT}),
                json.dumps({"output": "short"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "scorecard.json"
    csv_path = tmp_path / "scorecard.csv"
    prompts_path = tmp_path / "prompts.json"

    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--text-field",
            "custom_generation",
            "--min-chars",
            "80",
            "--contrast-mode",
            "portable",
            "--output",
            str(output_path),
            "--csv-output",
            str(csv_path),
            "--write-prompts",
            str(prompts_path),
        ]
    )

    payload = run_eqbench_slop_score(args)

    assert payload["metrics"]["sample_count"] == 3
    assert payload["metrics"]["valid_sample_count"] == 2
    assert payload["metrics"]["eqbench_slop_score"] > 0
    assert output_path.exists()
    assert csv_path.read_text(encoding="utf-8").startswith("eqbench_slop_score,")
    assert len(json.loads(prompts_path.read_text(encoding="utf-8"))) == 300
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["metrics"]["word_hit_count"] == payload["metrics"]["word_hit_count"]


def test_eqbench_slop_score_cli_loads_upstream_results_shape(tmp_path):
    input_path = tmp_path / "upstream.json"
    input_path.write_text(
        json.dumps(
            {
                "model-a": {"samples": [{"output": SLOPPY_OUTPUT}]},
                "model-b": {"samples": [{"response": SLOPPY_OUTPUT}]},
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "scorecard.json"

    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--min-chars",
            "80",
            "--contrast-mode",
            "portable",
            "--output",
            str(output_path),
        ]
    )

    payload = run_eqbench_slop_score(args)

    assert payload["metrics"]["sample_count"] == 2
    assert payload["metrics"]["valid_sample_count"] == 2


def test_exact_eqbench_contrast_backend_uses_node_helper():
    matches = iter_eqbench_exact_contrast_matches("It is not simple; it is complex.")

    assert matches == [("S1_RE_PRON_BE_NOT_SEP_BE", "It is not simple; it is")]


def test_eqbench_slop_score_exact_mode_records_backend():
    scorecard = score_eqbench_outputs([SLOPPY_OUTPUT], min_chars=80, contrast_mode="exact")

    assert scorecard.contrast_mode == "exact"
    assert scorecard.metrics.contrast_hit_count >= 1
    assert scorecard.contrast_matches[0][0].startswith("S")
