from __future__ import annotations

import csv
from pathlib import Path

from slop_sftdiv.cli.audit_paper_release_inventory import (
    ReleaseItem,
    build_parser,
    build_release_inventory_rows,
    run_audit_paper_release_inventory,
)


def test_release_inventory_rows_mark_ready_and_missing(tmp_path: Path) -> None:
    ready_path = tmp_path / "src/slop_sftdiv/features/tier1_matchers.py"
    ready_path.parent.mkdir(parents=True)
    ready_path.write_text("matcher\n", encoding="utf-8")
    items = (
        ReleaseItem(
            "feature_definitions",
            Path("src/slop_sftdiv/features/tier1_matchers.py"),
            "Role",
            "Caveat",
        ),
        ReleaseItem("cached_statistics", Path("missing.csv"), "Role", "Caveat"),
    )

    rows = build_release_inventory_rows(tmp_path, items)

    assert rows[0]["status"] == "ready"
    assert rows[0]["size_bytes"] == "8"
    assert rows[1]["status"] == "missing"
    assert rows[1]["size_bytes"] == ""


def test_release_inventory_cli_writes_markdown_and_csv(tmp_path: Path) -> None:
    for relative in [
        "src/slop_sftdiv/features/tier1_matchers.py",
        "src/slop_sftdiv/features/eqbench_slop.py",
        "src/slop_sftdiv/features/pybiber_full.py",
        "src/slop_sftdiv/propensity.py",
        "artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_detector_tier3_matcher_specs.json",
        "src/slop_sftdiv/cli/teacher_forced_propensity.py",
        "src/slop_sftdiv/cli/free_running_emission.py",
        "src/slop_sftdiv/generation_text.py",
        "src/slop_sftdiv/cli/import_phase4_blind_labels.py",
        "src/slop_sftdiv/cli/summarize_phase4_label_agreement.py",
        "src/slop_sftdiv/cli/apply_phase4_label_adjudication.py",
        "src/slop_sftdiv/cli/check_paper_package.py",
        "src/slop_sftdiv/cli/materialize_paper_submission_bundle.py",
        "src/slop_sftdiv/cli/audit_paper_reproducibility_manifest.py",
        "docs/experiments/paper_reproduction_runbook.md",
        "docs/experiments/paper_venue_sizing_inventory.md",
        "docs/experiments/phase3_production_runbook.md",
        "docs/experiments/phase4_production_runbook.md",
        "artifacts/stage1/census/feature_rates_by_corpus.parquet",
        "artifacts/stage1/census/preference_pair_deltas.parquet",
        "artifacts/stage1/census/phase1_pybiber_register_intervals.csv",
        "artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv",
        "artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv",
        (
            "artifacts/phase3/analysis/"
            "smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_"
            "baselines_data_rates_slop_neutral_rule3_production.csv"
        ),
        (
            "artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/"
            "olmo3_phase4_tier3_512_exact_sequence_stage_grid.csv"
        ),
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
    args = build_parser().parse_args(["--root", str(tmp_path)])

    rows = run_audit_paper_release_inventory(args)

    assert len(rows) == 25
    assert {row["status"] for row in rows} == {"ready"}
    with (
        tmp_path / "artifacts/paper/submission/paper_release_inventory.csv"
    ).open(encoding="utf-8", newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert persisted[0]["category"] == "feature_definitions"
    markdown = (tmp_path / "docs/experiments/paper_release_inventory.md").read_text(
        encoding="utf-8"
    )
    assert "Readiness status: `ready`." in markdown
    assert "| cached_statistics |" in markdown
