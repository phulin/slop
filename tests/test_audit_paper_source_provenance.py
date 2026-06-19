from __future__ import annotations

import csv
from pathlib import Path

from slop_sftdiv.cli.audit_paper_source_provenance import (
    audit_source_provenance,
    build_parser,
    run_audit_paper_source_provenance,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _clean_package(root: Path) -> None:
    _write(
        root / "docs/experiments/source_card_notes.md",
        "\n".join(
            [
                "Delta Learning heuristic pairs",
                "GPT-judge pairs",
                "multiturn preference pairs",
                "`chosen_model`",
                "`rejected_model`",
                "`preference_type`",
                "SmolTalk2 Preference contains two APO sources",
            ]
        )
        + "\n",
    )
    paper_text = (
        "Dolci DPO is not pure human-preference evidence; it uses synthetic "
        "and Delta Learning-style construction regimes, so we call it "
        "preference-data evidence. The Dolma reference is an English-filtered "
        "retained sample, not full Dolma. Full pybiber is a Phase 1 corpus-side "
        "result; generated-output full pybiber is not claimed. The SmolLM3 "
        "preference cell is APO rather than OLMo's DPO.\n"
    )
    for relative in [
        "docs/experiments/paper_manuscript_draft.md",
        "docs/experiments/paper_methods_draft.md",
        "docs/experiments/paper_results_draft.md",
        "docs/experiments/paper_reviewer_faq.md",
        "docs/experiments/paper_claim_matrix.md",
    ]:
        _write(root / relative, paper_text)


def test_source_provenance_audit_passes_clean_boundaries(tmp_path: Path) -> None:
    _clean_package(tmp_path)

    rows = audit_source_provenance(tmp_path)

    assert {row["status"] for row in rows} == {"ready"}


def test_source_provenance_audit_flags_overclaim(tmp_path: Path) -> None:
    _clean_package(tmp_path)
    path = tmp_path / "docs/experiments/paper_results_draft.md"
    path.write_text(
        path.read_text(encoding="utf-8")
        + "The preference data is direct evidence of human taste.\n",
        encoding="utf-8",
    )

    rows = audit_source_provenance(tmp_path)
    by_id = {row["check_id"]: row for row in rows}

    assert by_id["dolci_dpo_not_pure_human_preference"]["status"] == "review"
    assert (
        "is direct evidence of human taste"
        in by_id["dolci_dpo_not_pure_human_preference"]["forbidden_hits"]
    )


def test_source_provenance_audit_flags_generated_output_pybiber_boundary(tmp_path: Path) -> None:
    _clean_package(tmp_path)
    path = tmp_path / "docs/experiments/paper_claim_matrix.md"
    path.write_text(
        path.read_text(encoding="utf-8")
        + "The main result is feature-specific: post-training shifts model outputs "
        "toward polished answer-register prose.\n",
        encoding="utf-8",
    )

    rows = audit_source_provenance(tmp_path)
    by_id = {row["check_id"]: row for row in rows}

    assert by_id["pybiber_generated_output_boundary"]["status"] == "review"
    assert (
        "post-training shifts model outputs toward polished answer-register prose"
        in by_id["pybiber_generated_output_boundary"]["forbidden_hits"]
    )


def test_source_provenance_cli_writes_markdown_and_csv(tmp_path: Path) -> None:
    _clean_package(tmp_path)
    args = build_parser().parse_args(["--root", str(tmp_path)])

    rows = run_audit_paper_source_provenance(args)

    assert len(rows) == 5
    with (
        tmp_path / "artifacts/paper/submission/paper_source_provenance_audit.csv"
    ).open(encoding="utf-8", newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert [row["check_id"] for row in persisted] == [row["check_id"] for row in rows]
    markdown = (
        tmp_path / "docs/experiments/paper_source_provenance_audit.md"
    ).read_text(encoding="utf-8")
    assert "Readiness status: `ready`." in markdown
    assert "| dolci_dpo_source_card_mixture | ready |" in markdown
