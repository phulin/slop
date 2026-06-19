from __future__ import annotations

import csv
from pathlib import Path

from slop_sftdiv.cli.audit_paper_claim_language import (
    CLAIM_LANGUAGE_SPECS,
    GLOBAL_FORBIDDEN_CLAIMS,
    build_parser,
    run_audit_paper_claim_language,
)


def _clean_manuscript() -> str:
    phrases: list[str] = []
    for spec in CLAIM_LANGUAGE_SPECS:
        phrases.extend(spec.required_phrases)
        phrases.extend(spec.caveat_phrases)
    return "\n".join(phrases) + "\n"


def _run(tmp_path: Path, manuscript: str):
    manuscript_path = tmp_path / "manuscript.md"
    output_csv = tmp_path / "claims.csv"
    output_md = tmp_path / "claims.md"
    manuscript_path.write_text(manuscript, encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--manuscript",
            str(manuscript_path),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ]
    )
    return run_audit_paper_claim_language(args), output_csv, output_md


def test_audit_paper_claim_language_marks_complete_manuscript_ready(tmp_path):
    rows, output_csv, output_md = _run(tmp_path, _clean_manuscript())

    assert len(rows) == len(CLAIM_LANGUAGE_SPECS) + len(GLOBAL_FORBIDDEN_CLAIMS)
    assert {row["status"] for row in rows} == {"ready"}
    assert "Readiness status: `ready`." in output_md.read_text(encoding="utf-8")
    with output_csv.open(encoding="utf-8", newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert persisted[0]["claim_id"] == "C1"
    assert persisted[0]["status"] == "ready"


def test_audit_paper_claim_language_marks_missing_phrase_for_review(tmp_path):
    manuscript = _clean_manuscript().replace(
        "post-training corpora move sharply away from narrative and personal web prose",
        "",
    )

    rows, _, _ = _run(tmp_path, manuscript)

    c1 = next(row for row in rows if row["claim_id"] == "C1")
    assert c1["status"] == "review"
    assert "post-training corpora move sharply away" in c1["missing_required"]


def test_audit_paper_claim_language_marks_forbidden_overclaim_for_review(tmp_path):
    manuscript = _clean_manuscript() + "\nDPO universally creates slop.\n"

    rows, _, _ = _run(tmp_path, manuscript)

    forbidden = next(
        row for row in rows if row["forbidden_hits"] == "DPO universally creates slop."
    )
    assert forbidden["status"] == "review"
    assert forbidden["forbidden_hits"] == "DPO universally creates slop."


def test_audit_paper_claim_language_flags_universal_scope_overclaim(tmp_path):
    manuscript = _clean_manuscript() + "\nAll LLMs exhibit the same slop style.\n"

    rows, _, _ = _run(tmp_path, manuscript)

    forbidden = next(
        row
        for row in rows
        if row["forbidden_hits"] == "All LLMs exhibit the same slop style."
    )
    assert forbidden["status"] == "review"


def test_audit_paper_claim_language_flags_dpo_human_preference_overclaim(tmp_path):
    manuscript = _clean_manuscript() + "\nDolci DPO chosen responses reveal human preference.\n"

    rows, _, _ = _run(tmp_path, manuscript)

    forbidden = next(
        row
        for row in rows
        if row["forbidden_hits"] == "Dolci DPO chosen responses reveal human preference."
    )
    assert forbidden["status"] == "review"


def test_audit_paper_claim_language_flags_eqbench_causal_overclaim(tmp_path):
    manuscript = (
        _clean_manuscript()
        + "\nThe EQ-Bench Slop Score is the causal measurement layer.\n"
    )

    rows, _, _ = _run(tmp_path, manuscript)

    forbidden = next(
        row
        for row in rows
        if row["forbidden_hits"] == "The EQ-Bench Slop Score is the causal measurement layer."
    )
    assert forbidden["status"] == "review"


def test_audit_paper_claim_language_flags_phase4_human_perceived_overclaim(tmp_path):
    manuscript = _clean_manuscript() + "\nDetector clusters are human-perceived slop.\n"

    rows, _, _ = _run(tmp_path, manuscript)

    forbidden = next(
        row
        for row in rows
        if row["forbidden_hits"] == "Detector clusters are human-perceived slop."
    )
    assert forbidden["status"] == "review"


def test_audit_paper_claim_language_flags_full_grid_overclaim(tmp_path):
    manuscript = (
        _clean_manuscript()
        + "\nWe ran the full 5,000 prompt x 8 completion x 3 temperature OLMo grid.\n"
    )

    rows, _, _ = _run(tmp_path, manuscript)

    forbidden = next(
        row
        for row in rows
        if row["forbidden_hits"]
        == "We ran the full 5,000 prompt x 8 completion x 3 temperature OLMo grid."
    )
    assert forbidden["status"] == "review"


def test_audit_paper_claim_language_flags_completed_full_grid_overclaim(tmp_path):
    manuscript = (
        _clean_manuscript()
        + "\nWe completed the 5,000 prompt x 8 completion x 3 temperature OLMo grid.\n"
    )

    rows, _, _ = _run(tmp_path, manuscript)

    forbidden = next(
        row
        for row in rows
        if row["forbidden_hits"]
        == "We completed the 5,000 prompt x 8 completion x 3 temperature OLMo grid."
    )
    assert forbidden["status"] == "review"


def test_audit_paper_claim_language_flags_tier1_precision_overclaim(tmp_path):
    manuscript = _clean_manuscript() + "\nAll Tier-1 regex features pass precision validation.\n"

    rows, _, _ = _run(tmp_path, manuscript)

    forbidden = next(
        row
        for row in rows
        if row["forbidden_hits"] == "All Tier-1 regex features pass precision validation."
    )
    assert forbidden["status"] == "review"


def test_audit_paper_claim_language_flags_derived_feature_overclaim(tmp_path):
    manuscript = _clean_manuscript() + "\nstock_openers_closers is an independent validated matcher.\n"

    rows, _, _ = _run(tmp_path, manuscript)

    forbidden = next(
        row
        for row in rows
        if row["forbidden_hits"]
        == "stock_openers_closers is an independent validated matcher."
    )
    assert forbidden["status"] == "review"


def test_audit_paper_claim_language_flags_smollm3_replication_overclaim(tmp_path):
    manuscript = _clean_manuscript() + "\nSmolLM3 proves universal replication.\n"

    rows, _, _ = _run(tmp_path, manuscript)

    forbidden = next(
        row for row in rows if row["forbidden_hits"] == "SmolLM3 proves universal replication."
    )
    assert forbidden["status"] == "review"


def test_audit_paper_claim_language_flags_tier3_sparse_overclaim(tmp_path):
    manuscript = _clean_manuscript() + "\nFollow-up offers are a standalone headline claim.\n"

    rows, _, _ = _run(tmp_path, manuscript)

    forbidden = next(
        row
        for row in rows
        if row["forbidden_hits"] == "Follow-up offers are a standalone headline claim."
    )
    assert forbidden["status"] == "review"
