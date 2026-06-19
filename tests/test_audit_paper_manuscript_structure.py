from __future__ import annotations

import argparse
from pathlib import Path

from slop_sftdiv.cli.audit_paper_manuscript_structure import (
    audit_manuscript_structure,
    run_audit_paper_manuscript_structure,
)


def _manuscript(*, abstract_words: int = 120) -> str:
    abstract = " ".join(["abstract"] * abstract_words)
    body = " ".join(["body"] * 430)
    conclusion = " ".join(["conclusion"] * 110)
    figure_captions = "\n\n".join(
        f"**Figure {index}. Caption.**\nCaption text." for index in range(1, 6)
    )
    table_captions = "\n\n".join(
        f"**Table {index}. Caption.**\nCaption text." for index in range(1, 6)
    )
    return f'''# Localizing "Slop Style" In Open Post-Training Ladders

## Abstract

{abstract}

## 1. Introduction

{body}

## 2. Related Work

{body}

## 3. Methods

{body}

## 4. Results

### 4.1 Post-Training Data Shift Toward Answer-Register Prose

{body}

### 4.2 Aggregate EQ-Bench Scores Are Useful But Insufficient

{body}

### 4.3 OLMo Slop-Lexicon Propensity Peaks At DPO

{body}

### 4.4 SmolLM3 Replicates Style Pressure With Different Stage Localization

{body}

### 4.5 Detector Discovery Produces Candidate Tier-3 Style Families

{body}

## 5. Discussion

{body}

## 6. Limitations

{body}

## 7. Conclusion

{conclusion}

## Figure And Table Captions

{figure_captions}

{table_captions}

## Reproducibility Appendices

Appendix material records data scope, exact measurement surfaces, and bounded
validation status for the integrated manuscript package and submission checks.
'''


def test_audit_manuscript_structure_flags_missing_result_subsection() -> None:
    text = _manuscript().replace(
        "### 4.3 OLMo Slop-Lexicon Propensity Peaks At DPO",
        "### 4.3 Different Heading",
    )

    rows = audit_manuscript_structure(text)

    by_check = {row["check_id"]: row for row in rows}
    assert by_check["result_subsection_order"]["status"] == "review"
    assert "missing:### 4.3 OLMo Slop-Lexicon Propensity Peaks At DPO" in by_check[
        "result_subsection_order"
    ]["warnings"]


def test_run_audit_manuscript_structure_writes_ready_artifacts(tmp_path: Path) -> None:
    manuscript = tmp_path / "paper.md"
    output_csv = tmp_path / "audit.csv"
    output_md = tmp_path / "audit.md"
    manuscript.write_text(_manuscript(abstract_words=130), encoding="utf-8")

    rows = run_audit_paper_manuscript_structure(
        argparse.Namespace(
            manuscript=manuscript,
            output_csv=output_csv,
            output_md=output_md,
        )
    )

    assert all(row["status"] == "ready" for row in rows)
    assert "result_subsection_order,ready,5/5" in output_csv.read_text(encoding="utf-8")
    assert "Readiness status: `ready`." in output_md.read_text(encoding="utf-8")
