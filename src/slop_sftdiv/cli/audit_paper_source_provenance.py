from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path.cwd()
DEFAULT_OUTPUT_CSV = Path("artifacts/paper/submission/paper_source_provenance_audit.csv")
DEFAULT_OUTPUT_MD = Path("docs/experiments/paper_source_provenance_audit.md")

SOURCE_CARD_NOTES = Path("docs/experiments/source_card_notes.md")
PAPER_SCOPE_DOCS = (
    Path("docs/experiments/paper_manuscript_draft.md"),
    Path("docs/experiments/paper_methods_draft.md"),
    Path("docs/experiments/paper_results_draft.md"),
    Path("docs/experiments/paper_reviewer_faq.md"),
    Path("docs/experiments/paper_claim_matrix.md"),
)


@dataclass(frozen=True)
class RequiredGroup:
    label: str
    paths: tuple[Path, ...]
    phrases: tuple[str, ...]


@dataclass(frozen=True)
class ProvenanceCheck:
    check_id: str
    scope: str
    required: tuple[RequiredGroup, ...]
    forbidden: tuple[str, ...]
    purpose: str


CHECKS: tuple[ProvenanceCheck, ...] = (
    ProvenanceCheck(
        "dolci_dpo_source_card_mixture",
        str(SOURCE_CARD_NOTES),
        (
            RequiredGroup(
                "mixture_components",
                (SOURCE_CARD_NOTES,),
                ("Delta Learning heuristic pairs", "GPT-judge pairs", "multiturn preference pairs"),
            ),
            RequiredGroup(
                "metadata_fields",
                (SOURCE_CARD_NOTES,),
                ("`chosen_model`", "`rejected_model`", "`preference_type`"),
            ),
        ),
        (),
        "Records the primary-source Dolci DPO mixture and metadata needed for interpretation.",
    ),
    ProvenanceCheck(
        "dolci_dpo_not_pure_human_preference",
        "; ".join(str(path) for path in PAPER_SCOPE_DOCS),
        (
            RequiredGroup(
                "paper_boundary",
                PAPER_SCOPE_DOCS,
                (
                    "not pure human-preference",
                    "synthetic and Delta Learning-style construction regimes",
                    "preference-data",
                ),
            ),
        ),
        ("is direct evidence of human taste", "pure human preference result"),
        "Prevents Dolci DPO chosen/rejected contrasts from being overread as pure human preference.",
    ),
    ProvenanceCheck(
        "dolma_sample_boundary",
        "; ".join(str(path) for path in PAPER_SCOPE_DOCS),
        (
            RequiredGroup(
                "sample_boundary",
                PAPER_SCOPE_DOCS,
                ("English-filtered", "retained", "not full Dolma"),
            ),
        ),
        ("full Dolma baseline", "entire Dolma corpus", "complete Dolma corpus"),
        "Keeps the pretraining reference framed as a bounded retained English sample.",
    ),
    ProvenanceCheck(
        "pybiber_generated_output_boundary",
        "; ".join(str(path) for path in PAPER_SCOPE_DOCS),
        (
            RequiredGroup(
                "pybiber_boundary",
                PAPER_SCOPE_DOCS,
                ("Phase 1 corpus-side", "generated-output full pybiber is not claimed"),
            ),
        ),
        (
            "generated-output full pybiber evidence",
            "full generated-output pybiber result",
            "post-training shifts model outputs toward polished answer-register prose",
        ),
        "Preserves the active boundary between corpus-side full pybiber and generated-output diagnostics.",
    ),
    ProvenanceCheck(
        "smollm3_apo_source_boundary",
        f"{SOURCE_CARD_NOTES}; " + "; ".join(str(path) for path in PAPER_SCOPE_DOCS),
        (
            RequiredGroup(
                "source_card_boundary",
                (SOURCE_CARD_NOTES,),
                ("SmolTalk2 Preference contains two APO sources",),
            ),
            RequiredGroup(
                "paper_boundary",
                PAPER_SCOPE_DOCS,
                ("preference cell is APO rather than OLMo's DPO",),
            ),
        ),
        ("SmolLM3 proves generic DPO behavior",),
        "Keeps SmolLM3 preference-stage claims tied to the documented APO/source-card boundary.",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit paper source/provenance interpretation.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def run_audit_paper_source_provenance(args: argparse.Namespace) -> list[dict[str, str]]:
    root = args.root.resolve()
    rows = audit_source_provenance(root)
    output_csv = _resolve_path(root, args.output_csv)
    output_md = _resolve_path(root, args.output_md)
    _write_csv(output_csv, rows)
    _write_markdown(output_md, rows, output_csv=args.output_csv)
    return rows


def audit_source_provenance(root: Path) -> list[dict[str, str]]:
    return [_row(root, check) for check in CHECKS]


def _row(root: Path, check: ProvenanceCheck) -> dict[str, str]:
    missing_required: list[str] = []
    required_hits = 0
    required_total = 0
    for group in check.required:
        text, missing_paths = _combined_text(root, group.paths)
        if missing_paths:
            missing_required.extend(f"{group.label}:missing path {path}" for path in missing_paths)
            required_total += len(group.phrases)
            continue
        for phrase in group.phrases:
            required_total += 1
            if phrase in text:
                required_hits += 1
            else:
                missing_required.append(f"{group.label}:{phrase}")

    forbidden_hits: list[str] = []
    scope_paths = _scope_paths(check)
    scope_text, missing_paths = _combined_text(root, scope_paths)
    if missing_paths:
        missing_required.extend(f"scope:missing path {path}" for path in missing_paths)
    else:
        forbidden_hits.extend(phrase for phrase in check.forbidden if phrase in scope_text)

    return {
        "check_id": check.check_id,
        "status": "ready" if not missing_required and not forbidden_hits else "review",
        "scope": check.scope,
        "required_hits": f"{required_hits}/{required_total}",
        "missing_required": "; ".join(missing_required),
        "forbidden_hits": "; ".join(forbidden_hits),
        "purpose": check.purpose,
    }


def _scope_paths(check: ProvenanceCheck) -> tuple[Path, ...]:
    paths: list[Path] = []
    for group in check.required:
        for path in group.paths:
            if path not in paths:
                paths.append(path)
    return tuple(paths)


def _combined_text(root: Path, paths: tuple[Path, ...]) -> tuple[str, tuple[str, ...]]:
    texts: list[str] = []
    missing: list[str] = []
    for relative in paths:
        path = root / relative
        if not path.exists():
            missing.append(str(relative))
            continue
        texts.append(path.read_text(encoding="utf-8"))
    return "\n".join(texts), tuple(missing)


def _resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "check_id",
        "status",
        "scope",
        "required_hits",
        "missing_required",
        "forbidden_hits",
        "purpose",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, str]], *, output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readiness = "ready" if all(row["status"] == "ready" for row in rows) else "review"
    lines = [
        "# Paper Source Provenance Audit",
        "",
        f"Machine-readable audit: `{output_csv}`",
        "",
        f"Readiness status: `{readiness}`.",
        (
            "This audit checks source-card and manuscript-facing interpretation "
            "boundaries for Dolci DPO, Dolma sampling, full pybiber, and SmolLM3 "
            "APO/source claims. It is a provenance guardrail, not a new empirical "
            "measurement."
        ),
        "",
        "| Check | Status | Required hits | Missing required | Forbidden hits | Purpose |",
        "|---|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['check_id']} | {row['status']} | {row['required_hits']} | "
            f"{row['missing_required'] or 'none'} | {row['forbidden_hits'] or 'none'} | "
            f"{row['purpose']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_audit_paper_source_provenance(args)
    ready = sum(row["status"] == "ready" for row in rows)
    review = sum(row["status"] != "ready" for row in rows)
    print(f"Wrote source provenance audit with {ready} ready rows and {review} review rows")


if __name__ == "__main__":
    main()
