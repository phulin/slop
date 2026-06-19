from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MANUSCRIPT = Path("docs/experiments/paper_manuscript_draft.md")
DEFAULT_OUTPUT_CSV = Path("artifacts/paper/manuscript/paper_limitations_audit.csv")
DEFAULT_OUTPUT_MD = Path("docs/experiments/paper_limitations_audit.md")
DEFAULT_ROOT = Path.cwd()


@dataclass(frozen=True)
class LimitationSpec:
    check_id: str
    required_phrases: tuple[str, ...]


LIMITATION_SPECS = (
    LimitationSpec(
        "bounded_open_ladders",
        (
            "bounded to specific open ladders and retained samples",
            "English-filtered retained Dolma sample from an 80k scan",
            "no-think mode",
            "not structurally identical to OLMo's DPO/final path",
        ),
    ),
    LimitationSpec(
        "tier1_validation_boundary",
        (
            "Tier-1 precision validation is a scoped boundary",
            "interim gate passes",
            "rule_of_three_approx` is demoted",
            "derived convenience metric",
        ),
    ),
    LimitationSpec(
        "sparse_denominators",
        (
            "sparse reference denominators",
            "large AF point estimates",
            "treated as provisional",
        ),
    ),
    LimitationSpec(
        "dpo_preference_caveat",
        (
            "Dolci DPO chosen/rejected contrasts are not pure human-preference contrasts",
            "synthetic and Delta Learning-style construction regimes",
            "strong-model versus weak-model style",
        ),
    ),
    LimitationSpec(
        "pybiber_generation_boundary",
        (
            "full pybiber has been run over the Phase 1 corpus-side samples",
            "not over the generation caches",
            "does not make generated-output full-register claims",
        ),
    ),
    LimitationSpec(
        "eqbench_aggregate_boundary",
        (
            "EQ-Bench Slop Score is aggregate by design",
            "output-bootstrap intervals",
            "main causal outcome variable",
        ),
    ),
    LimitationSpec(
        "phase4_human_validation_boundary",
        (
            "Phase 4 detector features remain candidates",
            "human perceptibility labels are still absent",
        ),
    ),
    LimitationSpec(
        "reduced_grid_boundary",
        (
            "completed production scope is intentionally smaller",
            "reduced SGLang panels",
            "not the same evidence as a completed 5,000 prompt x 8 completion x 3 temperature OLMo grid",
        ),
    ),
)


FORBIDDEN_PHRASES = (
    "unbounded evidence",
    "complete 5,000 prompt x 8 completion x 3 temperature OLMo grid",
    "generated-output full pybiber",
    "human perceptibility labels are complete",
    "are pure human-preference contrasts",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit paper limitations coverage.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def run_audit_paper_limitations(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = args.root.resolve()
    manuscript = _resolve_path(root, args.manuscript)
    output_csv = _resolve_path(root, args.output_csv)
    output_md = _resolve_path(root, args.output_md)
    text = manuscript.read_text(encoding="utf-8") if manuscript.exists() else ""
    rows = audit_limitations(text)
    _write_csv(output_csv, rows)
    _write_markdown(output_md, rows, output_csv=args.output_csv)
    return rows


def audit_limitations(text: str) -> list[dict[str, str]]:
    section = _normalize_text(_section(text, "## 6. Limitations"))
    rows = [_audit_spec(spec, section) for spec in LIMITATION_SPECS]
    forbidden_hits = [
        phrase for phrase in FORBIDDEN_PHRASES if _normalize_text(phrase) in section
    ]
    rows.append(
        {
            "check_id": "forbidden_overclaims",
            "status": "ready" if not forbidden_hits else "review",
            "required_phrases": "0",
            "required_present": "0",
            "missing_required": "",
            "forbidden_hits": "; ".join(forbidden_hits),
        }
    )
    return rows


def _audit_spec(spec: LimitationSpec, section: str) -> dict[str, str]:
    missing_required = [
        phrase for phrase in spec.required_phrases if _normalize_text(phrase) not in section
    ]
    return {
        "check_id": spec.check_id,
        "status": "ready" if not missing_required else "review",
        "required_phrases": str(len(spec.required_phrases)),
        "required_present": str(len(spec.required_phrases) - len(missing_required)),
        "missing_required": "; ".join(missing_required),
        "forbidden_hits": "",
    }


def _section(text: str, heading: str) -> str:
    start = text.find(heading)
    if start < 0:
        return ""
    remainder_start = start + len(heading)
    next_heading = re.search(r"\n## ", text[remainder_start:])
    if next_heading:
        return text[start : remainder_start + next_heading.start()]
    return text[start:]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "check_id",
        "status",
        "required_phrases",
        "required_present",
        "missing_required",
        "forbidden_hits",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readiness = "ready" if all(row["status"] == "ready" for row in rows) else "review"
    lines = [
        "# Paper Limitations Audit",
        "",
        f"Machine-readable audit: `{output_csv}`",
        "",
        f"Readiness status: `{readiness}`.",
        "This audit checks that the integrated manuscript limitations section "
        "preserves the bounded-scope, pybiber, EQ-Bench, Tier-1, and Phase 4 "
        "human-validation boundaries.",
        "",
        "| Check | Status | Required | Forbidden hits | Missing |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        missing = row["missing_required"] or "none"
        forbidden = row["forbidden_hits"] or "none"
        lines.append(
            "| {check_id} | {status} | {required_present}/{required_phrases} | "
            "{forbidden} | {missing} |".format(
                **row,
                forbidden=forbidden,
                missing=missing,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_audit_paper_limitations(args)
    print(f"Wrote {len(rows)} limitations audit rows to {args.output_csv}")
    print(f"Wrote limitations audit summary to {args.output_md}")


if __name__ == "__main__":
    main()
