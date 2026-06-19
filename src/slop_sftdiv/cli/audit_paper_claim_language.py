from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MANUSCRIPT = Path("docs/experiments/paper_manuscript_draft.md")
DEFAULT_OUTPUT_CSV = Path("artifacts/paper/claims/paper_claim_language_audit.csv")
DEFAULT_OUTPUT_MD = Path("docs/experiments/paper_claim_language_audit.md")


@dataclass(frozen=True)
class ClaimLanguageSpec:
    claim_id: str
    required_phrases: tuple[str, ...]
    caveat_phrases: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()


CLAIM_LANGUAGE_SPECS = (
    ClaimLanguageSpec(
        "C1",
        (
            "post-training corpora move sharply away from narrative and personal web prose",
            "toward nominalized, adjectival, answer-like exposition",
        ),
        ("retained Dolma sample from an 80k scan",),
    ),
    ClaimLanguageSpec(
        "C2",
        (
            "not simply an increase in hedging or intensification",
            "Hedges, amplifiers, and emphatics are lower in SFT than in Dolma",
        ),
    ),
    ClaimLanguageSpec(
        "C3",
        (
            "not in a way that makes chosen uniformly more slop-like",
            "chosen responses are more descriptive/expository",
        ),
        ("not pure human-preference",),
    ),
    ClaimLanguageSpec(
        "C4",
        (
            "EQ-Bench Slop Score is implemented as an exploratory benchmark bridge",
            "not the causal measurement layer",
        ),
    ),
    ClaimLanguageSpec(
        "C5",
        (
            "the score rises from base `5.179`",
            "to SFT `7.580`",
            "DPO `7.006`",
            "final `6.650`",
        ),
        ("bounded OLMo 512-prompt/8-completion temperature-1 panel",),
    ),
    ClaimLanguageSpec(
        "C6",
        (
            "base `2.798`",
            "SFT `7.169`",
            "DPO/APO `8.345`",
            "final `8.254`",
        ),
        ("The DPO and final intervals overlap",),
    ),
    ClaimLanguageSpec(
        "C7",
        (
            "rises from base `1.467` and SFT `1.695` to DPO `1.999`",
            "final/RLVR to `1.659`",
        ),
        ("This local-propensity peak is stage-specific",),
    ),
    ClaimLanguageSpec(
        "C8",
        (
            "BH-adjusted q-value is `0.963`",
            "not a simple imitation of higher slop rates in chosen preference responses",
        ),
        ("not pure human-preference",),
    ),
    ClaimLanguageSpec(
        "C9",
        (
            "separate fixed-context propensity from sampled-output rates",
            "Compounding is visible for the same feature",
        ),
    ),
    ClaimLanguageSpec(
        "C10",
        (
            "SmolLM3 no-think provides replication pressure but not identical stage localization",
            "Spearman `0.762`",
            "Pearson `0.978`",
        ),
        ("rather than robust universal replication",),
    ),
    ClaimLanguageSpec(
        "C11",
        (
            "candidate machine-detectable style families",
            "human perceptibility labels are still absent",
        ),
        ("until manual perceptibility labels and matcher precision checks promote them",),
    ),
    ClaimLanguageSpec(
        "C12",
        (
            "Process framing has the most stable denominator-supported signal",
            "raw AF rising from base `1.401` to final `1.957`",
            "Additive transitions have stronger denominator support",
        ),
    ),
    ClaimLanguageSpec(
        "C13",
        (
            "Follow-up offers are the largest Tier-3 signal but remain sparse",
            "11 reference initiations",
            "not a standalone headline claim",
        ),
    ),
    ClaimLanguageSpec(
        "C14",
        (
            "Formatting-only features such as list/header/bold lead-ins are retired",
            "does not make generated-output full-register claims",
        ),
    ),
)

GLOBAL_FORBIDDEN_CLAIMS = (
    "All LLMs exhibit the same slop style.",
    "All alignment methods produce the same slop style.",
    "Slop style is a single global score.",
    "DPO universally creates slop.",
    "Preference data are uniformly more slop-like than rejected responses.",
    "Dolci DPO chosen responses reveal human preference.",
    "Chosen responses are direct evidence of human taste.",
    "Alignment simply adds hedging.",
    "The EQ-Bench Slop Score is the project outcome variable.",
    "The EQ-Bench Slop Score is the causal measurement layer.",
    "EQ-Bench is the causal estimand.",
    "Phase 4 discovered human-perceived slop features.",
    "Detector clusters are human-perceived slop.",
    "Phase 4 detector clusters are validated human-perceived slop.",
    "Generated-output register proxy artifacts are equivalent to full pybiber.",
    "The reduced SGLang panels are the same as the originally proposed exhaustive",
    "We ran the full 5,000 prompt x 8 completion x 3 temperature OLMo grid.",
    "We completed the 5,000 prompt x 8 completion x 3 temperature OLMo grid.",
    "The original exhaustive OLMo generation grid is complete.",
    "rule_of_three_approx is publication-grade validated.",
    "stock_openers_closers is an independent validated matcher.",
    "All Tier-1 regex features pass precision validation.",
    "The slop_lexicon is a fully precise item-level slop detector.",
    "SmolLM3 proves universal replication.",
    "SmolLM3 replicates OLMo stage localization.",
    "Follow-up offers are a standalone headline claim.",
    "Sparse Tier-3 AF estimates are denominator-stable.",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit paper manuscript claim language.")
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def run_audit_paper_claim_language(args: argparse.Namespace) -> list[dict[str, Any]]:
    text = args.manuscript.read_text(encoding="utf-8") if args.manuscript.exists() else ""
    normalized = _normalize_text(text)
    rows = [_audit_claim(spec, normalized) for spec in CLAIM_LANGUAGE_SPECS]
    rows.extend(_audit_global_forbidden(normalized))
    _write_csv(args.output_csv, rows)
    _write_markdown(args.output_md, rows, output_csv=args.output_csv)
    return rows


def _audit_claim(spec: ClaimLanguageSpec, text: str) -> dict[str, Any]:
    missing_required = [
        phrase for phrase in spec.required_phrases if _normalize_text(phrase) not in text
    ]
    missing_caveats = [
        phrase for phrase in spec.caveat_phrases if _normalize_text(phrase) not in text
    ]
    forbidden_hits = [
        phrase for phrase in spec.forbidden_phrases if _normalize_text(phrase) in text
    ]
    status = (
        "ready"
        if not missing_required and not missing_caveats and not forbidden_hits
        else "review"
    )
    return {
        "claim_id": spec.claim_id,
        "check_type": "claim",
        "status": status,
        "required_phrases": str(len(spec.required_phrases)),
        "required_present": str(len(spec.required_phrases) - len(missing_required)),
        "caveat_phrases": str(len(spec.caveat_phrases)),
        "caveats_present": str(len(spec.caveat_phrases) - len(missing_caveats)),
        "missing_required": "; ".join(missing_required),
        "missing_caveats": "; ".join(missing_caveats),
        "forbidden_hits": "; ".join(forbidden_hits),
    }


def _audit_global_forbidden(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, phrase in enumerate(GLOBAL_FORBIDDEN_CLAIMS, start=1):
        hit = _normalize_text(phrase) in text
        rows.append(
            {
                "claim_id": f"forbidden_{index}",
                "check_type": "forbidden",
                "status": "review" if hit else "ready",
                "required_phrases": "0",
                "required_present": "0",
                "caveat_phrases": "0",
                "caveats_present": "0",
                "missing_required": "",
                "missing_caveats": "",
                "forbidden_hits": phrase if hit else "",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "claim_id",
        "check_type",
        "status",
        "required_phrases",
        "required_present",
        "caveat_phrases",
        "caveats_present",
        "missing_required",
        "missing_caveats",
        "forbidden_hits",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readiness = "ready" if all(row["status"] == "ready" for row in rows) else "review"
    claim_rows = [row for row in rows if row["check_type"] == "claim"]
    forbidden_rows = [row for row in rows if row["check_type"] == "forbidden"]
    lines = [
        "# Paper Claim Language Audit",
        "",
        f"Machine-readable audit: `{output_csv}`",
        "",
        f"Readiness status: `{readiness}`.",
        "This audit checks that the integrated manuscript carries required claim/caveat phrases and avoids the paper's explicitly forbidden overclaims.",
        "",
        "| Claim | Status | Required phrases | Caveat phrases | Missing |",
        "|---|---|---:|---:|---|",
    ]
    for row in claim_rows:
        missing_parts = [
            part
            for part in (row["missing_required"], row["missing_caveats"], row["forbidden_hits"])
            if part
        ]
        missing = _shorten("; ".join(missing_parts)) if missing_parts else "none"
        lines.append(
            "| {claim_id} | {status} | {required_present}/{required_phrases} | "
            "{caveats_present}/{caveat_phrases} | {missing} |".format(
                **row,
                missing=missing,
            )
        )
    forbidden_hits = [row for row in forbidden_rows if row["status"] != "ready"]
    lines.extend(
        [
            "",
            "Forbidden overclaim checks: `{passed}/{total}` passed.".format(
                passed=len(forbidden_rows) - len(forbidden_hits),
                total=len(forbidden_rows),
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _shorten(text: str, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def main() -> None:
    args = build_parser().parse_args()
    rows = run_audit_paper_claim_language(args)
    print(f"Wrote {len(rows)} paper claim-language rows to {args.output_csv}")
    print(f"Wrote paper claim-language summary to {args.output_md}")


if __name__ == "__main__":
    main()
