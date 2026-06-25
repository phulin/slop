from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from build_pangram_sae_long_stress_targets import NEGATION_PAIRS


REPLACEMENTS = [
    (re.compile(r"\bdid not\b"), "didn't"),
    (re.compile(r"\bwas not\b"), "wasn't"),
    (re.compile(r"\bwere not\b"), "weren't"),
    (re.compile(r"\bcould not\b"), "couldn't"),
    (re.compile(r"\bwould not\b"), "wouldn't"),
    (re.compile(r"\bshould not\b"), "shouldn't"),
    (re.compile(r"\bdo not\b"), "don't"),
    (re.compile(r"\bdoes not\b"), "doesn't"),
    (re.compile(r"\bhad not\b"), "hadn't"),
]


def _contract_first_n(text: str, count: int) -> tuple[str, int]:
    remaining = int(count)
    total = 0
    out = text
    while remaining > 0:
        changed = False
        for pattern, replacement in REPLACEMENTS:
            if remaining <= 0:
                break

            def repl(_match: re.Match[str]) -> str:
                nonlocal remaining, total, changed
                if remaining <= 0:
                    return _match.group(0)
                remaining -= 1
                total += 1
                changed = True
                return replacement

            out = pattern.sub(repl, out, count=remaining)
        if not changed:
            break
    return out, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build controlled contraction-density sweep targets for the Pangram negation SAE path.")
    parser.add_argument("--output-csv", type=Path, default=Path("artifacts/pangram_llama_sae/circuit_discovery/negation_density_sweep_targets.csv"))
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 2, 4, 8, 16, 32])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expanded = "\n\n".join(pair[1] for pair in NEGATION_PAIRS)
    rows = []
    for level in args.levels:
        text, actual = _contract_first_n(expanded, int(level))
        rows.append(
            {
                "target_name": "negation_fragment",
                "pair_id": "negation_density_sweep",
                "variant": f"contract_{actual:02d}",
                "requested_contractions": int(level),
                "actual_contractions": actual,
                "doc_id": f"negation_density_sweep_contract_{actual:02d}",
                "model": "negation_density_sweep",
                "text": text,
            }
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "target_name",
        "pair_id",
        "variant",
        "requested_contractions",
        "actual_contractions",
        "doc_id",
        "model",
        "text",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
