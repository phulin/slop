from __future__ import annotations

import argparse
import csv
from pathlib import Path


DISCOURSE_PAIRS = [
    (
        "The city budget can support a new library branch. However, the first phase should focus on shared study rooms, reliable evening hours, and a clear plan for maintaining the building after the opening year.",
        "The city budget can support a new library branch. The first phase should focus on shared study rooms, reliable evening hours, and a clear plan for maintaining the building after the opening year.",
    ),
    (
        "A good onboarding guide should start with the tools people use every day. Additionally, it should name the person responsible for each approval so new employees are not left guessing.",
        "A good onboarding guide should start with the tools people use every day. It should also name the person responsible for each approval so new employees are not left guessing.",
    ),
    (
        "The lesson works best when students handle real materials. For example, they can compare two soil samples, record the texture and smell, and then explain which one would drain faster.",
        "The lesson works best when students handle real materials. They can compare two soil samples, record the texture and smell, and then explain which one would drain faster.",
    ),
    (
        "The restaurant should keep the menu short during the first month. Furthermore, each dish should use ingredients that can be prepared by the same morning crew without slowing service.",
        "The restaurant should keep the menu short during the first month. Each dish should use ingredients that can be prepared by the same morning crew without slowing service.",
    ),
    (
        "The museum exhibit needs objects visitors can understand quickly, such as ticket stubs, hand tools, uniforms, and annotated maps from the neighborhood.",
        "The museum exhibit needs objects visitors can understand quickly, including ticket stubs, hand tools, uniforms, and annotated maps from the neighborhood.",
    ),
    (
        "The proposal should describe the safety risks plainly. In addition, it should explain how the team will report near misses before they become expensive accidents.",
        "The proposal should describe the safety risks plainly. It should also explain how the team will report near misses before they become expensive accidents.",
    ),
    (
        "The app can be useful for families coordinating errands. Moreover, it should let a user mark a task as blocked when the store is closed or an item is out of stock.",
        "The app can be useful for families coordinating errands. It should let a user mark a task as blocked when the store is closed or an item is out of stock.",
    ),
    (
        "The training session should include a short practice round. However, the instructor should stop before the exercise becomes a memory test rather than a workflow test.",
        "The training session should include a short practice round. The instructor should stop before the exercise becomes a memory test rather than a workflow test.",
    ),
]


NEGATION_PAIRS = [
    (
        "Mara didn't answer the first call because she was checking the invoice line by line, and she didn't want to promise a refund before she understood the mistake.",
        "Mara did not answer the first call because she was checking the invoice line by line, and she did not want to promise a refund before she understood the mistake.",
    ),
    (
        "The old printer wasn't broken, but it wasn't connected to the new network, so the office manager taped a temporary note beside the screen.",
        "The old printer was not broken, but it was not connected to the new network, so the office manager taped a temporary note beside the screen.",
    ),
    (
        "Jon doesn't mind taking the late shift, but he doesn't want the schedule changed after he has already arranged child care.",
        "Jon does not mind taking the late shift, but he does not want the schedule changed after he has already arranged child care.",
    ),
    (
        "They don't need a larger truck for the delivery, and they don't need extra staff if the boxes are labeled before noon.",
        "They do not need a larger truck for the delivery, and they do not need extra staff if the boxes are labeled before noon.",
    ),
    (
        "The committee couldn't approve the grant because the budget didn't include the required insurance estimate.",
        "The committee could not approve the grant because the budget did not include the required insurance estimate.",
    ),
    (
        "Rina hadn't seen the revised floor plan, so she wasn't comfortable signing the lease before the weekend.",
        "Rina had not seen the revised floor plan, so she was not comfortable signing the lease before the weekend.",
    ),
    (
        "The repair crew wouldn't reopen the road until the inspector confirmed that the temporary barrier wasn't shifting.",
        "The repair crew would not reopen the road until the inspector confirmed that the temporary barrier was not shifting.",
    ),
    (
        "The baker didn't chill the dough long enough, so the cookies didn't keep their shape when the trays went into the oven.",
        "The baker did not chill the dough long enough, so the cookies did not keep their shape when the trays went into the oven.",
    ),
]


def _write_pairs(path: Path, *, target_name: str, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["target_name", "pair_id", "variant", "doc_id", "model", "text"]
    rows = []
    for index, (marked, plain) in enumerate(pairs, start=1):
        rows.append(
            {
                "target_name": target_name,
                "pair_id": f"{target_name}_{index:02d}",
                "variant": "marked",
                "doc_id": f"minimal_{target_name}_{index:02d}_marked",
                "model": "minimal_pair",
                "text": marked,
            }
        )
        rows.append(
            {
                "target_name": target_name,
                "pair_id": f"{target_name}_{index:02d}",
                "variant": "plain",
                "doc_id": f"minimal_{target_name}_{index:02d}_plain",
                "model": "minimal_pair",
                "text": plain,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build independent Pangram SAE lexical minimal-pair target CSVs.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pangram_llama_sae/circuit_discovery"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _write_pairs(args.output_dir / "minimal_pair_discourse_marker_targets.csv", target_name="discourse_marker", pairs=DISCOURSE_PAIRS)
    _write_pairs(args.output_dir / "minimal_pair_negation_fragment_targets.csv", target_name="negation_fragment", pairs=NEGATION_PAIRS)


if __name__ == "__main__":
    main()
