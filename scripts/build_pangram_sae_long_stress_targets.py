from __future__ import annotations

import argparse
import csv
from pathlib import Path


DISCOURSE_PAIRS = [
    (
        "A city planning team should begin a bus redesign by mapping the trips people already make, not by drawing an ideal network from scratch. However, the map should not only show the busiest corridors; it should also show missed connections, long walks to stops, and neighborhoods where service ends too early. Additionally, the team should interview riders who use the system at night, because their problems often disappear in average weekday numbers. For example, a nurse leaving a hospital at midnight may care more about a safe transfer point than about shaving two minutes from a daytime route. Furthermore, the city should publish the tradeoffs before the vote, including which routes become faster and which riders need a new transfer. Such as it is, a redesign earns trust when people can see what changed, why it changed, and how complaints will be handled after launch.",
        "A city planning team should begin a bus redesign by mapping the trips people already make, not by drawing an ideal network from scratch. The map should not only show the busiest corridors; it should also show missed connections, long walks to stops, and neighborhoods where service ends too early. The team should interview riders who use the system at night, because their problems often disappear in average weekday numbers. A nurse leaving a hospital at midnight may care more about a safe transfer point than about shaving two minutes from a daytime route. The city should publish the tradeoffs before the vote, including which routes become faster and which riders need a new transfer. In practice, a redesign earns trust when people can see what changed, why it changed, and how complaints will be handled after launch.",
    ),
    (
        "A school garden can become more than a decoration if teachers connect it to ordinary lessons. However, the project should begin with a simple maintenance schedule, because a neglected bed teaches the wrong lesson by October. Additionally, each grade should own a specific task, such as testing soil, measuring plant growth, drawing pollinators, or writing instructions for younger students. For example, a math class can compare the yield from shaded and sunny beds while a language class interviews cafeteria staff about what herbs they actually use. Furthermore, the principal should budget for replacement tools and summer watering before celebrating the first harvest. In addition, families should know when they can volunteer without turning the garden into a private club.",
        "A school garden can become more than a decoration if teachers connect it to ordinary lessons. The project should begin with a simple maintenance schedule, because a neglected bed teaches the wrong lesson by October. Each grade should own a specific task, including testing soil, measuring plant growth, drawing pollinators, or writing instructions for younger students. A math class can compare the yield from shaded and sunny beds while a language class interviews cafeteria staff about what herbs they actually use. The principal should budget for replacement tools and summer watering before celebrating the first harvest. Families should know when they can volunteer without turning the garden into a private club.",
    ),
    (
        "A small clinic choosing new scheduling software should resist the most complicated package. However, it still needs a system that shows no-shows, urgent openings, and follow-up calls in one place. Additionally, the staff should test the software with the phones ringing and patients waiting, not during a quiet demonstration arranged by the vendor. For example, the front desk should try moving a child's appointment, adding an interpreter note, and printing tomorrow's lab reminders without calling a supervisor. Furthermore, the clinic should ask how the data can be exported if the contract ends. In addition, the training plan should include temporary staff, because they often handle the busiest hours.",
        "A small clinic choosing new scheduling software should resist the most complicated package. It still needs a system that shows no-shows, urgent openings, and follow-up calls in one place. The staff should test the software with the phones ringing and patients waiting, not during a quiet demonstration arranged by the vendor. The front desk should try moving a child's appointment, adding an interpreter note, and printing tomorrow's lab reminders without calling a supervisor. The clinic should ask how the data can be exported if the contract ends. The training plan should include temporary staff, because they often handle the busiest hours.",
    ),
]


NEGATION_PAIRS = [
    (
        "Lena didn't trust the inventory report, and she didn't hide that fact when the warehouse manager asked why she was still at her desk. The totals didn't match the pallets by the loading door, the return bin wasn't labeled, and the shipment log wasn't signed by the driver. She couldn't accuse anyone yet, because a bad scanner could explain part of the problem, but she wouldn't close the file just to make the afternoon meeting easier. When Omar said the missing boxes probably didn't matter, Lena didn't answer right away. She walked the aisle again, counted every blue tag, and wrote down the numbers that didn't fit. By sunset, the mistake wasn't dramatic, but it was real: three orders had been packed twice, and two customers hadn't received anything at all.",
        "Lena did not trust the inventory report, and she did not hide that fact when the warehouse manager asked why she was still at her desk. The totals did not match the pallets by the loading door, the return bin was not labeled, and the shipment log was not signed by the driver. She could not accuse anyone yet, because a bad scanner could explain part of the problem, but she would not close the file just to make the afternoon meeting easier. When Omar said the missing boxes probably did not matter, Lena did not answer right away. She walked the aisle again, counted every blue tag, and wrote down the numbers that did not fit. By sunset, the mistake was not dramatic, but it was real: three orders had been packed twice, and two customers had not received anything at all.",
    ),
    (
        "Marcus didn't want the council hearing to turn into a lecture, but he couldn't let the drainage plan pass without questions. The engineer's map didn't show the alley behind Cedar Street, and the cost estimate wasn't clear about who would repair the cracked sidewalk. Residents don't usually read every appendix, he thought, but they shouldn't have to guess whether their basements would flood again. He hadn't prepared a speech, so he didn't sound polished. Still, he wouldn't apologize for asking basic questions. When the chair tried to move on, Marcus didn't sit down. He pointed to the blank corner of the map and said the plan wasn't finished until the people living there could see themselves in it.",
        "Marcus did not want the council hearing to turn into a lecture, but he could not let the drainage plan pass without questions. The engineer's map did not show the alley behind Cedar Street, and the cost estimate was not clear about who would repair the cracked sidewalk. Residents do not usually read every appendix, he thought, but they should not have to guess whether their basements would flood again. He had not prepared a speech, so he did not sound polished. Still, he would not apologize for asking basic questions. When the chair tried to move on, Marcus did not sit down. He pointed to the blank corner of the map and said the plan was not finished until the people living there could see themselves in it.",
    ),
    (
        "The rehearsal didn't begin well. The lights weren't focused, the stage manager couldn't find the spare headset, and the lead actor hadn't learned the scene change that everyone else had practiced. Priya didn't panic, because panic wouldn't make the curtain rise any faster. She told the crew they didn't need a perfect run; they needed one honest pass through every failure point. The director wasn't pleased, but he didn't argue when the second cue failed exactly where Priya expected. By the end of the night, the show still wasn't ready, yet the team finally knew what wasn't working. That was more useful than pretending the problems didn't exist.",
        "The rehearsal did not begin well. The lights were not focused, the stage manager could not find the spare headset, and the lead actor had not learned the scene change that everyone else had practiced. Priya did not panic, because panic would not make the curtain rise any faster. She told the crew they did not need a perfect run; they needed one honest pass through every failure point. The director was not pleased, but he did not argue when the second cue failed exactly where Priya expected. By the end of the night, the show still was not ready, yet the team finally knew what was not working. That was more useful than pretending the problems did not exist.",
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
                "pair_id": f"long_{target_name}_{index:02d}",
                "variant": "marked",
                "doc_id": f"long_{target_name}_{index:02d}_marked",
                "model": "long_stress_pair",
                "text": marked,
            }
        )
        rows.append(
            {
                "target_name": target_name,
                "pair_id": f"long_{target_name}_{index:02d}",
                "variant": "plain",
                "doc_id": f"long_{target_name}_{index:02d}_plain",
                "model": "long_stress_pair",
                "text": plain,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def _write_combined(path: Path, *, target_name: str, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    marked = "\n\n".join(pair[0] for pair in pairs)
    plain = "\n\n".join(pair[1] for pair in pairs)
    rows = [
        {
            "target_name": target_name,
            "pair_id": f"combined_{target_name}",
            "variant": "marked",
            "doc_id": f"combined_{target_name}_marked",
            "model": "combined_long_stress_pair",
            "text": marked,
        },
        {
            "target_name": target_name,
            "pair_id": f"combined_{target_name}",
            "variant": "plain",
            "doc_id": f"combined_{target_name}_plain",
            "model": "combined_long_stress_pair",
            "text": plain,
        },
    ]
    columns = ["target_name", "pair_id", "variant", "doc_id", "model", "text"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build longer synthetic Pangram SAE lexical stress-pair target CSVs.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pangram_llama_sae/circuit_discovery"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _write_pairs(args.output_dir / "long_stress_discourse_marker_targets.csv", target_name="discourse_marker", pairs=DISCOURSE_PAIRS)
    _write_pairs(args.output_dir / "long_stress_negation_fragment_targets.csv", target_name="negation_fragment", pairs=NEGATION_PAIRS)
    _write_combined(args.output_dir / "combined_long_stress_discourse_marker_targets.csv", target_name="discourse_marker", pairs=DISCOURSE_PAIRS)
    _write_combined(args.output_dir / "combined_long_stress_negation_fragment_targets.csv", target_name="negation_fragment", pairs=NEGATION_PAIRS)


if __name__ == "__main__":
    main()
