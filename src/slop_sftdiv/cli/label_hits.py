from __future__ import annotations

import argparse
import csv
import curses
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any

from slop_sftdiv.cli.score_labels import VALID_LABELS


LABEL_KEYS = {
    "e": "exact",
    "p": "partial",
    "f": "false_positive",
    "a": "ambiguous",
}
REQUIRED_COLUMNS = ("label", "labeler", "notes", "feature", "hit_text", "context")


@dataclass
class LabelState:
    rows: list[dict[str, str]]
    fieldnames: list[str]
    output: Path
    labeler: str
    index: int = 0
    dirty: bool = False
    status: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast terminal UI for labeling matcher hits.")
    parser.add_argument("--input", type=Path, required=True, help="Hit sample CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Labeled CSV to write/resume.")
    parser.add_argument("--labeler", required=True, help="Labeler name or initials.")
    parser.add_argument(
        "--feature",
        action="append",
        default=[],
        help="Only label this feature; repeat to include multiple features.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Zero-based row offset to open instead of first unlabeled row.",
    )
    return parser


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    _validate_columns(fieldnames)
    return rows, fieldnames


def _validate_columns(fieldnames: list[str]) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f"missing required label columns: {', '.join(missing)}")


def _load_rows(input_path: Path, output_path: Path, features: set[str]) -> tuple[list[dict[str, str]], list[str]]:
    source_path = output_path if output_path.exists() else input_path
    rows, fieldnames = _read_csv(source_path)
    if features:
        rows = [row for row in rows if row.get("feature") in features]
    return rows, fieldnames


def _write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _first_unlabeled(rows: list[dict[str, str]]) -> int:
    for index, row in enumerate(rows):
        if not (row.get("label") or "").strip():
            return index
    return max(0, len(rows) - 1)


def _progress(rows: list[dict[str, str]]) -> tuple[int, int]:
    labeled = sum(1 for row in rows if (row.get("label") or "").strip() in VALID_LABELS)
    return labeled, len(rows)


def _feature_progress(rows: list[dict[str, str]], feature: str) -> tuple[int, int]:
    feature_rows = [row for row in rows if row.get("feature") == feature]
    labeled = sum(1 for row in feature_rows if (row.get("label") or "").strip() in VALID_LABELS)
    return labeled, len(feature_rows)


def _label_counts(rows: list[dict[str, str]], feature: str) -> dict[str, int]:
    counts = {label: 0 for label in sorted(VALID_LABELS)}
    for row in rows:
        if row.get("feature") != feature:
            continue
        label = (row.get("label") or "").strip()
        if label in counts:
            counts[label] += 1
    return counts


def _next_unlabeled(rows: list[dict[str, str]], start: int, *, direction: int = 1) -> int:
    if not rows:
        return 0
    index = start
    for _ in range(len(rows)):
        index = (index + direction) % len(rows)
        if not (rows[index].get("label") or "").strip():
            return index
    return min(max(start, 0), len(rows) - 1)


def _set_label(state: LabelState, label: str) -> None:
    row = state.rows[state.index]
    row["label"] = label
    if not row.get("labeler"):
        row["labeler"] = state.labeler
    state.dirty = True
    _save(state, status=f"saved {label}")
    state.index = _next_unlabeled(state.rows, state.index)


def _save(state: LabelState, *, status: str = "saved") -> None:
    _write_rows(state.output, state.rows, state.fieldnames)
    state.dirty = False
    state.status = status


def _prompt_note(stdscr: Any, state: LabelState) -> None:
    curses.echo()
    curses.curs_set(1)
    height, width = stdscr.getmaxyx()
    prompt = "note: "
    stdscr.move(height - 1, 0)
    stdscr.clrtoeol()
    stdscr.addstr(height - 1, 0, prompt)
    current = state.rows[state.index].get("notes", "")
    if current:
        stdscr.addstr(height - 1, len(prompt), current[: max(0, width - len(prompt) - 1)])
    stdscr.refresh()
    raw = stdscr.getstr(height - 1, len(prompt), max(1, width - len(prompt) - 1))
    state.rows[state.index]["notes"] = raw.decode("utf-8", errors="replace")
    state.dirty = True
    _save(state, status="saved note")
    curses.noecho()
    curses.curs_set(0)


def _safe_addstr(stdscr: Any, y: int, x: int, text: str, attr: int = 0) -> None:
    height, width = stdscr.getmaxyx()
    if y < 0 or y >= height or x >= width:
        return
    stdscr.addstr(y, x, text[: max(0, width - x - 1)], attr)


def _draw_wrapped(stdscr: Any, y: int, title: str, text: str, *, max_lines: int) -> int:
    height, width = stdscr.getmaxyx()
    if y >= height - 1:
        return y
    _safe_addstr(stdscr, y, 0, title, curses.A_BOLD)
    y += 1
    wrap_width = max(20, width - 2)
    for line in text.splitlines() or [""]:
        for wrapped in textwrap.wrap(line, width=wrap_width, replace_whitespace=False) or [""]:
            if y >= height - 2 or max_lines <= 0:
                return y
            _safe_addstr(stdscr, y, 2, wrapped)
            y += 1
            max_lines -= 1
    return y


def _draw(stdscr: Any, state: LabelState) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    row = state.rows[state.index]
    labeled, total = _progress(state.rows)
    feature = row.get("feature", "")
    feature_labeled, feature_total = _feature_progress(state.rows, feature)
    counts = _label_counts(state.rows, feature)

    header = (
        f"{state.index + 1}/{total} labeled {labeled}/{total} | "
        f"{feature} {feature_labeled}/{feature_total} | "
        "e exact  p partial  f false  a ambiguous  n note  u clear  b back  q quit"
    )
    _safe_addstr(stdscr, 0, 0, header, curses.A_REVERSE)
    y = 2
    meta = (
        f"source={row.get('source', '')} role={row.get('role', '')} "
        f"feature={feature} subtype={row.get('subtype', '')} "
        f"label={row.get('label', '') or '<empty>'}"
    )
    _safe_addstr(stdscr, y, 0, meta, curses.A_BOLD)
    y += 1
    _safe_addstr(
        stdscr,
        y,
        0,
        "counts: " + " ".join(f"{key}={value}" for key, value in sorted(counts.items())),
    )
    y += 2
    y = _draw_wrapped(stdscr, y, "HIT", row.get("hit_text", ""), max_lines=4)
    y += 1
    context_lines = max(4, height - y - 5)
    y = _draw_wrapped(stdscr, y, "CONTEXT", row.get("context", ""), max_lines=context_lines)
    y += 1
    note = row.get("notes", "")
    if note:
        _safe_addstr(stdscr, y, 0, f"note: {note}", curses.A_DIM)
    footer = state.status or f"output: {state.output}"
    _safe_addstr(stdscr, height - 1, 0, footer[: width - 1], curses.A_REVERSE)
    stdscr.refresh()


def _run_curses(stdscr: Any, state: LabelState) -> None:
    curses.curs_set(0)
    stdscr.keypad(True)
    while True:
        _draw(stdscr, state)
        key = stdscr.getch()
        char = chr(key).lower() if 0 <= key < 256 else ""
        if char in LABEL_KEYS:
            _set_label(state, LABEL_KEYS[char])
        elif char == "u":
            state.rows[state.index]["label"] = ""
            state.dirty = True
            _save(state, status="cleared label")
        elif char == "n":
            _prompt_note(stdscr, state)
        elif char == "b" or key == curses.KEY_LEFT:
            state.index = max(0, state.index - 1)
            state.status = ""
        elif key == curses.KEY_RIGHT or char == " ":
            state.index = min(len(state.rows) - 1, state.index + 1)
            state.status = ""
        elif char == "j" or key == curses.KEY_DOWN:
            state.index = _next_unlabeled(state.rows, state.index)
            state.status = ""
        elif char == "k" or key == curses.KEY_UP:
            state.index = _next_unlabeled(state.rows, state.index, direction=-1)
            state.status = ""
        elif char == "s":
            _save(state)
        elif char == "q":
            if state.dirty:
                _save(state)
            return


def run_label_hits(args: argparse.Namespace) -> LabelState:
    rows, fieldnames = _load_rows(args.input, args.output, set(args.feature))
    if not rows:
        raise ValueError("no rows to label after applying filters")
    index = args.start if args.start is not None else _first_unlabeled(rows)
    state = LabelState(
        rows=rows,
        fieldnames=fieldnames,
        output=args.output,
        labeler=args.labeler,
        index=min(max(index, 0), len(rows) - 1),
        status=f"loaded {len(rows)} rows",
    )
    curses.wrapper(_run_curses, state)
    return state


def main() -> None:
    args = build_parser().parse_args()
    state = run_label_hits(args)
    labeled, total = _progress(state.rows)
    print(f"Labeled {labeled}/{total} rows in {state.output}")


if __name__ == "__main__":
    main()
