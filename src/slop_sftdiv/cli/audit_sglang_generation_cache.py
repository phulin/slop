from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OUTPUT_COLUMNS = [
    "generation_cache",
    "stage",
    "records",
    "expected_records",
    "complete",
    "passed",
    "unique_generation_keys",
    "prompt_groups",
    "incomplete_prompt_groups",
    "overfull_prompt_groups",
    "duplicate_generation_keys",
    "invalid_json_rows",
    "non_object_rows",
    "missing_required_fields",
    "wrong_backend",
    "wrong_model",
    "wrong_temperature",
    "wrong_top_p",
    "bad_completion_index",
    "bad_generated_tokens",
    "short_generated_tokens",
    "bad_feature_text_tokens",
    "empty_generation",
    "bad_features_json",
    "bad_repeated_features_json",
    "over_expected_records",
    "example_failures_json",
]

REQUIRED_FIELDS = (
    "backend",
    "completion_index",
    "feature_text_tokens",
    "features_json",
    "generated_tokens",
    "generation",
    "model",
    "prompt_id",
    "repeated_features_json",
    "source",
    "source_row_index",
    "temperature",
    "top_p",
)


@dataclass(frozen=True)
class AuditSpec:
    generation_cache: Path
    stage: str = ""
    expected_records: int | None = None
    expected_backend: str = "sglang"
    expected_model: str | None = None
    expected_temperature: float | None = None
    expected_top_p: float | None = None
    completions_per_prompt: int | None = None
    max_new_tokens: int | None = None


@dataclass
class IntegrityResult:
    generation_cache: str
    stage: str
    records: int = 0
    expected_records: int | None = None
    complete: bool = False
    passed: bool = False
    unique_generation_keys: int = 0
    prompt_groups: int = 0
    incomplete_prompt_groups: int = 0
    overfull_prompt_groups: int = 0
    duplicate_generation_keys: int = 0
    invalid_json_rows: int = 0
    non_object_rows: int = 0
    missing_required_fields: int = 0
    wrong_backend: int = 0
    wrong_model: int = 0
    wrong_temperature: int = 0
    wrong_top_p: int = 0
    bad_completion_index: int = 0
    bad_generated_tokens: int = 0
    short_generated_tokens: int = 0
    bad_feature_text_tokens: int = 0
    empty_generation: int = 0
    bad_features_json: int = 0
    bad_repeated_features_json: int = 0
    over_expected_records: int = 0
    example_failures: list[str] | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "generation_cache": self.generation_cache,
            "stage": self.stage,
            "records": self.records,
            "expected_records": "" if self.expected_records is None else self.expected_records,
            "complete": self.complete,
            "passed": self.passed,
            "unique_generation_keys": self.unique_generation_keys,
            "prompt_groups": self.prompt_groups,
            "incomplete_prompt_groups": self.incomplete_prompt_groups,
            "overfull_prompt_groups": self.overfull_prompt_groups,
            "duplicate_generation_keys": self.duplicate_generation_keys,
            "invalid_json_rows": self.invalid_json_rows,
            "non_object_rows": self.non_object_rows,
            "missing_required_fields": self.missing_required_fields,
            "wrong_backend": self.wrong_backend,
            "wrong_model": self.wrong_model,
            "wrong_temperature": self.wrong_temperature,
            "wrong_top_p": self.wrong_top_p,
            "bad_completion_index": self.bad_completion_index,
            "bad_generated_tokens": self.bad_generated_tokens,
            "short_generated_tokens": self.short_generated_tokens,
            "bad_feature_text_tokens": self.bad_feature_text_tokens,
            "empty_generation": self.empty_generation,
            "bad_features_json": self.bad_features_json,
            "bad_repeated_features_json": self.bad_repeated_features_json,
            "over_expected_records": self.over_expected_records,
            "example_failures_json": json.dumps(self.example_failures or []),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit SGLang generation JSONL caches for structural integrity."
    )
    parser.add_argument(
        "--generation-plan",
        type=Path,
        help="Generation plan CSV. Audits every generations_output row with plan metadata.",
    )
    parser.add_argument(
        "--audit-existing-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="With --generation-plan, audit only generation caches that already exist and are non-empty.",
    )
    parser.add_argument(
        "--generation-cache",
        action="append",
        type=Path,
        default=[],
        help="Generation JSONL cache to audit directly. May be repeated.",
    )
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--expected-backend", default="sglang")
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-temperature", type=float)
    parser.add_argument("--expected-top-p", type=float)
    parser.add_argument("--completions-per-prompt", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--require-max-new-tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail rows whose generated_tokens is below --max-new-tokens.",
    )
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _specs_from_plan(path: Path, *, expected_backend: str) -> list[AuditSpec]:
    specs: list[AuditSpec] = []
    for row in _read_csv_rows(path):
        generation_cache = row.get("generations_output")
        if not generation_cache:
            continue
        specs.append(
            AuditSpec(
                generation_cache=Path(generation_cache),
                stage=str(row.get("stage", "")),
                expected_records=_int_or_none(row.get("expected_generations")),
                expected_backend=expected_backend,
                expected_model=row.get("model") or None,
                expected_temperature=_float_or_none(row.get("temperature")),
                expected_top_p=_float_or_none(row.get("top_p")),
                completions_per_prompt=_int_or_none(row.get("completions_per_prompt")),
                max_new_tokens=_int_or_none(row.get("max_new_tokens")),
            )
        )
    return specs


def _specs_from_cache_args(args: argparse.Namespace) -> list[AuditSpec]:
    return [
        AuditSpec(
            generation_cache=path,
            expected_records=args.expected_records,
            expected_backend=args.expected_backend,
            expected_model=args.expected_model,
            expected_temperature=args.expected_temperature,
            expected_top_p=args.expected_top_p,
            completions_per_prompt=args.completions_per_prompt,
            max_new_tokens=args.max_new_tokens,
        )
        for path in args.generation_cache
    ]


def _generation_key(row: dict[str, Any]) -> tuple[str, str, str, int, float, float] | None:
    completion_index = row.get("completion_index")
    temperature = row.get("temperature")
    top_p = row.get("top_p")
    if completion_index is None or temperature is None or top_p is None:
        return None
    try:
        return (
            str(row.get("source", "")),
            str(row.get("prompt_id", "")),
            str(row.get("source_row_index", "")),
            int(completion_index),
            float(temperature),
            float(top_p),
        )
    except (TypeError, ValueError):
        return None


def _prompt_group_key(row: dict[str, Any]) -> tuple[str, str, str, float, float] | None:
    temperature = row.get("temperature")
    top_p = row.get("top_p")
    if temperature is None or top_p is None:
        return None
    try:
        return (
            str(row.get("source", "")),
            str(row.get("prompt_id", "")),
            str(row.get("source_row_index", "")),
            float(temperature),
            float(top_p),
        )
    except (TypeError, ValueError):
        return None


def _json_dict(value: Any) -> bool:
    if isinstance(value, dict):
        return True
    if not isinstance(value, str):
        return False
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict)


def _float_equal(left: Any, right: float | None) -> bool:
    if right is None:
        return True
    try:
        return abs(float(left) - right) <= 1e-9
    except (TypeError, ValueError):
        return False


def _add_example(result: IntegrityResult, examples: list[str], message: str, limit: int) -> None:
    if len(examples) < limit:
        examples.append(message)


def audit_cache(
    spec: AuditSpec,
    *,
    require_max_new_tokens: bool,
    max_examples: int,
) -> IntegrityResult:
    result = IntegrityResult(
        generation_cache=str(spec.generation_cache),
        stage=spec.stage,
        expected_records=spec.expected_records,
    )
    examples: list[str] = []
    seen_keys: set[tuple[str, str, str, int, float, float]] = set()
    prompt_completion_indices: dict[tuple[str, str, str, float, float], set[int]] = {}
    if not spec.generation_cache.exists():
        _add_example(result, examples, "cache missing", max_examples)
        result.example_failures = examples
        return _finalize_result(result)

    with spec.generation_cache.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            result.records += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                result.invalid_json_rows += 1
                _add_example(result, examples, f"line {line_number}: invalid JSON", max_examples)
                continue
            if not isinstance(row, dict):
                result.non_object_rows += 1
                _add_example(result, examples, f"line {line_number}: row is not an object", max_examples)
                continue

            missing_fields = [field for field in REQUIRED_FIELDS if field not in row]
            if missing_fields:
                result.missing_required_fields += 1
                _add_example(
                    result,
                    examples,
                    f"line {line_number}: missing fields {','.join(missing_fields)}",
                    max_examples,
                )

            generation_key = _generation_key(row)
            if generation_key is None:
                result.bad_completion_index += 1
                _add_example(
                    result,
                    examples,
                    f"line {line_number}: invalid generation key fields",
                    max_examples,
                )
            elif generation_key in seen_keys:
                result.duplicate_generation_keys += 1
                _add_example(
                    result,
                    examples,
                    f"line {line_number}: duplicate generation key {generation_key}",
                    max_examples,
                )
            else:
                seen_keys.add(generation_key)
                prompt_group_key = _prompt_group_key(row)
                if prompt_group_key is not None:
                    prompt_completion_indices.setdefault(prompt_group_key, set()).add(
                        generation_key[3]
                    )

            if row.get("backend") != spec.expected_backend:
                result.wrong_backend += 1
            if spec.expected_model is not None and row.get("model") != spec.expected_model:
                result.wrong_model += 1
            if not _float_equal(row.get("temperature"), spec.expected_temperature):
                result.wrong_temperature += 1
            if not _float_equal(row.get("top_p"), spec.expected_top_p):
                result.wrong_top_p += 1

            try:
                completion_index = int(row.get("completion_index"))
            except (TypeError, ValueError):
                result.bad_completion_index += 1
            else:
                if completion_index < 0 or (
                    spec.completions_per_prompt is not None
                    and completion_index >= spec.completions_per_prompt
                ):
                    result.bad_completion_index += 1

            try:
                generated_tokens = int(row.get("generated_tokens"))
            except (TypeError, ValueError):
                result.bad_generated_tokens += 1
            else:
                if generated_tokens <= 0:
                    result.bad_generated_tokens += 1
                if spec.max_new_tokens is not None and generated_tokens > spec.max_new_tokens:
                    result.bad_generated_tokens += 1
                if (
                    require_max_new_tokens
                    and spec.max_new_tokens is not None
                    and generated_tokens < spec.max_new_tokens
                ):
                    result.short_generated_tokens += 1

            try:
                feature_text_tokens = int(row.get("feature_text_tokens"))
            except (TypeError, ValueError):
                result.bad_feature_text_tokens += 1
            else:
                if feature_text_tokens <= 0:
                    result.bad_feature_text_tokens += 1

            if not row.get("generation"):
                result.empty_generation += 1
            if not _json_dict(row.get("features_json")):
                result.bad_features_json += 1
            if not _json_dict(row.get("repeated_features_json")):
                result.bad_repeated_features_json += 1

    result.unique_generation_keys = len(seen_keys)
    result.prompt_groups = len(prompt_completion_indices)
    if spec.completions_per_prompt is not None:
        for prompt_group_key, completion_indices in prompt_completion_indices.items():
            if len(completion_indices) < spec.completions_per_prompt:
                result.incomplete_prompt_groups += 1
                _add_example(
                    result,
                    examples,
                    f"prompt group {prompt_group_key} has {len(completion_indices)}/{spec.completions_per_prompt} completions",
                    max_examples,
                )
            if len(completion_indices) > spec.completions_per_prompt:
                result.overfull_prompt_groups += 1
                _add_example(
                    result,
                    examples,
                    f"prompt group {prompt_group_key} has {len(completion_indices)}>{spec.completions_per_prompt} completions",
                    max_examples,
                )
    if spec.expected_records is not None and result.records > spec.expected_records:
        result.over_expected_records = result.records - spec.expected_records
    result.example_failures = examples
    return _finalize_result(result)


def _finalize_result(result: IntegrityResult) -> IntegrityResult:
    result.complete = (
        result.expected_records is not None and result.records >= result.expected_records
    )
    failure_total = (
        result.duplicate_generation_keys
        + result.incomplete_prompt_groups
        + result.overfull_prompt_groups
        + result.invalid_json_rows
        + result.non_object_rows
        + result.missing_required_fields
        + result.wrong_backend
        + result.wrong_model
        + result.wrong_temperature
        + result.wrong_top_p
        + result.bad_completion_index
        + result.bad_generated_tokens
        + result.short_generated_tokens
        + result.bad_feature_text_tokens
        + result.empty_generation
        + result.bad_features_json
        + result.bad_repeated_features_json
        + result.over_expected_records
    )
    result.passed = failure_total == 0 and result.records > 0
    return result


def _write_csv(path: Path, rows: list[IntegrityResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_row())


def _write_summary(path: Path, rows: list[IntegrityResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for row in rows if row.passed)
    lines = [
        "# SGLang Generation Cache Integrity Audit",
        "",
        f"Passed caches: `{passed}/{len(rows)}`",
        "",
        "| Generation cache | Stage | Records | Expected | Complete | Passed | Prompt groups | Bad prompt groups | Duplicate keys | Invalid JSON | Missing fields | Wrong metadata | Bad token fields | Bad feature JSON |",
        "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        wrong_metadata = row.wrong_backend + row.wrong_model + row.wrong_temperature + row.wrong_top_p
        bad_token_fields = (
            row.bad_completion_index
            + row.bad_generated_tokens
            + row.short_generated_tokens
            + row.bad_feature_text_tokens
            + row.empty_generation
        )
        bad_feature_json = row.bad_features_json + row.bad_repeated_features_json
        expected = "" if row.expected_records is None else str(row.expected_records)
        lines.append(
            "| `{cache}` | {stage} | {records} | {expected} | {complete} | {passed} | {prompt_groups} | {bad_prompt_groups} | {duplicates} | {invalid_json} | {missing} | {wrong_metadata} | {bad_token_fields} | {bad_feature_json} |".format(
                cache=row.generation_cache,
                stage=row.stage,
                records=row.records,
                expected=expected,
                complete="yes" if row.complete else "no",
                passed="yes" if row.passed else "no",
                prompt_groups=row.prompt_groups,
                bad_prompt_groups=row.incomplete_prompt_groups + row.overfull_prompt_groups,
                duplicates=row.duplicate_generation_keys,
                invalid_json=row.invalid_json_rows,
                missing=row.missing_required_fields,
                wrong_metadata=wrong_metadata,
                bad_token_fields=bad_token_fields,
                bad_feature_json=bad_feature_json,
            )
        )
    lines.extend(
        [
            "",
            "A cache can pass integrity while still being incomplete. `Complete` only means the observed row count reaches the expected generation count.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[IntegrityResult]:
    if args.max_examples < 0:
        raise ValueError("--max-examples must be non-negative")
    specs: list[AuditSpec] = []
    plan_requested = args.generation_plan is not None
    direct_requested = bool(args.generation_cache)
    if args.generation_plan is not None:
        specs.extend(_specs_from_plan(args.generation_plan, expected_backend=args.expected_backend))
        if args.audit_existing_only:
            specs = [
                spec
                for spec in specs
                if spec.generation_cache.exists() and spec.generation_cache.stat().st_size > 0
            ]
    specs.extend(_specs_from_cache_args(args))
    if not specs:
        if plan_requested and args.audit_existing_only and not direct_requested:
            _write_csv(args.output, [])
            _write_summary(args.summary_output, [])
            print(f"Wrote 0 SGLang cache integrity audits to {args.output}; failures=0")
            return []
        raise ValueError("provide --generation-plan or at least one --generation-cache")
    rows = [
        audit_cache(
            spec,
            require_max_new_tokens=args.require_max_new_tokens,
            max_examples=args.max_examples,
        )
        for spec in specs
    ]
    _write_csv(args.output, rows)
    _write_summary(args.summary_output, rows)
    failures = sum(1 for row in rows if not row.passed)
    print(f"Wrote {len(rows)} SGLang cache integrity audits to {args.output}; failures={failures}")
    return rows


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
