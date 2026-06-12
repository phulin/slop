"""Lightweight corpus readers for sampled corpus measurements.

The module is intentionally dependency-light. Local JSONL works with the
standard library; Hugging Face datasets are imported only when requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


DEFAULT_TEXT_FIELDS = (
    "text",
    "messages",
    "response",
    "completion",
    "answer",
    "content",
    "output",
    "solution",
    "outputs",
)
DEFAULT_PROMPT_FIELDS = ("prompt", "instruction", "question", "input", "query", "messages", "source_prompt")
DEFAULT_CHOSEN_FIELDS = ("chosen", "accepted", "preferred", "winner")
DEFAULT_REJECTED_FIELDS = ("rejected", "discarded", "non_preferred", "loser")
DEFAULT_ID_FIELDS = ("id", "uid", "uuid", "doc_id", "document_id", "source_id")
PROMOTED_METADATA_FIELDS = (
    "source",
    "source_dataset",
    "stratum",
    "provenance",
    "text_role",
    "domain",
    "dataset",
    "dataset_source",
    "data_source",
    "original_dataset",
    "preference_type",
    "chosen_model",
    "rejected_model",
    "version",
    "created",
    "added",
)
JSON_METADATA_FIELDS = ("metadata", "doc", "attributes")


@dataclass(frozen=True)
class CorpusSource:
    """Input corpus definition for local JSONL or Hugging Face datasets."""

    kind: str
    name: str
    path: str | None = None
    hf_name: str | None = None
    hf_config: str | None = None
    split: str | None = None
    streaming: bool = True
    data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None
    revision: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def jsonl(cls, path: str | Path, *, name: str | None = None, **metadata: Any) -> "CorpusSource":
        path = str(path)
        return cls(kind="jsonl", name=name or Path(path).stem, path=path, metadata=metadata)

    @classmethod
    def hf(
        cls,
        hf_name: str,
        *,
        name: str | None = None,
        hf_config: str | None = None,
        split: str | None = None,
        streaming: bool = True,
        data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None,
        revision: str | None = None,
        **metadata: Any,
    ) -> "CorpusSource":
        return cls(
            kind="hf",
            name=name or hf_name,
            hf_name=hf_name,
            hf_config=hf_config,
            split=split,
            streaming=streaming,
            data_files=data_files,
            revision=revision,
            metadata=metadata,
        )


@dataclass(frozen=True)
class SamplingConfig:
    """Controls deterministic row selection.

    ``strategy="first"`` is cheapest and works well for already shuffled inputs.
    ``strategy="hash_reservoir"`` keeps the rows with the smallest stable hashes,
    using O(cap) memory and deterministic output for finite streams.
    """

    cap: int | None = None
    seed: int = 0
    strategy: str = "first"
    max_scan: int | None = None

    def __post_init__(self) -> None:
        if self.cap is not None and self.cap < 0:
            raise ValueError("cap must be non-negative")
        if self.max_scan is not None and self.max_scan < 0:
            raise ValueError("max_scan must be non-negative")
        if self.strategy not in {"first", "hash_reservoir"}:
            raise ValueError("strategy must be 'first' or 'hash_reservoir'")


@dataclass(frozen=True)
class ExtractedRecord:
    """Normalized corpus row with metadata suitable for tabular logging."""

    text: str
    prompt: str | None
    chosen: str | None
    rejected: str | None
    record_id: str
    row_index: int
    source: str
    source_kind: str
    split: str | None
    fields: Mapping[str, str | None]
    metadata: Mapping[str, Any]
    raw: Mapping[str, Any] | None = None

    def wandb_row(self, *, include_raw: bool = False) -> dict[str, Any]:
        row: dict[str, Any] = {
            "source": self.source,
            "source_kind": self.source_kind,
            "split": self.split,
            "record_id": self.record_id,
            "row_index": self.row_index,
            "text": self.text,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "text_chars": len(self.text),
            "prompt_chars": len(self.prompt or ""),
            "chosen_chars": len(self.chosen or ""),
            "rejected_chars": len(self.rejected or ""),
            "text_field": self.fields.get("text"),
            "prompt_field": self.fields.get("prompt"),
            "chosen_field": self.fields.get("chosen"),
            "rejected_field": self.fields.get("rejected"),
        }
        row.update(self.metadata)
        if include_raw:
            row["raw"] = self.raw
        return row


def iter_corpus_records(
    source: CorpusSource,
    sampling: SamplingConfig | None = None,
    *,
    text_fields: Sequence[str] = DEFAULT_TEXT_FIELDS,
    prompt_fields: Sequence[str] = DEFAULT_PROMPT_FIELDS,
    chosen_fields: Sequence[str] = DEFAULT_CHOSEN_FIELDS,
    rejected_fields: Sequence[str] = DEFAULT_REJECTED_FIELDS,
    id_fields: Sequence[str] = DEFAULT_ID_FIELDS,
    keep_raw: bool = False,
) -> Iterator[ExtractedRecord]:
    """Yield normalized records from ``source`` under deterministic sampling."""

    sampling = sampling or SamplingConfig()
    rows = iter_source_rows(source)
    sampled_rows = _sample_rows(rows, source=source, sampling=sampling, id_fields=id_fields)
    for row_index, row in sampled_rows:
        extracted = extract_record(
            row,
            source=source,
            row_index=row_index,
            text_fields=text_fields,
            prompt_fields=prompt_fields,
            chosen_fields=chosen_fields,
            rejected_fields=rejected_fields,
            id_fields=id_fields,
            keep_raw=keep_raw,
        )
        if extracted is not None:
            sample_key = _sampling_key(
                row,
                source=source,
                row_index=row_index,
                id_fields=id_fields,
                seed=sampling.seed,
            )
            yield replace(
                extracted,
                metadata={
                    **extracted.metadata,
                    "sampling_strategy": sampling.strategy,
                    "sampling_seed": sampling.seed,
                    "sampling_cap": sampling.cap,
                    "sampling_max_scan": sampling.max_scan,
                    "sample_key": f"{sample_key:016x}",
                },
            )


def iter_source_rows(source: CorpusSource) -> Iterator[Mapping[str, Any]]:
    if source.kind == "jsonl":
        if source.path is None:
            raise ValueError("jsonl source requires path")
        yield from iter_jsonl_rows(source.path)
        return
    if source.kind == "hf":
        yield from iter_hf_dataset_rows(source)
        return
    raise ValueError(f"unsupported corpus source kind: {source.kind}")


def iter_jsonl_rows(path: str | Path) -> Iterator[Mapping[str, Any]]:
    """Stream JSONL objects from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
            yield row


def iter_hf_dataset_rows(source: CorpusSource) -> Iterator[Mapping[str, Any]]:
    """Stream rows from a Hugging Face dataset when ``datasets`` is installed."""

    if source.hf_name is None:
        raise ValueError("hf source requires hf_name")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Hugging Face dataset input requires `datasets` to be installed") from exc

    kwargs: dict[str, Any] = {
        "path": source.hf_name,
        "name": source.hf_config,
        "split": source.split,
        "streaming": source.streaming,
        "data_files": source.data_files,
        "revision": source.revision,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    dataset = load_dataset(**kwargs)
    for row in dataset:
        if isinstance(row, Mapping):
            yield row


def extract_record(
    row: Mapping[str, Any],
    *,
    source: CorpusSource,
    row_index: int,
    text_fields: Sequence[str] = DEFAULT_TEXT_FIELDS,
    prompt_fields: Sequence[str] = DEFAULT_PROMPT_FIELDS,
    chosen_fields: Sequence[str] = DEFAULT_CHOSEN_FIELDS,
    rejected_fields: Sequence[str] = DEFAULT_REJECTED_FIELDS,
    id_fields: Sequence[str] = DEFAULT_ID_FIELDS,
    keep_raw: bool = False,
) -> ExtractedRecord | None:
    """Extract text/prompt/preference fields from common corpus row shapes."""

    prompt, prompt_field = _first_text(row, prompt_fields, preferred_roles=("user", "system"))
    chosen, chosen_field = _first_text(row, chosen_fields, preferred_roles=("assistant",))
    rejected, rejected_field = _first_text(row, rejected_fields, preferred_roles=("assistant",))
    text, text_field = _first_text(row, text_fields, preferred_roles=("assistant",))

    if text is None:
        text = chosen or rejected or prompt
        text_field = chosen_field or rejected_field or prompt_field
    if text is None:
        return None

    record_id = _record_id(row, id_fields=id_fields, source=source, row_index=row_index)
    fields = {
        "text": text_field,
        "prompt": prompt_field,
        "chosen": chosen_field,
        "rejected": rejected_field,
    }
    metadata = {
        "corpus_name": source.name,
        "hf_name": source.hf_name,
        "hf_config": source.hf_config,
        "streaming": source.streaming if source.kind == "hf" else None,
    }
    metadata.update(_row_metadata(row))
    metadata.update(source.metadata)

    return ExtractedRecord(
        text=text,
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        record_id=record_id,
        row_index=row_index,
        source=source.name,
        source_kind=source.kind,
        split=source.split,
        fields=fields,
        metadata=metadata,
        raw=row if keep_raw else None,
    )


def _sample_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    source: CorpusSource,
    sampling: SamplingConfig,
    id_fields: Sequence[str],
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    if sampling.cap == 0:
        return
    if sampling.strategy == "first":
        for row_index, row in enumerate(rows):
            if sampling.max_scan is not None and row_index >= sampling.max_scan:
                break
            if sampling.cap is not None and row_index >= sampling.cap:
                break
            yield row_index, row
        return
    if sampling.cap is None:
        for row_index, row in enumerate(rows):
            if sampling.max_scan is not None and row_index >= sampling.max_scan:
                break
            yield row_index, row
        return

    heap: list[tuple[int, int]] = []
    heap_rows: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row_index, row in enumerate(rows):
        if sampling.max_scan is not None and row_index >= sampling.max_scan:
            break
        key = _sampling_key(row, source=source, row_index=row_index, id_fields=id_fields, seed=sampling.seed)
        item = (-key, row_index)
        if sampling.cap is None:
            heapq.heappush(heap, item)
            heap_rows[item] = row
        elif len(heap) < sampling.cap:
            heapq.heappush(heap, item)
            heap_rows[item] = row
        elif item > heap[0]:
            removed = heapq.heapreplace(heap, item)
            heap_rows.pop(removed, None)
            heap_rows[item] = row

    for item in sorted(heap, key=lambda item: (-item[0], item[1])):
        _, row_index = item
        row = heap_rows[item]
        yield row_index, row


def _first_text(
    row: Mapping[str, Any],
    fields: Sequence[str],
    *,
    preferred_roles: Sequence[str] = (),
) -> tuple[str | None, str | None]:
    for field_name in fields:
        value = _lookup_path(row, field_name)
        text = _coerce_text(value, preferred_roles=preferred_roles)
        if text:
            return text, field_name
    return None, None


def _lookup_path(row: Mapping[str, Any], field_name: str) -> Any:
    current: Any = row
    for part in field_name.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _coerce_text(value: Any, *, preferred_roles: Sequence[str] = ()) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        role_parts = [
            _coerce_text(item, preferred_roles=())
            for item in value
            if isinstance(item, Mapping)
            and str(item.get("role", "")).lower() in {role.lower() for role in preferred_roles}
        ]
        if role_parts:
            text = "\n".join(part for part in role_parts if part)
            return text or None
        parts = [_coerce_text(item, preferred_roles=preferred_roles) for item in value]
        text = "\n".join(part for part in parts if part)
        return text or None
    if isinstance(value, Mapping):
        for key in ("content", "text", "value"):
            text = _coerce_text(value.get(key), preferred_roles=preferred_roles)
            if text:
                return text
    return None


def _row_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for field_name in PROMOTED_METADATA_FIELDS:
        value = _lookup_path(row, field_name)
        if _is_metadata_scalar(value):
            metadata[field_name] = value

    for field_name in JSON_METADATA_FIELDS:
        value = _lookup_path(row, field_name)
        parsed = _coerce_metadata_mapping(value)
        if parsed is None:
            continue
        for key, item in parsed.items():
            if _is_metadata_scalar(item):
                metadata[f"{field_name}.{key}"] = item
    metadata["inferred_stratum"] = _infer_stratum(metadata)
    return metadata


def _coerce_metadata_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped.startswith("{"):
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, Mapping):
            return parsed
    return None


def _is_metadata_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _infer_stratum(metadata: Mapping[str, Any]) -> str:
    haystack = " ".join(
        str(value).lower()
        for value in metadata.values()
        if isinstance(value, (str, int, float, bool)) and value not in ("", None)
    )
    if not haystack:
        return "unknown"
    if any(marker in haystack for marker in ("stack_edu", "cranecode", "code", "github")):
        return "code"
    if any(marker in haystack for marker in ("olmocr", "science", "scientific", "paper", "arxiv")):
        return "scientific"
    if any(marker in haystack for marker in ("wiki", "wikipedia")):
        return "wiki"
    if any(marker in haystack for marker in ("forum", "reddit", "stackexchange", "stack_exchange", "qa", "q&a")):
        return "forums_qa"
    if any(
        marker in haystack
        for marker in ("common_crawl", "cc-main", "full_cc", "dclm", "c4", "web", "crawl")
    ):
        return "web_cc"
    return "unknown"


def _record_id(
    row: Mapping[str, Any],
    *,
    id_fields: Sequence[str],
    source: CorpusSource,
    row_index: int,
) -> str:
    for field_name in id_fields:
        value = _lookup_path(row, field_name)
        if value is not None:
            return str(value)
    return f"{source.name}:{source.split or 'none'}:{row_index}"


def _sampling_key(
    row: Mapping[str, Any],
    *,
    source: CorpusSource,
    row_index: int,
    id_fields: Sequence[str],
    seed: int,
) -> int:
    record_id = _record_id(row, id_fields=id_fields, source=source, row_index=row_index)
    payload = f"{seed}\0{source.name}\0{source.split or ''}\0{record_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)
