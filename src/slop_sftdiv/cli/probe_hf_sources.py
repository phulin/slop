from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe Hugging Face source metadata safely.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset repo ID to inspect. Use repo::config::split for config-specific samples.",
    )
    parser.add_argument("--model", action="append", default=[], help="Model repo ID to inspect.")
    parser.add_argument("--dataset-search", action="append", default=[], help="Dataset search query.")
    parser.add_argument("--model-search", action="append", default=[], help="Model search query.")
    parser.add_argument("--sample-rows", type=int, default=0, help="Bounded streaming rows per dataset.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="source-identification")
    parser.add_argument("--wandb-job-type", default="hf-source-probe")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


@dataclass(frozen=True)
class DatasetSpec:
    raw: str
    repo_id: str
    config: str | None = None
    split: str | None = None


def _parse_dataset_spec(raw: str) -> DatasetSpec:
    parts = raw.split("::")
    if not 1 <= len(parts) <= 3 or any(not part for part in parts):
        msg = "dataset must be repo, repo::config, or repo::config::split"
        raise ValueError(msg)
    repo_id = parts[0]
    config = parts[1] if len(parts) >= 2 else None
    split = parts[2] if len(parts) == 3 else None
    return DatasetSpec(raw=raw, repo_id=repo_id, config=config, split=split)


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return str(value)


def _shape_value(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"type": "str", "chars": len(value)}
    if isinstance(value, dict):
        return {"type": "dict", "keys": list(value)[:20], "size": len(value)}
    if isinstance(value, list):
        first = _shape_value(value[0]) if value else None
        return {"type": "list", "len": len(value), "first": first}
    if isinstance(value, (int, float, bool)) or value is None:
        return {"type": type(value).__name__, "value": value}
    return {"type": type(value).__name__}


def _repo_url(repo_id: str, *, repo_type: str) -> str:
    return f"https://huggingface.co/{repo_type}s/{repo_id}" if repo_type == "dataset" else f"https://huggingface.co/{repo_id}"


def _read_card(repo_id: str, *, repo_type: str) -> dict[str, Any]:
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename="README.md")
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    lower = text.lower()
    markers = [
        "pretrain",
        "training data",
        "smoltalk",
        "tulu",
        "preference",
        "apo",
        "qwen",
        "synthetic",
        "mixture",
    ]
    hits = [marker for marker in markers if marker in lower]
    return {
        "available": True,
        "chars": len(text),
        "marker_hits": hits,
        "heading_lines": [line for line in text.splitlines() if line.startswith("#")][:20],
    }


def _dataset_configs_and_splits(dataset_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {"configs": [], "splits_by_config": {}, "error": None}
    try:
        from datasets import get_dataset_config_names, get_dataset_split_names

        configs = get_dataset_config_names(dataset_id)
        result["configs"] = configs
        for config in configs[:20]:
            try:
                result["splits_by_config"][config] = get_dataset_split_names(dataset_id, config)
            except Exception as exc:
                result["splits_by_config"][config] = [f"{type(exc).__name__}: {exc}"]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _sample_dataset_rows(spec: DatasetSpec, *, sample_rows: int) -> list[dict[str, Any]]:
    if sample_rows <= 0:
        return []
    try:
        from datasets import load_dataset
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]

    try:
        dataset = load_dataset(
            spec.repo_id,
            name=spec.config,
            split=spec.split or "train",
            streaming=True,
        )
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(dataset):
            if index >= sample_rows:
                break
            rows.append(
                {
                    "row_index": index,
                    "columns": list(row),
                    "shapes": {str(key): _shape_value(value) for key, value in row.items()},
                }
            )
        return rows
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def _dataset_probe(api: Any, spec: DatasetSpec, *, sample_rows: int) -> dict[str, Any]:
    try:
        info = api.dataset_info(spec.repo_id, files_metadata=False)
    except Exception as exc:
        return {
            "id": spec.repo_id,
            "dataset_spec": spec.raw,
            "config": spec.config,
            "split": spec.split,
            "error": f"{type(exc).__name__}: {exc}",
        }

    siblings = _safe_getattr(info, "siblings", []) or []
    return {
        "id": spec.repo_id,
        "dataset_spec": spec.raw,
        "config": spec.config,
        "split": spec.split,
        "url": _repo_url(spec.repo_id, repo_type="dataset"),
        "downloads": _safe_getattr(info, "downloads"),
        "likes": _safe_getattr(info, "likes"),
        "tags": list(_safe_getattr(info, "tags", []) or []),
        "card_data": _serializable(_safe_getattr(info, "card_data", None)),
        "siblings_count": len(siblings),
        "sibling_names_sample": [str(_safe_getattr(item, "rfilename", item)) for item in siblings[:30]],
        "card": _read_card(spec.repo_id, repo_type="dataset"),
        "configs_and_splits": _dataset_configs_and_splits(spec.repo_id),
        "sample_rows": _sample_dataset_rows(spec, sample_rows=sample_rows),
    }


def _model_probe(api: Any, model_id: str) -> dict[str, Any]:
    try:
        info = api.model_info(model_id, files_metadata=False)
    except Exception as exc:
        return {"id": model_id, "error": f"{type(exc).__name__}: {exc}"}
    siblings = _safe_getattr(info, "siblings", []) or []
    return {
        "id": model_id,
        "url": _repo_url(model_id, repo_type="model"),
        "downloads": _safe_getattr(info, "downloads"),
        "likes": _safe_getattr(info, "likes"),
        "tags": list(_safe_getattr(info, "tags", []) or []),
        "card_data": _serializable(_safe_getattr(info, "card_data", None)),
        "siblings_count": len(siblings),
        "sibling_names_sample": [str(_safe_getattr(item, "rfilename", item)) for item in siblings[:30]],
        "card": _read_card(model_id, repo_type="model"),
    }


def _search_rows(items: Iterable[Any], *, query: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        rows.append(
            {
                "query": query,
                "id": _safe_getattr(item, "id", ""),
                "downloads": _safe_getattr(item, "downloads"),
                "likes": _safe_getattr(item, "likes"),
            }
        )
    return rows


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    dataset_specs = [_parse_dataset_spec(raw) for raw in args.dataset]
    dataset_searches: dict[str, list[dict[str, Any]]] = {}
    model_searches: dict[str, list[dict[str, Any]]] = {}
    for query in args.dataset_search:
        dataset_searches[query] = _search_rows(api.list_datasets(search=query, limit=20), query=query)
    for query in args.model_search:
        model_searches[query] = _search_rows(api.list_models(search=query, limit=20), query=query)

    datasets = [_dataset_probe(api, spec, sample_rows=args.sample_rows) for spec in dataset_specs]
    models = [_model_probe(api, model_id) for model_id in args.model]
    payload = {
        "datasets": datasets,
        "models": models,
        "dataset_searches": dataset_searches,
        "model_searches": model_searches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage1", "source_identification", *args.wandb_tag],
        config={
            "datasets": args.dataset,
            "models": args.model,
            "dataset_searches": args.dataset_search,
            "model_searches": args.model_search,
            "sample_rows": args.sample_rows,
            "output": str(args.output),
        },
    )
    try:
        sampled_rows = sum(len(item.get("sample_rows", [])) for item in datasets)
        run.log(
            {
                "source_probe/datasets": len(datasets),
                "source_probe/models": len(models),
                "source_probe/dataset_searches": len(dataset_searches),
                "source_probe/model_searches": len(model_searches),
                "source_probe/sample_rows": sampled_rows,
            }
        )
        log_summary_table(
            run,
            "source_probe_datasets",
            [
                {
                    "id": item.get("id"),
                    "dataset_spec": item.get("dataset_spec"),
                    "config": item.get("config"),
                    "split": item.get("split"),
                    "url": item.get("url"),
                    "downloads": item.get("downloads"),
                    "likes": item.get("likes"),
                    "configs": len(item.get("configs_and_splits", {}).get("configs", [])),
                    "sample_rows": len(item.get("sample_rows", [])),
                    "card_markers": ",".join(item.get("card", {}).get("marker_hits", [])),
                    "error": item.get("error"),
                }
                for item in datasets
            ],
        )
        log_summary_table(
            run,
            "source_probe_models",
            [
                {
                    "id": item.get("id"),
                    "url": item.get("url"),
                    "downloads": item.get("downloads"),
                    "likes": item.get("likes"),
                    "card_markers": ",".join(item.get("card", {}).get("marker_hits", [])),
                    "error": item.get("error"),
                }
                for item in models
            ],
        )
        search_rows = [
            {"kind": "dataset", **row}
            for rows in dataset_searches.values()
            for row in rows
        ] + [
            {"kind": "model", **row}
            for rows in model_searches.values()
            for row in rows
        ]
        log_summary_table(run, "source_probe_search_results", search_rows)
    finally:
        run.finish()
    return payload


def main() -> None:
    args = build_parser().parse_args()
    run_probe(args)
    print(f"Wrote source probe to {args.output}")


if __name__ == "__main__":
    main()
