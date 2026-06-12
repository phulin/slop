from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal

from dotenv import load_dotenv


def init_wandb(
    *,
    project: str,
    entity: str | None = None,
    run_name: str | None = None,
    config: Mapping[str, Any] | None = None,
    tags: list[str] | None = None,
    group: str | None = None,
    job_type: str | None = None,
    mode: Literal["online", "offline", "disabled", "shared"] | None = None,
):
    """Initialize W&B after loading `.env`, without requiring callers to handle secrets."""
    load_dotenv()

    import wandb

    if mode == "disabled":
        return wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=dict(config or {}),
            tags=tags,
            group=group,
            job_type=job_type,
            mode=mode,
        )

    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=False)

    return wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=dict(config or {}),
        tags=tags,
        group=group,
        job_type=job_type,
        mode=mode,
    )


def log_summary_table(run, table_name: str, rows: list[dict[str, Any]]) -> None:
    """Log a list-of-dicts table when rows exist; otherwise emit an empty count metric."""
    import wandb

    if not rows:
        run.log({f"{table_name}/rows": 0})
        return

    columns = sorted({key for row in rows for key in row})
    table = wandb.Table(columns=columns)
    for row in rows:
        table.add_data(*[row.get(column) for column in columns])
    run.log({table_name: table, f"{table_name}/rows": len(rows)})
