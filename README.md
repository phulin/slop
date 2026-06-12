# SFT Divergence Slop Study

This repository implements the staged experiments in [EXPERIMENTS.md](EXPERIMENTS.md).

Stage 1 focuses on sampled corpus measurements:

- stream or read corpus records without materializing full datasets
- run Tier-1 stylistic feature matchers
- log aggregate rates and small diagnostic tables to Weights & Biases

Secrets stay local. Put `WANDB_API_KEY` in `.env`; the file is ignored by Git.

## Quick Start

```bash
uv sync --extra dev
uv run slop-census --help
```
