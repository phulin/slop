# Paper Limitations Audit

Machine-readable audit: `artifacts/paper/manuscript/paper_limitations_audit.csv`

Readiness status: `ready`.
This audit checks that the integrated manuscript limitations section preserves the bounded-scope, pybiber, EQ-Bench, Tier-1, and Phase 4 human-validation boundaries.

| Check | Status | Required | Forbidden hits | Missing |
|---|---|---:|---|---|
| bounded_open_ladders | ready | 4/4 | none | none |
| tier1_validation_boundary | ready | 4/4 | none | none |
| sparse_denominators | ready | 3/3 | none | none |
| dpo_preference_caveat | ready | 3/3 | none | none |
| pybiber_generation_boundary | ready | 3/3 | none | none |
| eqbench_aggregate_boundary | ready | 3/3 | none | none |
| phase4_human_validation_boundary | ready | 2/2 | none | none |
| reduced_grid_boundary | ready | 3/3 | none | none |
| forbidden_overclaims | ready | 0/0 | none | none |
