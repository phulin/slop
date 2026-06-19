# Paper Source Provenance Audit

Machine-readable audit: `artifacts/paper/submission/paper_source_provenance_audit.csv`

Readiness status: `ready`.
This audit checks source-card and manuscript-facing interpretation boundaries for Dolci DPO, Dolma sampling, full pybiber, and SmolLM3 APO/source claims. It is a provenance guardrail, not a new empirical measurement.

| Check | Status | Required hits | Missing required | Forbidden hits | Purpose |
|---|---|---:|---|---|---|
| dolci_dpo_source_card_mixture | ready | 6/6 | none | none | Records the primary-source Dolci DPO mixture and metadata needed for interpretation. |
| dolci_dpo_not_pure_human_preference | ready | 3/3 | none | none | Prevents Dolci DPO chosen/rejected contrasts from being overread as pure human preference. |
| dolma_sample_boundary | ready | 3/3 | none | none | Keeps the pretraining reference framed as a bounded retained English sample. |
| pybiber_generated_output_boundary | ready | 2/2 | none | none | Preserves the active boundary between corpus-side full pybiber and generated-output diagnostics. |
| smollm3_apo_source_boundary | ready | 2/2 | none | none | Keeps SmolLM3 preference-stage claims tied to the documented APO/source-card boundary. |
