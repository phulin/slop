# Project Status Memo

Date: 2026-06-17

Scope reviewed: `README.md`, `EXPERIMENTS.md`, and all project research
documentation under `docs/experiments/`. I also inspected the generated
`.pytest_cache/README.md` and `src/slop_sftdiv.egg-info/*.txt` files; those are
tool/cache metadata, not substantive research documentation.

## Executive Status

The project has produced a coherent bounded research package, but it has not
completed the literal full scope described in `EXPERIMENTS.md`.

The strongest defensible current claim is narrow:

> In the OLMo 3 Instruct ladder, `slop_lexicon` has a bounded DPO-stage
> teacher-forced propensity peak, visible generation-side compounding, and no
> direct Dolci chosen-response enrichment. SmolLM3 no_think shows related
> style amplification, but with different stage localization, especially a
> strong SFT-level `slop_lexicon` signal.

The project is therefore best framed as an amplification-spectrum study with
feature-specific, ladder-specific conclusions. It should not be framed as a
completed proof that preference optimization universally creates slop, nor as a
completed execution of the original full-grid plan.

## Phase Status

| Area | Current status | Publishability read |
|---|---|---|
| Phase 1 corpus census | English-filtered Dolci SFT, Dolci DPO, and bounded Dolma 3 corpus measurements exist. DPO pair deltas and full 67-feature pybiber corpus-register rates exist with complete retained-row coverage. | Useful as a sampled data layer and register baseline, but not publication-grade as a precision-validated claim package. |
| Phase 1 validation | Precision gate remains incomplete, but a versioned status table now exists at `docs/experiments/precision_validation_status.md`: `contrastive_negation` and `stock_openers` pass the interim gate; four core features still block publication-grade regex claims. | This is a major blocker for headline claims using regex features, but it is now auditable feature by feature. |
| Phase 2 OLMo propensity/generation | Bounded OLMo/Dolci slice is complete: teacher-forced propensity, target-shape free-running generation, compounding, Biber-lite, style signature. | Strong enough for a bounded OLMo result, not the original full production grid. |
| Phase 3 amplification spectrum | Bounded OLMo and SmolLM3 spectra, classifications, Biber-lite signatures, compounding CIs, and cross-ladder comparison exist. | Coherent bounded conclusion; full `EXPERIMENTS.md` scope remains incomplete. |
| Phase 4 detector discovery | ModernBERT detector, IG attribution, GTE/HDBSCAN clustering, provisional Tier-3 specs, lightweight census, and bounded OLMo Tier-3 teacher-forced rerun exist. | Pipeline-executed, not validated-discovery complete. Human labels and larger denominators remain required. |
| Documentation | There is extensive audit detail and artifact naming, plus the claim matrix at `docs/experiments/paper_claim_matrix.md`. | Status language is now anchored by the matrix, but older phase reports still need to be read with bounded-scope caveats. |

## Core Evidence Now In Hand

Phase 1 corpus rates support the basic data-side landscape. Dolci DPO chosen
responses exceed rejected responses on contrastive negation, rule-of-three,
stock closers, and pooled openers/closers, but rejected responses are slightly
higher on the pooled `slop_lexicon` and stock openers. This is central because
the strongest OLMo model-side DPO signal is `slop_lexicon`, where the DPO data
does not provide a simple chosen-greater-rejected explanation.

Phase 2's strongest OLMo result is `slop_lexicon`:

- teacher-forced neutral-normalized AF: base `1.467`, SFT `1.695`, DPO
  `1.999`, final/RLVR `1.659`;
- DPO is the point-estimate peak, but CIs overlap across stages;
- free-running target-shape rates are base `0.233`, SFT `0.171`, DPO `0.229`,
  final/RLVR `0.193` per 1k generated tokens;
- prior-hit windows are about 6x-9x more likely to contain another
  `slop_lexicon` hit than no-prior windows.

Phase 3 adds a bounded cross-ladder result. OLMo classifies `slop_lexicon` as
preference-amplified and dynamics-driven; SmolLM3 no_think classifies it as
SFT-amplified, with chosen-greater-rejected preference data. The cross-ladder
AF comparison is positive but thin: 24 aligned rows, 8 shared non-missing AF
values, and support mainly from `slop_lexicon` and `rule_of_three_approx`.

Phase 4 executed the detector-discovery pipeline. The combined ModernBERT
detector performs well on held-out HAP-E and SmolLM3 generations, but transfers
poorly to OLMo generations and marks many Dolci SFT references as detector
positive. The attribution/clustering pass produced plausible candidate
families such as greetings, follow-up offers, process framing, additive
transitions, prescriptive instruction, and response-constraint wording. The
bounded OLMo Tier-3 teacher-forced rerun gives initial signals, especially
follow-up offers and process framing, but several denominators are very small.

## Claims That Are Currently Defensible

The following claims are supported if worded carefully. The current authority
for exact paper language is `docs/experiments/paper_claim_matrix.md`.

1. The OLMo 3 ladder does not show a monotonic "later checkpoint equals more
   slop" curve.
2. `slop_lexicon` is the strongest OLMo feature-level model-side signal:
   DPO has the highest teacher-forced point estimate, while Base remains
   comparable in sampled output.
3. SFT often suppresses visible Tier-1 generated markers in the OLMo target
   shape, with DPO rebounding on selected markers.
4. `slop_lexicon` self-conditioning is real in the measured generation panels.
5. SmolLM3 provides useful replication pressure, but the stage localization
   differs from OLMo.
6. Detector discovery found plausible Tier-3 candidates, but they remain
   provisional until human perceptibility, matcher precision, and larger AF
   denominator support are complete.

## Claims That Should Be Avoided Or Downgraded

The current documents do not support these stronger claims:

1. "Phase 3 is complete" without the qualifier "bounded production-scope."
2. "The full `EXPERIMENTS.md` plan has been executed."
3. "DPO universally amplifies slop."
4. "`slop_lexicon` proves DPO preference-data complicity."
5. "The OLMo/SmolLM3 cross-ladder result is a robust replication."
6. "Phase 4 discovered human-perceived slop features."
7. "Generated-output Biber-lite is equivalent to full Biber register analysis."
8. "Tier-3 teacher-forced AF magnitudes are stable enough for headline claims."

## Main Gaps

The largest scientific gap is validation of the feature measurements. The
project relies heavily on regex and surface matchers, while the precision gate
for core features remains incomplete: the current status table has 2/6 core
features passing the interim gate, and the original 200-hit-per-feature plan is
not complete. This is the first thing a reviewer will attack because it sits
underneath every corpus rate, generation rate, and propensity label.

The second gap is scope drift. The work moved pragmatically from the original
large design to bounded panels and addenda. That is scientifically acceptable
only if the paper makes the reduced estimand explicit and stops treating
optional or incomplete grids as if they are latent support for the main result.

The third gap is denominator fragility. `contrastive_negation`, stock closers,
stock openers, several sparse SmolLM3 addendum cells, and multiple Tier-3
features have too little aligned reference support for strong AF claims.

The fourth gap is causal attribution. "Dynamics-driven" is a useful label for
the observed mismatch between OLMo DPO model behavior and Dolci chosen/rejected
rates, but it does not identify an optimization mechanism. It rules out one
simple data-imitation story more than it proves a specific alternative.

The fifth gap is detector validity. Phase 4 shows that the pipeline can run,
but OLMo calibration problems, detector-positive references, broad discourse
clusters, and absent human labels mean the detector features are candidates,
not validated discoveries.

## Recommended Next Steps

1. Finish the core precision validation package: continue from the versioned
   status table, complete or demote the four blocking core features, and keep
   `stock_openers_closers` derived unless a broader direct scan is budgeted.
2. Recast the paper around the bounded estimand: "feature-level amplification
   spectra in two open ladders under reduced generation panels."
3. Add sensitivity analyses for neutral normalization, classifier thresholds,
   generation stop policy, and opportunity definitions.
4. Expand teacher-forced denominators only for features that can actually
   support AF: `contrastive_negation`, stock closers/openers, and selected
   Tier-3 candidates such as process framing and follow-up offers.
5. Complete human perceptibility and matcher precision validation before
   promoting Phase 4 features into the main taxonomy.
6. Export or version the key derived tables needed for review, because many
   claims currently rely on local artifacts and W&B references.

## Bottom Line

This is a serious, technically substantial project with a real measurement
contribution: fixed-context propensity plus free-running compounding gives a
more nuanced view than raw generation counts. The current result is already
interesting. Its publishable form, however, is narrower than the original plan:
bounded, feature-specific, and strongest for `slop_lexicon`. The remaining work
is less about running more broad sweeps and more about validation, denominator
support, scope discipline, and making the claim language match the evidence.
