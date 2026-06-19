# Project Status Memo

Date: 2026-06-17  
Scope: Full review of EXPERIMENTS.md and all docs/experiments/

---

## What exists

All four phases have run as bounded scopes. The core pipeline is end-to-end
functional: census → teacher-forced propensity → amplification spectrum →
detector discovery. The main findings are internally consistent and the
infrastructure (SGLang generation, cached scorer, W&B logging, FDR-corrected
stage effects) is solid.

The current authority for paper-safe claim wording is
`docs/experiments/paper_claim_matrix.md`.

---

## Phase 1 (census): Closeable with caveats

Dolci SFT/DPO and Dolma 3 pretrain rates for four Tier-1 feature families are
measured and logged. The key data-side result is that DPO chosen > rejected for
rule-of-three and stock closers but NOT slop_lexicon.

**Gap:** 200-hit precision validation is incomplete. The current status artifact
(`docs/experiments/precision_validation_status.md`) has interim passes for
`contrastive_negation` and `stock_openers`, while `rule_of_three_approx`,
`slop_lexicon`, `stock_closers`, and pooled `stock_openers_closers` still block
publication-grade regex claims. The direct bounded queue attempt for
`stock_openers_closers` found zero retained hits, so that pooled feature should
be treated as derived unless a broader scan is budgeted. The full paired
Wilcoxon / logistic preference model called for in EXPERIMENTS.md was never
run; only a sign test / BH-FDR first-pass exists.

---

## Phase 2 (propensity harness): Strongest phase

The teacher-forced neutral-normalized AF grid is complete for slop_lexicon across
all four OLMo stages at 1,024 prompts. DPO peak (1.999, CI 1.446–2.837). The
4×512 free-running generation grid is complete. Compounding is measured and
confirmed for slop_lexicon (6–9x risk ratio after a prior hit).

**Gap:** contrastive_negation has only 7 reference initiations in the 5k-prompt
denominator; no AF claim is possible for it. rule_of_three_approx uses a
bounded third-item extension proxy, not open-vocabulary initiation. Temperature
grid is DPO-only at bounded scale. Full 5k×8×3-temperature grid is paused
(partial SGLang caches, not finished).

---

## Phase 3 (amplification spectrum): Bounded conclusion valid

OLMo reduced SGLang panels complete (16,384 generations). SmolLM3 no_think
production-shape panels complete (49,152 generations). Cross-ladder AF rank
correlation computed (Spearman 0.762, Pearson 0.978 over 8 shared AF values).
Instruct/Think/RL-Zero stretch comparison complete as output-only evidence.

Key conclusion: slop_lexicon is preference-amplified / dynamics-driven in OLMo,
SFT-amplified in SmolLM3. Stage localization differs by ladder and algorithm.

**Gap:** Only 8 shared AF values across both ladders; correlation is computed
over a very thin overlap. SmolLM3 "DPO" cell is APO soup + long-context merge,
not a clean single-step preference checkpoint. Full 5k×8×3 OLMo grid is
paused. Corpus-side pybiber register baselines now exist for Phase 1, but they
are not bootstrapped. Teacher-forced
coverage for contrastive_negation, stock_openers remains zero-reference in the
512-prompt SmolLM3 package. Sparse addendum (2,258-prompt SmolLM3) gives 2-3
reference initiations for several features — far too sparse for stable AF.

---

## Phase 4 (detector discovery): Conceptually complete, not validated

ModernBERT-large fine-tuned on HAP-E + generations; IG attribution over 10,000
docs; 56 HDBSCAN clusters; 10 Tier-3 matcher specs. The bounded OLMo
teacher-forced rerun now has a 512-reference-document exact sequence-mass
production pass.

**Gap:** Human perceptibility annotation package prepared (1,000 examples) but
no human has labeled any of them. The human-label Venn (detector features vs.
perceived slop) does not exist yet. OLMo detector transfer is weak (base 0.469
accuracy); OLMo SFT references score as 35% LLM-positive — a calibration
problem that means discovered OLMo-side features must be treated as candidate
artifacts until validated. The LLM-labeling cluster workflow mentioned in
EXPERIMENTS.md was not run; labeling is manual triage in a report document.
The 512-document exact rerun gives real propensity evidence for selected
candidate features, but it does not replace human perceptibility validation.

---

## Open critical gaps (ranked by impact on publishable claims)

1. **Precision validation** for core Tier-1 matchers: status table exists, but
   only 2/6 core features pass the current interim gate.
2. **Full Wilcoxon/logistic preference model** for Result A: not run; only a
   sign-test first pass exists.
3. **contrastive_negation teacher-forced AF**: zero usable denominator in both
   ladders at production scale; the feature has only output-visible evidence.
4. **Cross-ladder comparison** rests on 8 shared AF values; the stated Pearson
   0.978 is almost certainly inflated by small n and likely non-independent pairs.
5. **SmolLM3 preference cell ambiguity**: APO soup + long-context merge is not
   equivalent to a clean DPO/APO checkpoint; stage attribution is blurry.
6. **Human perceptibility annotation** for Phase 4 Tier-3 features: 0/1,000
   examples labeled; the Venn figure is proxy-only.
7. **Full OLMo generation grid** (5k×8×3): not complete; the reduced SGLang
   scope is the current basis.
8. **full pybiber on generated-output caches**: still not run as a replacement
   for the Biber-lite generated-output style layer. Phase 1 corpus-side full
   pybiber is now complete.

---

## Overall assessment

The project has a working pipeline and one clean positive result: slop_lexicon
preference-stage AF rise in OLMo appears to be dynamics-driven (chosen data not
complicit). That is a genuinely interesting finding if it survives full
validation. The infrastructure investment is high and well-documented. The main
risk is that the bounded scope — reduced generation panels, sparse teacher-forced
denominators, missing precision labels — means several headline claims rest on
thin or unvalidated evidence. Before submission, precision validation, the full
Wilcoxon model, and a real human perceptibility pass are the minimum outstanding
obligations.
