# SFT-Divergence Study: Localizing Stylistic Slop in the Post-Training Pipeline

**One-line summary.** Measure, for a taxonomy of LLM stylistic tics, how much of each tic is (a) inherited from pretraining data, (b) inherited from SFT/preference data, and (c) amplified by the optimization process itself — using fully open model/data ladders and a teacher-forced propensity measurement that controls for context exactly.

**Headline deliverable.** An *amplification spectrum*: for each stylistic feature, its empirical rate in pretrain / SFT / DPO-chosen / DPO-rejected data, and the model's amplification factor at each post-training stage (base → SFT → DPO → final), decomposed into per-step propensity shift vs. self-conditioning compounding.

**Current claim surface.** The paper-facing claim matrix is
`docs/experiments/paper_claim_matrix.md`. It is the current authority for
which bounded claims are headline-ready, caveated, or methods/status-only; the
reviewer-facing scope FAQ is `docs/experiments/paper_reviewer_faq.md`; the
manuscript claim-to-evidence map is
`docs/experiments/paper_claim_evidence_map.md`; the Tier-1 publication-use
policy is `docs/experiments/paper_tier1_publication_policy.md`; the current
submission readiness audit is `docs/experiments/paper_readiness_audit.md`.
Claim evidence-map path integrity is checked in
`docs/experiments/paper_claim_evidence_audit.md`.
New-reader terminology is summarized in
`docs/experiments/paper_reader_glossary.md`.
Abstract/conclusion claim alignment is checked in
`docs/experiments/paper_abstract_conclusion_audit.md`.
Reader-facing measurement terminology and interpretation boundaries are
checked in `docs/experiments/paper_terminology_audit.md`.
Headline manuscript numbers are checked against their source CSV/JSON artifacts
in `docs/experiments/paper_numeric_claims_audit.md`.
Paper-facing review/anonymity hygiene is checked in
`docs/experiments/paper_review_hygiene_audit.md`.
Source-card and provenance interpretation boundaries are checked in
`docs/experiments/paper_source_provenance_audit.md`.
Open-release inventory coverage is tracked in
`docs/experiments/paper_release_inventory.md`.
The manuscript section/figure scaffold is `docs/experiments/paper_scaffold.md`;
the submission-exit audit is
`docs/experiments/paper_submission_exit_audit.md`;
the paper-facing reproducibility checksum manifest is
`docs/experiments/paper_reproducibility_manifest.md`; the paper-package
refresh command sequence is `docs/experiments/paper_reproduction_runbook.md`;
draft manuscript tables, camera-ready table drafts, and Methods prose are
`docs/experiments/paper_tables.md`,
`docs/experiments/paper_camera_ready_tables.md`, and
`docs/experiments/paper_methods_draft.md`; draft Results prose is
`docs/experiments/paper_results_draft.md`; draft Introduction/Discussion/
Limitations prose is `docs/experiments/paper_intro_discussion_draft.md`; the
integrated manuscript draft is `docs/experiments/paper_manuscript_draft.md`;
the paper-facing figure/table manifest is
`docs/experiments/paper_figure_table_manifest.md`; the caption and
reproducibility appendix draft is
`docs/experiments/paper_caption_appendix_draft.md`;
venue-template decisions that remain blocked on final venue selection are
tracked in `docs/experiments/paper_venue_decision_checklist.md`, with
post-decision adaptation steps in
`docs/experiments/paper_venue_adaptation_runbook.md`;
figure/table readiness, rendered-layout, and generic table-typography audits
are tracked in
`docs/experiments/paper_figure_readiness_audit.md`,
`docs/experiments/paper_figure_visual_review.md`,
`docs/experiments/paper_table_readiness_audit.md`, and
`docs/experiments/paper_table_typography_review.md`;
the citation-slot plan is `docs/experiments/paper_citation_plan.md`; citation
and source-inventory integrity is checked in
`docs/experiments/paper_citation_audit.md`; checked reference-source and BibTeX drafts are
`docs/experiments/paper_reference_sources.md` and
`docs/experiments/paper_references.bib`. The repeatable paper-package
consistency check is `uv run slop-check-paper-package --root /home/user/slop`;
it verifies required paper artifacts, manuscript/table path hygiene,
manuscript-structure, limitations, terminology, and numeric-claims audits,
figure/table references, figure/table audit status, C1-C14 claim-map coverage,
cited BibTeX keys, and linked local paths. Figure 2
intervals are regenerated with:
`uv run slop-summarize-eqbench-intervals --bootstrap-samples 500 --seed 1729`
before rerunning `uv run slop-render-paper-figures`.

---

## 0. Core hypotheses

- **H1 (inheritance vs. amplification).** Slop features decompose into two classes: inherited (rate in model output ≈ rate in SFT targets; e.g., markdown habits) and amplified (model rate ≫ SFT-target rate; predicted: contrastive negation, rule-of-three, stock lexical items).
- **H2 (preference-signal complicity).** For amplified features, DPO-chosen responses exhibit higher feature rates than DPO-rejected. If H2 fails while amplification still appears at the DPO stage, amplification is attributable to optimization dynamics (mode-seeking) rather than the preference signal.
- **H3 (compounding).** Free-running emission rates exceed teacher-forced propensity predictions for register-locking features (lexical style, stock openers/closers, and recurring rhetorical forms), i.e., slop is partly self-conditioning.
- **H4 (stage localization).** Amplification concentrates at the preference-optimization stage, not the SFT stage.

Every hypothesis is informative in both directions; there is no null-result failure mode for the study as a whole.

## 1. Models, checkpoints, and data

Primary ladder — **OLMo 3 7B, Instruct path** (fully open data at every stage; Dolma 3 pretraining + Dolci post-training suite with separate open mixes per stage):

| Stage | Checkpoint | Paired training data |
|---|---|---|
| Base | `allenai/Olmo-3-1025-7B` | Dolma 3 (sampled, stratified) |
| SFT | `allenai/Olmo-3-7B-Instruct-SFT` | `allenai/Dolci-Instruct-SFT-7B` |
| DPO | `allenai/Olmo-3-7B-Instruct-DPO` | `allenai/Dolci-Instruct-DPO-7B` (chosen/rejected pairs) |
| Final (RLVR) | `allenai/Olmo-3-7B-Instruct` | `allenai/Dolci-Instruct-RL-7B` |

Two OLMo 3-specific properties to design around:

- **Delta-Learning DPO data.** OLMo 3's preference pairs are constructed by pairing strong-model responses (chosen) with weak-model responses (rejected) — e.g., Qwen3-32B vs. Qwen3-0.6B in the Think path — rather than human/RM-ranked same-model responses. This *reinterprets* Result A (§3): chosen-vs-rejected differences measure "strong-vs-weak model style that DPO distills," not human preference per se. It also adds a new named angle: **cross-model slop transmission** — slop patterns in the chosen responses are Qwen's fingerprint being distilled into OLMo, directly measurable by running our matchers on chosen responses vs. Qwen3-32B's known fingerprint. Verify the exact construction of `Dolci-Instruct-DPO-7B` (vs. Think) from the dataset card before locking interpretation.
- **Length pre-filtering.** AI2 filters Instruct preference pairs to cap chosen-rejected length differences (~100 tokens), partially controlling the length confound in Result A for us; keep length as a covariate anyway and report its residual effect.
- **Model merging caveat.** SFT checkpoints are produced by linearly merging two learning-rate variants; stage attribution is to the *merged* released checkpoint, which is fine for our purposes but should be stated.

Bonus within-family comparison (stretch): OLMo 3 releases a parallel **Think path** (`Olmo-3-7B-Think-SFT/-DPO/-Think`) and **RL Zero** checkpoints from the same base. Running the Tier-1 matchers (final-response text only, reasoning traces excluded) across Instruct vs. Think vs. RL Zero paths tests whether reasoning-oriented post-training produces a different slop profile from the same base — and RL Zero gives a no-SFT control cell that the original design lacked.

Replication ladder — **SmolLM3-3B** (Hugging Face; fully open with public pretraining data mixture, training configs, and intermediate checkpoints including mid-training and SFT stages at `HuggingFaceTB/SmolLM3-3B-checkpoints`; SFT data = SmolTalk2; preference stage = APO on the Tulu-3 preference dataset plus synthetic Qwen3-32B/0.6B pairs). This is a *stronger* cross-validation than a same-org ladder: different organization, different pretraining corpus, different scale (3B vs. 7B), and a different preference algorithm (APO vs. DPO). Features that amplify on both ladders are properties of preference optimization generally, not of one recipe. Caveats: SmolLM3's final model involves checkpoint averaging (model soup) plus a long-context merge, so its "post-preference" cell is a merged artifact; and its dual think/no_think mode means generations must be collected in no_think mode for comparability. Alternative replication ladder if SmolLM3's stage checkpoints prove too entangled by merging: **Apertus-8B** (fully open weights and data, intermediate checkpoints, SFT + QRPO preference alignment — a third distinct preference algorithm).

Reference corpora:

- **Pretrain reference:** stratified sample of Dolma 3, *tagged by source domain* (web/CC, forums/Q&A, wiki, scientific PDFs/olmOCR, code excluded). The forums/Q&A stratum is the designated "assistant-register" pretrain baseline; all pretrain rates reported per-stratum (register confound control). For the SmolLM3 ladder, sample its published pretraining mixture equivalently.
- **Human parallel reference:** HAP-E human–AI parallel corpus (Brown et al., Hugging Face, with pre-extracted Biber features) — human and LLM texts from common prompts; used for detector training (Phase 4) and as a topic-controlled human baseline.
- **SFT provenance split:** tag Dolci-Instruct-SFT examples by subset provenance (synthetic/LLM-generated vs. human-written, identifiable from mixture metadata). Note Dolci skews heavily synthetic — SFT responses are largely generated by external frontier/open models — which strengthens the framing that "inherited slop" is substantially *cross-model transmitted* slop. Report all SFT-data rates per provenance subset; the human-written subset (if large enough; else HAP-E) is the closest true target distribution.

## 2. Feature set

**Tier 1 — revised Phase 1 seed slop patterns (regex/POS matchers):**

1. Contrastive negation "not X, but Y" / "it's not just X — it's Y" (port the EQ-Bench two-stage regex+POS matcher).
2. Slop lexicon (EQ-Bench over-represented word and trigram lists: "delve", "tapestry", "testament to", "it's important to note", …) — track as per-1k-token rates per item and as a pooled index.
   Addendum after the 2026-06-17 matcher audit: keep this historical
   `slop_lexicon` frozen for comparability; additionally report
   `slop_lexicon_v2_candidate` and exploratory `eqbench_slop_score`, a
   document-level 0-100 EQ-Bench Slop Score using vendored EQ-Bench word/trigram
   lists, leaderboard normalization ranges, and portable not-X-but-Y regexes.
   For final-output benchmark comparisons, use `uv run slop-eqbench-score` so
   outputs are filtered, concatenated, and scored in the EQ-Bench aggregate
   shape rather than averaged as row-level corpus features.
3. Rule-of-three approximation: coordinated triple spans of the form X, Y(,) and Z. Phase 1 reports the current regex approximation with caveats rather than a full NP/VP/clause POS-chunk split.
4. Sycophantic/stock openers and closers ("Great question", "I hope this helps", "In conclusion", "Overall,").

Retired from the active feature surface: list/header/bold lead-in formatting. Deferred from Phase 1 core claims: punctuation/rhythm and generic hedging. These should not block the revised Phase 1 close-out.

**Tier 2 — Biber/linguistic features:** use full `pybiber` extraction for
Biber/register analysis. The active implementation exposes the 67 pybiber
features through `uv run slop-pybiber-full`; the old sampled surface-proxy
layer is retired. Full pybiber is currently a corpus-side Phase 1 register
surface for retained pretrain/SFT/DPO samples; generated-output full pybiber is
not part of the active paper claim surface.

**Tier 3 — discovered features (from Phase 4):** clusters extracted from detector attribution; added to the pipeline with hand-written or classifier-based matchers.

Every feature gets: (i) a span matcher, (ii) a defined *opportunity context* (where the pattern could begin — see §4), (iii) a precision estimate from 200 hand-checked matches (target ≥ 0.9; refine until met).

## 3. Phase 1 — Frequency census (grounding)

**What:** run the revised Tier-1 matchers over pretrain strata, SFT targets
(split by available provenance/source metadata), DPO-chosen, DPO-rejected, and
later generations from each checkpoint. Run full pybiber register features over
the retained Phase 1 corpus-side samples only: pretrain, SFT target,
DPO-chosen, and DPO-rejected.

**Execution status (2026-06-18).** The OLMo 3 data-side Phase 1 rerun is
complete over detector-backed English-filtered fixed samples: 40,000 Dolci SFT
targets, 40,000 Dolci DPO pairs expanded to chosen/rejected rows, and 5,740
retained Dolma 3 pretraining documents from the 80k scan. The canonical Tier-1
artifacts are
`artifacts/stage1/census/feature_rates_by_corpus.parquet`,
`artifacts/stage1/census/feature_rates_by_stratum.parquet`,
`artifacts/stage1/census/preference_pair_deltas.parquet`, and
`artifacts/stage1/census/census_summary.md`. Full pybiber outputs now exist for
all three retained samples under `artifacts/stage1/census/*_pybiber_full.csv`
and `*_pybiber_full_long.csv`; pybiber returned complete coverage on the
English-filtered samples: 5,740 Dolma rows, 40,000 SFT rows, and 80,000 DPO
rows across the 67-feature surface. The token-weighted register analysis is
`docs/experiments/phase1_pybiber_register_analysis.md`, with selected-feature
document-bootstrap intervals in
`artifacts/stage1/census/phase1_pybiber_register_intervals.csv`; its main read is that
alignment data shift away from narrative/personal web prose toward
nominalized, adjectival, answer-like exposition rather than simply adding
hedging. The detailed readable report is `docs/experiments/phase1_conclusions.md`;
the caveat/status audit is `docs/experiments/phase1_gap_audit.md`. The
integrated paper limitations boundary is audited in
`docs/experiments/paper_limitations_audit.md`.

**Normalization:** per-1k-tokens (default), per-sentence (for clause-level constructions), and per-opportunity (see §4). Report all three for Tier-1.

**Named result A — preference-signal complicity (H2):** paired comparison of feature rates in chosen vs. rejected responses *within preference pairs* (paired Wilcoxon signed-rank, per feature, FDR-corrected). Because pairs share a prompt, topic is controlled by construction. Also fit a per-pair logistic model (chosen ~ feature deltas) to estimate each feature's marginal association with being preferred, controlling for length — keep length as a covariate even though Dolci pre-filters length gaps, and report its residual effect. **Interpretation under Delta Learning:** where chosen/rejected come from strong vs. weak external models (per §1), this measures which stylistic features the contrastive signal teaches — i.e., the strong model's fingerprint being distilled — rather than human preference. Run the same analysis on the SmolLM3 ladder's Tulu-3 preference subset (RM/human-ranked) to recover the human-preference reading; the contrast between the two preference-data regimes is itself a result. **Named result A′ — cross-model slop transmission:** compare Tier-1 feature rates in Dolci chosen responses against the generating models' published fingerprints (EQ-Bench per-model slop profiles) to quantify which upstream model's tics the preference data imports.

**Output:** rate table (features × corpora), chosen-vs-rejected forest plot.

**Go/no-go check:** if Tier-1 matcher precision can't reach 0.9 on a feature, demote it; if chosen/rejected shows nothing for *any* feature including length (a known reward-model bias), suspect a harness bug before believing it.

## 4. Phase 2 — Propensity harness (the novel measurement)

Two measurements per (feature, checkpoint):

**4a. Teacher-forced propensity (decoding-free).**
Run held-out SFT target texts through each checkpoint with teacher forcing (held out **by prompt**, with near-duplicate prompt filtering via MinHash to prevent paraphrase leakage). For each feature *f*:

- Define **opportunity positions** O_f: token positions where the pattern *could* initiate (contrastive negation: post-clause-boundary positions; rule-of-three: candidate third-item continuations; lexical items: all positions, or POS-gated). Opportunity definitions are part of the feature spec and frozen before measurement.
- Define **initiating token sets** I_f(context): the token prefixes that commit to the pattern (e.g., for contrastive negation after a clause boundary: {"It's not", "This isn't", "not just", …} tokenized; built by enumerating pattern matches in the corpora and collecting their first 1–3 tokens, then deduplicating).
- At each o ∈ O_f, record p_model(I_f | prefix) — total probability mass on initiating continuations — and y(o) ∈ {0,1}, whether the reference text actually initiates the pattern there.

**Amplification factor:** AF_f = mean_o[p_model(I_f|o)] / mean_o[y(o)], with cluster-bootstrap CIs (resample documents). AF_f = 1 ⇒ calibrated to the SFT target distribution; AF_f = 3 ⇒ "given identical context, the model wants to start this pattern 3× as often as its training targets do."

Validity checks: (i) calibration sanity — for a basket of neutral control n-grams ("for example", "such as"), AF should ≈ 1 at the SFT checkpoint; (ii) the harness must reproduce known amplification on positive controls ("delve", contrastive negation) before being trusted on discovered features; (iii) report AF computed against both all-SFT and human-written-SFT references.

**4b. Free-running emission.**
Production generation uses bounded SGLang panels rather than the original all-cells grid: OLMo 3 uses a 1,024-prompt × 2-completion main panel at temperature 0.7, a 512-prompt × 1-completion temperature panel over {0.0, 0.7, 1.0}, and a 256-prompt × 2-completion long-output compounding panel at temperature 0.7; SmolLM3 no_think uses a 512-prompt × 8-completion production panel at temperature 1.0 plus the matched temperature/compounding panels needed for H3. Top-p = 0.95 and max 1024 tokens are fixed and reported. Measure realized per-opportunity feature rates in generations. The original 5k-prompt × 8-completion × 3-temperature OLMo grid is an optional SGLang addendum, not a blocker for Phase 3 production conclusions.

**Named result B — compounding decomposition (H3):** simulate expected free-running rate from teacher-forced per-step propensities under an independence assumption (per-opportunity initiation ≈ AF_f × base rate); compare to observed free-running rate. Excess = self-conditioning compounding. Direct test: measure P(initiate f | f already occurred earlier in this generation) vs. P(initiate f | no prior occurrence), within-prompt, per checkpoint. Also report the temperature dependence of realized AF (one plot).

**Compute estimate:** teacher forcing remains bounded by measured denominator support. Production generation is the reduced SGLang grid above, not the 5k × 8 × 3-temperature grid; the latter is retained only as an optional addendum. Tier-1 first; Biber/Tier-3 reuse cached generations. The Think/RL Zero path comparison is a reduced SGLang stretch panel.

## 5. Phase 3 — Amplification spectrum (main result)

Assemble: features × {data rates from Phase 1} × {AF and free-running rates per checkpoint from Phase 2}.

Classify each feature:
- **Inherited:** high SFT-data rate, AF ≈ 1 across stages.
- **SFT-amplified:** AF > 1 already at SFT checkpoint.
- **Preference-amplified:** AF jumps base→…→DPO/final (H4), cross-referenced with chosen-vs-rejected complicity (H2) to assign cause: *signal-driven* (complicit preference data) vs. *dynamics-driven* (non-complicit data, amplification anyway).
- **Compounding-dominant:** modest AF, large free-running excess (H3).

Statistics: cluster bootstrap (document-level) for all CIs; Benjamini–Hochberg FDR across the full feature set; paired designs wherever corpora share prompts. Replicate the full spectrum on the SmolLM3-3B ladder (no_think mode); report feature-level rank correlation of AFs between ladders (the model-to-model variation result, now also a DPO-vs-APO algorithm comparison). Stretch: Instruct vs. Think vs. RL Zero path comparison within OLMo 3 (§1).

## 6. Phase 4 — Detector-driven feature discovery

Runs after the Phase 1–3 pipeline is validated; feeds Tier-3 features back into it.

**Execution status (2026-06-18).** Phase 4 now has a real detector-driven run under the 24h A100 cap. ModernBERT-large was fine-tuned on HAP-E plus OLMo/SmolLM3 generation/reference slices, evaluated on held-out HAP-E and external OLMo/SmolLM3 sources, attributed with integrated gradients over 10,000 generated documents, embedded with GTE-large, clustered with HDBSCAN, and converted into provisional Tier-3 matcher specs plus a generated-output census. The run also produced a 1,000-example annotation package, a proxy detector-vs-perceived Venn against the Shaib taxonomy, and an OLMo base/SFT/DPO/final exact sequence-mass teacher-forced Tier-3 AF rerun over 512 SFT-reference documents under `artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/`. All four stage summaries, the stage grid, the primary comparison, and paired stage effects are complete. The true human-label Venn still requires annotators, but the human-perceptibility protocol, package-readiness audit, redacted annotator handoff bundle, coordinator execution checklist, 20-per-candidate pilot sheet, and reproducible label summarizer now exist: `docs/experiments/phase4_human_perceptibility_protocol.md`, `docs/experiments/phase4_human_annotation_readiness.md`, `docs/experiments/phase4_human_annotation_handoff.md`, `docs/experiments/phase4_human_labeling_execution_checklist.md`, `docs/experiments/phase4_human_perceptibility_summary.md`, `uv run slop-audit-phase4-human-package`, `uv run slop-materialize-phase4-human-handoff`, and `uv run slop-summarize-phase4-human-labels`. Primary artifacts live under `artifacts/phase4/modernbert_detector_combined_v2_clean/`; the reader report is `docs/experiments/phase4_detector_discovery_report.md`; the Tier-3 teacher-forced addendum is `docs/experiments/phase4_tier3_teacher_forced_addendum.md`; the requirement-by-requirement audit is `docs/experiments/phase4_completion_audit.md`; and the production runbook is `docs/experiments/phase4_production_runbook.md`. Remaining non-compute-complete items are Shaib-taxonomy human annotation/Venn analysis, larger post-validation Tier-3 AF denominator expansion for sparse features, and the SAE stretch path.

1. **Detector:** fine-tune ModernBERT-large on HAP-E (topic-controlled human vs. LLM pairs) + generations from our checkpoints vs. their human/SFT counterparts. Evaluate cross-model generalization (train on HAP-E's Llama/GPT-4o outputs, test on OLMo 3 and SmolLM3 outputs) to confirm it learned style, not model identity.
2. **Attribution (cheap path, default):** integrated-gradients token attribution over ≥10k documents; extract top-attribution spans; embed spans (e.g., GTE-large) and cluster (HDBSCAN); auto-label clusters with an LLM + manual review.
3. **Attribution (SAE path):** train a BatchTopK SAE on the detector's penultimate-layer residual activations; rank latents by ablation effect on the LLM-class logit; interpret via max-activating examples.
4. **Human-perceptibility validation:** for each discovered cluster, annotate 100 examples against the Shaib et al. slop taxonomy; report which clusters humans perceive as slop vs. detector-only ("machine-detectability") features. The Venn diagram of detector features vs. human-perceived slop is a standalone figure.
5. Write matchers + opportunity definitions for the top ~10 perceptible clusters; rerun Phases 1–3 on them.

## 7. Sequencing and milestones

| Week | Milestone |
|---|---|
| 1–2 | Corpora downloaded/stratified (Dolma 3 sample, Dolci stages, SmolTalk2); Tier-1 matchers ported + precision-validated; Phase 1 census on data corpora |
| 3 | Chosen-vs-rejected result (Results A, A′) on Dolci + Tulu-3-preference (SmolLM3 ladder); go/no-go review |
| 3–5 | Teacher-forced harness built; positive-control validation ("delve", contrastive negation, neutral controls) |
| 5–7 | Reduced production Phase 2 grid on OLMo 3 Instruct ladder; compounding decomposition (Result B); optional full-grid addendum only if compute budget allows |
| 7–8 | SmolLM3 ladder replication; amplification spectrum assembled (main figure) |
| 8–12 | Phase 4 detector + discovery loop; Tier-3 features through pipeline; stretch: Think/RL Zero path comparison |
| 12–14 | Analysis, robustness (temperature sweep, provenance-split references), writeup |

The thin-slice ordering is deliberate: the only genuinely novel measurement code (4a) is validated on known-answer patterns in week 3–5, before anything downstream depends on it.

## 8. Risks and mitigations

- **Opportunity-set gaming:** AF is sensitive to how O_f is defined. Freeze definitions pre-measurement; report AF under one alternative O_f per feature as a sensitivity analysis.
- **Tokenizer mismatch across ladders:** initiating sets I_f are tokenizer-specific; build per-tokenizer, verify coverage by checking that I_f captures ≥95% of actual pattern onsets in each corpus.
- **Synthetic SFT data muddies "target":** addressed by provenance split (§1); if human-written subsets are too small or entangled, fall back to HAP-E human texts as the reference for a secondary AF. Reframe positively: with Dolci, "inherited" slop is largely *cross-model transmitted* slop (Result A′).
- **Checkpoint merging blurs stage attribution:** both ladders use model merging (OLMo 3 at SFT; SmolLM3 post-APO soup + long-context merge). Attribution is to released stage artifacts as deployed; note in writeup, and prefer the OLMo 3 ladder for fine-grained stage claims since its merges stay within-stage.
- **SmolLM3 dual-mode contamination:** collect all SmolLM3 generations in no_think mode and strip any reasoning markup before feature extraction; verify mode compliance on a sample.
- **Detector learns topic/model-ID:** parallel-corpus training + cross-model evaluation (§6.1); discovered clusters validated for human perceptibility before promotion.
- **Compute overrun:** Tier-1-only is a publishable core; Biber and Tier-3 are additive.

## 9. Deliverables

1. Amplification-spectrum figure + rate tables (main result; tests H1, H4).
2. Preference-complicity results: chosen-vs-rejected feature analysis under both preference-data regimes (Delta-Learning Dolci vs. RM-ranked Tulu-3) (H2), plus cross-model slop-transmission analysis (A′).
3. Compounding decomposition + temperature dependence (H3).
4. Cross-ladder comparison (model-to-model and DPO-vs-APO variation in slop fingerprints); stretch: Instruct/Think/RL Zero path comparison within OLMo 3.
5. Detector-feature vs. human-perception Venn analysis + Tier-3 taxonomy additions.
6. Paper claim matrix with evidence source, strength, caveat, and allowed
   wording for each supported claim.
7. Manuscript scaffold and paper figure/table inventory.
8. Open release: matchers, opportunity definitions, harness code, cached per-checkpoint statistics.
