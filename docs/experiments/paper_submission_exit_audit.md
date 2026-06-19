# Paper Submission Exit Audit

Machine-readable audit: `artifacts/paper/submission/paper_submission_exit_audit.csv`

Submission status: `blocked`.
This audit separates paper-draft readiness from submission readiness by making human-validation, venue-review, and final-polish blockers explicit.

| Exit item | Status | Evidence | Required for submission | Next action |
|---|---|---|---|---|
| figure_table_finalization | venue_review_pending | Figure/table audits, rendered-layout review, PDF/PNG export candidates, table typography review, venue-decision checklist, sizing inventory, and draft artifacts present | Final venue-specific figure dimensions, export choice, and table spacing/placement | Adapt Figure 1-Figure 5 dimensions and table spacing/placement to the target venue template; use the existing PDF/PNG exports when they match venue requirements. |
| manuscript_body | ready | Manuscript structure, limitations, terminology, and numeric-claims audits | No repo-path prose, expected sections, bounded length, captions, appendix surface, limitations boundary coverage, reader-facing measurement definitions, and source-backed headline numbers | Perform final editorial polish without breaking the structure, limitations, terminology, or numeric-claims audits. |
| claim_discipline | ready | Claim language and claim-evidence audits | C1-C14 supported wording present, forbidden overclaims absent, and mapped evidence paths intact | Keep manuscript edits inside the claim matrix boundaries and preserve source-backed evidence mapping. |
| tier1_boundaries | ready | Tier-1 precision validation status | Pass/demote/derived boundaries preserved in final text and captions | Keep rule_of_three_approx exploratory and stock_openers_closers derived. |
| phase4_human_perceptibility | human_validation_pending | Phase 4 human perceptibility package and summary | Human labels collected and summarized before detector clusters are called human-perceived slop | Label the blinded pilot or full package, then rerun slop-summarize-phase4-human-labels. |
| bibliography_and_package | ready | Required paper package artifacts, citation/source audit, and reproducibility manifest | Citation/source inventories, citation audit, required artifacts, and checksum manifest present | Run slop-check-paper-package after any manuscript, citation, figure, or table edits. |
| overall_submission_status | blocked | Synthesis of submission exit rows | All exit rows ready | Resolve pending items: figure_table_finalization, phase4_human_perceptibility |
