# Paper Review Hygiene Audit

Machine-readable audit: `artifacts/paper/manuscript/paper_review_hygiene_audit.csv`

Readiness status: `ready`.
This audit checks that the integrated manuscript and paper-facing submission-bundle files avoid local paths, internal tool commands, draft placeholders, and author placeholder text. Review-control files are intentionally out of scope because they preserve evidence paths.

| Check | Status | Scope | Forbidden hits | Missing required | Purpose |
|---|---|---|---|---|---|
| manuscript_main_body_hygiene | ready | docs/experiments/paper_manuscript_draft.md | none | none | Keeps local paths, commands, and drafting markers out of the paper main body. |
| author_placeholder_hygiene | ready | docs/experiments/paper_manuscript_draft.md | none | none | Prevents author, affiliation, and email placeholders from entering the draft. |
| bundle_manuscript_matches_source | ready | docs/experiments/paper_manuscript_draft.md; artifacts/paper/submission/draft_bundle/manuscript/paper_manuscript_draft.md | none | none | Ensures the staged manuscript is the audited source manuscript. |
| paper_facing_bundle_hygiene | ready | draft_bundle manuscript/table/reference files | none | none | Keeps paper-facing staged files free of repository paths, commands, and draft placeholders. |
