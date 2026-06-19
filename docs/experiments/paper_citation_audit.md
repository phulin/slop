# Paper Citation Audit

Machine-readable audit: `artifacts/paper/references/paper_citation_audit.csv`

Readiness status: `ready`.
This audit checks cited manuscript keys, BibTeX entries, and verified source-inventory rows for the paper reference package.

| Check | Status | Observed | Missing or invalid | Notes |
|---|---|---|---|---|
| cited_keys_have_bib_entries | ready | 27 cited keys across 4 text files | none | docs/experiments/paper_intro_discussion_draft.md: 0 keys; docs/experiments/paper_manuscript_draft.md: 27 keys; docs/experiments/paper_methods_draft.md: 25 keys; docs/experiments/paper_results_draft.md: 0 keys |
| bib_keys_have_source_rows | ready | 27 BibTeX keys; 27 source rows | none | none |
| source_rows_have_bib_entries | ready | 27 source rows; 27 BibTeX keys | none | none |
| source_rows_are_populated | ready | 27 source rows | none | Source, verified-from, and notes cells must be populated. |
| source_rows_have_urls | ready | 27 source rows | none | Verified-from cells should contain URLs for paper provenance review. |
| uncited_bib_keys_recorded | ready | 0 uncited BibTeX keys | none | Not a blocker; uncited keys: none |
