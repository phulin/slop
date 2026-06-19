# Paper Manuscript Structure Audit

Machine-readable audit: `artifacts/paper/manuscript/paper_manuscript_structure_audit.csv`

Readiness status: `ready`.
This audit checks the integrated manuscript's section order, length bounds, result-section coverage, captions, and appendix presence.

| Check | Status | Value | Expected | Warnings |
|---|---|---:|---|---|
| main_heading_order | ready | 11/11 | all expected main headings present in order | none |
| result_subsection_order | ready | 5/5 | all expected Result 4.x subsections present in order | none |
| total_word_count | ready | 5592 words | 4500-9000 words | none |
| abstract_word_count | ready | 182 words | 100-250 words | none |
| conclusion_word_count | ready | 213 words | 80-250 words | none |
| figure_caption_count | ready | 5 | 5 figure captions | none |
| table_caption_count | ready | 5 | 5 table captions | none |
| reproducibility_appendix_present | ready | 102 words | appendix heading with at least 20 words | none |
| heading_count | ready | 22 | main headings plus result subsections present | none |
