# Paper Venue Decision Checklist

Machine-readable checklist: `artifacts/paper/submission/paper_venue_decision_checklist.csv`

Venue decision status: `venue_decision_pending`.
This checklist records submission-template decisions that cannot be finalized
until a target venue is selected. It does not replace the figure, table,
manuscript, or submission-exit audits.

| Item | Status | Current evidence | Decision needed | Next action |
|---|---|---|---|---|
| target_venue | decision_pending | Paper package is venue-neutral and passes internal consistency checks. | Choose target venue and template. | Select venue before final camera-ready export. |
| figure_dimensions | decision_pending | Figure SVGs pass package hygiene, PDF/PNG export candidates exist, rendered-layout review passes, and `paper_venue_sizing_inventory.md` records current dimensions. | Single-column/two-column widths, font scaling, and final export dimensions. | Apply target template dimensions to Figure 1-Figure 5. |
| figure_export_format | decision_pending | Editable SVGs, PDF/PNG submission-export candidates, and PNG visual-review sidecars exist. | Required final format such as PDF, SVG, PNG, or EPS. | Use the existing export candidate if it matches the venue requirement; otherwise export after final sizing. |
| table_placement | decision_pending | Markdown tables and generic booktabs/tabularx LaTeX drafts pass readiness and typography reviews; `paper_venue_sizing_inventory.md` records current table environments and widths. | Table float placement, width, and whether tables use table or table*. | Adapt Table 1/Table 2/Table 4 and appendix tables to the venue template. |
| appendix_table_inclusion | decision_pending | Appendix tables A1-A3 exist as scope-control and uncertainty-support material. | Whether appendix tables are allowed in the submission or supplemental material. | Decide appendix placement after venue selection. |
| manuscript_template | decision_pending | Integrated manuscript passes structure and claim-language audits. | Venue class/template, section limits, anonymization, and supplement policy. | Copyedit manuscript against the target template without weakening claim boundaries. |
| human_labeling_requirement | decision_pending | Phase 4 annotation package is ready but currently unlabeled. | Whether the submission needs human-perceptibility labels before review. | Label the pilot/full package if detector clusters will be described as human-perceived slop. |
