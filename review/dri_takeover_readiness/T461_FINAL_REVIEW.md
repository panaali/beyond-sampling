# Independent final review of T461

Reviewed `review/is_v4_saved_audit/check.py` (SHA256 `e7d8918f049f1ff4f01893f230801467bf2a814a61de7166a534c164e2daf329`) and `checks.json`. Separately parsed all fifteen published estimate/truth pairs at `a52da0d6a57d29aa2f76daa74c3360222f33aec2` with 60-digit Decimal arithmetic; no author code, model or RNG execution.

[measured] Independent recomputation agrees: panel F = 0.06383512783502176, draw-charged F = 0.27239070779485589; 15/15 combined-error |z| ≤ 2, maximum 1.461031267664489. Ratio range is 0.9677683248831524–1.031109004049829, so literal “within 3%” is inaccurate. Mean reported truth variance / observed squared discrepancy is 0.7652855928983014. This is a variance-scale comparison, not an observed decomposition attributing precisely that percentage of realized error to truth noise.

A distinct set-membership check adds a correction: unique triple **cardinality grows on 14/15**, but newly represented triples occur on **15/15**. For s503, three triples enter and three leave, retaining cardinality 25. Thus “added triples” and “cardinality grew” must not be conflated. The README's 13/15 matches neither measure.

[inferred] Checker logic supports saved arithmetic and committed freeze chronology. Its 100-hit scoring floor matches the author's exclusion rule; all observed truths exceed the separate 1000-hit generation target. Git ancestry and unchanged blobs do not establish historical execution order or absence of unpublished truth access. These records support the finite panel's stated scoring pass, not repeated-run calibration, a useful heavy-tail gate, a noise-free MSE estimate, full computation accounting or an ARC solution. No claim promotion is recommended.

Reproduce: `python review/dri_takeover_readiness/check_t461_decimal.py`; retained output: `t461_decimal_checks.json`. Final checker naming correction reviewed after the independent set comparison.
