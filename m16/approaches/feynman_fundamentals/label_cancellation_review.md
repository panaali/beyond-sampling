# T67 per-label cancellation diagnostic

T55 read-only source and independent saved-row arithmetic review, 2026-10-17.

`label_reference.py` intersects each gate cone with the chosen output's margins, rejecting identically tied earlier competitors. It then chooses exactly one input position above that target and constrains all others below. These rank pieces are disjoint apart from null continuous-input ties and exactly describe target correctness. Their masses are joint probabilities; no additional label prior belongs in this geometric sum. The source retains numerical geometry and GL20-bound qualifications. Full-space versus truncated-input/target comparisons require the separately reported omitted mass allowances.

Independent Decimal recomputation of raw contribution rows yields per-label reference sum0.30773074302392751 and integration-only bound1.78615e-8. Baseline label0 and label2 errors are about -0.00306009 and +0.00305512, cancelling to total -7.49419e-6 after label1. The direct target-first transition worsens label0 to -0.00323546 while the other components stay unchanged; its aggregate deterioration is therefore not solely a redistribution of otherwise improved components.

The earlier curvature/gate terminal meshes tell a different story: their component L2 errors are0.000339889/0.000431178 versus baseline0.00432411, despite worse total errors. The physical terminal-resolution improvements were real on these components; the baseline's unusually small total depended on cancellation. Both observations can hold simultaneously. `label_cancellation_review_results.json` records the independent sums without importing the author's summary code.

Component L2 is diagnostic and is not substituted for the challenge's accuracy-error metric. This does not authorize truth-selected combinations of label estimates. It supports the narrower conclusion that a finite aggregate negative can conceal substantial component improvement; robust mesh allocation or certified component errors remain open.
