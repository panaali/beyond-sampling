# T116 grouped positive triangle fallback review

Source inspection against Engine.positive confirms Grouped.select
preserves the scalar lower/upper radial estimates, trial ordering,
binomial remainder bound, tolerance comparison and every selected-rule
metadata field. Group insertion order preserves the first occurrence
of each order, so the rule-generation flag stays attached to the same
triangle; existing cached rules remain cached. Grouping modifies only
the evaluation of independent same-order positive nodes.

Batched vertices transpose times barycentric nodes, squared norms and
weighted reciprocal cubes equal the scalar formulas in real arithmetic.
Their floating reductions may differ; the saved controls appropriately
report rather than suppress that difference. The rule cap now rejects
n*n > max_nodes before evaluating, and each valid chunk obeys the cap.
The initially accepted too-small-cap fixture remains preserved.

All baseline setup/trial/node/metadata allowances remain, with an added
grouping/scatter allowance. This is not a fresh exact audit of every
Python operation. The fixed64-cone pilot retains values and statistics,
3641 fallbacks and58256 nodes; its arithmetic allowance increases by30120.
Five alternating calls per implementation support only the measured
6.066% median CPU reduction on this fixed pilot. They establish no FLOP
saving, portable speed guarantee, full-integral accuracy or floating
certificate. Existing truncation bounds keep their original scope.
