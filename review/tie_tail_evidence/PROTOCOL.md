# Fixed deterministic saved contraction, no execution yet

T227 claim050c6bc3 and delegation/source-preparation0b5a07d5. Snapshot2f4c25c4
contains stage1 final60967f89 only. Reviewer T55 source/resource clearance and
published immutable gate.json must precede the sole run.

All8 model files,2 rules/model,1639 tail records/rule,6 original label schemes
and L10/30/100 are fixed. Reconstruct the four deterministic in_A/oos_AB/oos_BA/
in_AB scores, label counts and exact top100 selections; all48 within-tail
statistics;36 cross-copy scores; singleton controls; raw tail/global-summary
consistency; schema/retained duplicate-flip diagnostics; and the two prefix
summary ratio tables. Preserve every model, threshold, field and discrepancy.
Do not recompute or certify labels, scalar/vector walks, all-face integrals,
model weights, floating LP feasibility, or historical sample generation.

Each saved binary64 number is converted to its exact Fraction value. Independent
group sums, selections and ratios are exact rational arithmetic. Stable selection
uses original point encounter order to break exactly equal group sums; retain
cutoff gaps. Compare unrounded historical saved numbers with absolute tolerance
5e-12, chosen before any contraction. This tolerates ordinary historical sum/
ratio rounding; it is not an integration/model error bound or population CI.
Any mismatch stops the fixed run with the decisive values/partial records intact.
No tolerance or label changes follow automatically from a failure.

The original primary decision is computed only from these fixed records: any
of orig/s005/s010 mech ruleA-to-ruleB L100 coverage below7/10. Its interpretation
remains the frozen finite-rule test, not a universal negative. Missing-kind
points and their weights are explicitly recorded before applying original filter.

Random half-tail and random-label assignments are absent and not regenerated.
Their scores remain source-qualified summaries. Rounded composition displays
and unarchived all-face blind/CV moments are outside this numerical contraction;
shared err/top5 summary fields get consistency checks only. SOURCE_NOTES.md
qualifications remain even if every contracted number passes.

One review-lock process,19CPU soft/20CPU hard,2GiB,one thread; internal17CPU guard
leaves serialization headroom. No NumPy, author imports, random generator, model,
fit, LP, geometry, mass integration or probability evaluation. Full-process
Bash user/system timing is retained separately from in-main CPU. Atomic result
replacement and a progress log preserve failure; output directory/log are new
and cannot be overwritten or silently resumed. Final independent saved review
is required before handoff.
