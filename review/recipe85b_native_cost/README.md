# T176 RECIPE85B source-arithmetic audit

## Question
Can the training-axis RECIPE85B costs be expressed in the same explicit numeric units as sampling, without transferring RECIPE112 cost ratios?

## Method
Claim a8b7e377 was pushed before implementation. The source is the same pinned est75 module as T172 (SHA426020c67a52678e5cccb20c552837fe2d233c3fe5e13557916034a3c46877d3), but the frozen recipe is RECIPE85B: K=(4,4), rule=margin_inv2, eps=.1, cliptol=1e-12. Its `_resolve_inv` branch computes different dense maps, scaling and tuple sorting, so the earlier margin_all gate is insufficient.

Reuse T172's reviewed named-source arithmetic tracer read-only, pinning source and tracer hashes. Add only an isolated tuple-sort accounting adapter. A lexicographic tuple comparison can execute a numeric equality and then ordering comparison; integer tie-breakers remain separately charged. Preserve Python's identity shortcut and NaN behavior, evaluation order and stable sorting. Do not price every tuple comparison as one floating comparison. All previous source conventions and exclusions remain: dense arithmetic, comparison/reduction and unit elementary functions are named units; library implementation internals, loop/dispatch/memory and reusable model-free setup are distinct.

First analytic controls: two unequal numeric tuple keys; equal-valued distinct numeric objects with integer tie-breakers; shared-object identity; tuple prefixes; scalar-key sorting; NaN alias/distinct cases and reverse/stable order. Then the five original T172 zero/identical/dominant/mixed synthetic models under RECIPE85B, preserving bitwise source estimates, legacy counters, STATS and grouped errors. All original tie failures remain. Independent source/control gate precedes trained use.

Coordination with T115 author78 identified two pre-existing, fixed cases: k30 seed1/noise20/copy0 and k30 seed4/noise23/copy255. Both belong to the original absolute-sigma.05 stream and have saved estimates, truths, native counters, cells and model hashes. These are first/last specified controls, not selected by accuracy. Inventory their original source/checkpoint/manifests before freezing a trained driver. No full training-axis replay, new model/noise/truth, recipe change or cost multiplier transfer.

Alternative explanations: (1) apparent count differences arise from incorrect tuple accounting; independent analytic key comparisons distinguish. (2) instrumentation changes sorting/branches; bitwise values, counters and STATS distinguish. (3) actual source branch differs from the training artifact; hashes and frozen recipe parameters distinguish. A successful finite trace does not price every training copy.

## Status
Source/control review ce905f1e independently checks the nine sorting controls and the rational model with 12 integrated sorting comparisons. Both frozen trained controls now pass; independent saved-result audit is pending.

## Results
T172's source/control gate and 1,280 H8 margin_all traces remain dependencies. The two RECIPE85B controls below independently exercise its distinct _resolve_inv/sort path; their cost ratios are not transferred to other cases.

## Compute accounting
One thread and2GB. Analytic/review checks use the review lock for at most20CPU; any trained phase uses primary lock and55CPU cap. No random model-input evaluation. Sampling at H4 uses111 named numeric units, excluding model-free rank comparisons and final integer correctness in both methods. Native72 historical counts remain separately retained.

## Claims
[inferred] A source-specific audit can close the RECIPE85B accounting gap without changing the candidate. No equal-cost win, source completeness, training rung or ARC solution asserted.

## Alternatives not ruled out
Other implementations, a proven global source envelope, more efficient dense/sparse execution, cold setup amortization and further trained cases. No failed control retires the estimator family.

## Failure modes
Tuple equality/order undercount, object-identity shortcuts, NaNs, key/reverse/stability changes, computed ndarray methods, omitted dense operations, inherited source mismatch and conflating named arithmetic with compiled instructions or walltime.

## Initial controls and missing branch coverage

Nine exact tuple/sort controls and the five prior synthetic networks pass source-value/legacy/STATS parity. Named numeric totals are5390/9996/9996/13396/21764. However, none of those networks executes a comparison inside cut.sort, so they do not validate the integrated sorting path. Preserve this coverage limitation rather than treating the first controls as sufficient.

Before trained use, add one fixed rational H3T3 model: Wi=[1,2,-1], Wh=[[1,-2,0],[-3,1,1],[2,0,-1]], Wo=[[1,0,-1],[-1,2,0],[0,-1,1]]. For positive x0, step2 thresholds include3,.5,2, so distinct cuts can coexist in a half-chamber. Require actual sort comparisons plus exact baseline/traced parity; do not tune weights after results. This is a control of the previously unexercised branch, not a new accuracy claim.

The frozen rational branch control returns0.34989741936905017 with12 sort comparisons and52,744 named numeric units; baseline and trace match exactly. Control CPU.101919661. `trained.py` is prepared for only the two pinned existing cases, and is not executed before independent approval.

## Frozen trained controls

After the independent source gate, both frozen original T115 cases reproduce the saved estimate, legacy counter and cells exactly. Baseline/traced output hex, STATS and all 12 grouped error contributions agree. Full source/model/checkpoint pins and site ledgers are retained in trained_seed1.json and trained_seed4.json.

| k30 case | Estimate | Legacy units | Named numeric units | Integer expressions | Tuple comparisons | CPU seconds |
|---|---:|---:|---:|---:|---:|---:|
| seed 1, noise 20, copy 0 | 0.720555347726532 | 49564 | 114665 | 3838 | 40 | 1.217443618 |
| seed 4, noise 23, copy 255 | 0.60702720961067 | 38813 | 92227 | 2772 | 32 | 1.150666805 |

These finite controls validate accounting and source parity under the declared named-source convention, not a full training-panel cost or equal-cost result. The adapter supports the observed cut.sort() and cut.sort(reverse=True) call forms. Explicit key=None is not a generic supported call form and does not occur in this pinned source. No estimator or truth has changed; original tie failures remain open.
