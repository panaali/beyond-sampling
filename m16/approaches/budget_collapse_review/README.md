# T44 budget-matched tie-slicing diagnostic

## Question
Does the gain decline from H16T4 to T5 and T6 persist when the nominal compute
budget is fixed? Claim9dec2e5 was pushed before work. This is a baseline audit,
not a new deductive recipe or a ladder pass.

## Method: frozen first phase
Historical T2 uses1024 faces per kind, legacy suffix-discounted FPE costs and
free model-free geometry. Its4096-face follow-up is not a matched-budget study.
The legacy walker has known first-index/tolerance and incomplete-cost defects.
Preserve those historical results; do not rescale them into purported measured
equal-budget estimates. Use the corrected Python sci_tao_v2 walker read-only,
SHA256 aea5f9008fe9cc8a5436226f1057b5e0527f119d832df536127d8370bc7968f9,
with pinned sci_tao geometry a78a45c1fae015507fb7b6a9764064a9b2425f52ff599f378408640f0617170e.

Models are public H16T{4,5,6}, seeds0..4, trained2^30, already downloaded under
T45 with explicit manifest hashes. No best-seed selection or new truths. Reuse
T2 independent truth counts and uncertainty, with their insufficiency explicit.

Before estimator outcomes, run one independent cost-only calibration of64 faces
per kind/model, fixed Sobol seeds44001/44008. Do not inspect/use calibration
masses for allocation or estimates. Freeze equal final face counts per kind:
n=max(16,floor((B-setup-pilot)/(meanCost23+meanCost12+nodeProxy))), for B16000
and64000 FPE. Pilot and construction are charged to EVERY estimator call even
when the diagnostic reuses identical pilot calculations across rule replicates.
If setup+pilot>=B, report infeasible, not a negative gain. No outcome-adaptive
allocation. Save all counts/cost plans BEFORE estimate/error calculations.

Cost unit is sci_tao_v2.fpe_flops(H,T); walker work uses its observed operation
categories and CDF50-operation proxy. Setup is the shared explicit30million-
operation face-table proxy. Additionally charge100*T operations per generated
face (Sobol+face mapping+aggregation proxy), plus configuration enumeration
32*T*(T-1)/2*(T-1) and256 allocation operations. These declared analytic proxies
are not verified portable instruction counts; report their components and both
cold and warm totals. No free pilot/model-dependent advice. Geometry table
roundoff/truncation and walker epsilon remain uncertified. Preserve setup CPU
and rule CPU separately; cached diagnostic execution is not a cold-time claim.

Primary outputs:16 scrambled Sobol rules per model/budget, seeds44100..44115
for kind23 and+7 for12; pair these seeds across T/budgets only for analysis.
Record every actual cost; budgets are targeted in expectation, not a hard
cap. If mean actual cost deviates more than10% from target for any cell, keep
that result as budget-mismatch evidence and freeze a cost-only correction
before any rerun. Do not stop a prefix based on its observed cost or error.
No interpolation between gains, no estimate clipping.

First phase stops after cost calibration and invariant controls, <=20CPU on
review lock. Subsequent bounded cycles operate one T at a time, <=100CPU each
on primary lock, single thread,2GB; partial completed rows saved if deadline.
Before the first production run freeze source and plan hashes. No expanded
model/noise sweep or fresh giant MC.

## Controls and alternatives before results
1. Historical collapse may partly reflect legacy accounting: report new observed
   walker counts and cold/warm proxy differences, never merge metric labels.
2. Setup may dominate at16k: preserve infeasible/mismatched cells and64k outcome;
   do not silently amortize setup or choose a favorable budget afterward.
3. Truth uncertainty may dominate small errors: report variance gain separately
   from bias-inclusive MSE; do not replace noisy nonpositive debiased MSE by a
   hidden floor. Bootstrap truth and rule uncertainty and report unresolvedness.
4. Cost calibration may mispredict new rule cost: fixed counts, observed mismatch,
   no optional stopping. Only cost-based replanning is a permitted rescue.
5. Walker/interface defects can mimic bias: reuse T30 exact first-index/tiny-gap
   fixtures as controls, and compare a small trained common-face batch with
   legacy outputs while retaining any disagreement. No broad family retirement.

## Statistical protocol (frozen before outcomes)
For each model/budget report p(1-p)/meanActualCost divided by sample variance,
and debiased MSE mean((estimate-p)^2)-truthSE^2 when positive. These are different
metrics. No resolved-best selection. Bootstrap2000 times, seed44021, paired
rule indices across the panel and independent normal truth perturbations with
recorded SE, clipped only the perturbed probability to[0,1]. If>2.5% bootstrap
MSE denominators are nonpositive, MSE gain interval is unresolved. Report the
nonpositive fraction, never discard those draws to sharpen the CI.
Report each seed's gain and the geometric mean across five seeds separately.
Decline factors are gain(T)/gain(T+1), for4->5 and5->6, both budgets and both
metrics; same bootstrap draws preserve shared truth/rule dependence within a
model. Five model seeds are fixed panel, not a population-generalization claim.

## Status
Initial correlated-rule panel completed; independent review caught a seed
namespace defect before statistical acceptance. Clean rescue completed;
independent point-statistic review requested. Finite H16T4/5/6 phase only.

## Frozen independent-namespace rescue
T55 found that the inherited kind12 offset+7 causes rule r's kind12 scramble
to equal rule r+7's kind23 scramble. Thus the16 rule estimates share inputs;
ordinary iid-rule variance/bootstrap inference is unjustified. This defect
also occurs in historical ht_seeds/est.py. Preserve all original480 results,
their source and33.27159CPU production cost; their point descriptions may be
reported but their ordinary rule-bootstrap CIs cannot be used as confidence
statements. No claim that this explains all historical gain differences.

Freeze kind23 seeds44100..44115 and kind12 seeds144100..144115, disjoint across
all16 replicates within each model/budget. Retain intentional same-rule pairing
across T/budget/model cases. Cost-only calibration remains unchanged (its64-face
pilot used different seeds; no cross-replicate pilot estimates enter outcomes).
Use EXACT same saved face counts, cost definitions, models, budgets, statistical
recipe and diagnostics. Save to separate independent_panel_T*.json artifacts,
preserving old panels. One bounded<=100CPU primary-lock cycle perT, no new
truths or model sweeps. Do not choose between old/new estimates by accuracy.

## Results
The disjoint-namespace results preserve a substantial T4->T5 decline at64k,
and show why a fixed threshold has no budget-independent meaning. Geometric
means across the five specified seeds, corrected cold FPE proxies:

| Nominal budget | T | Variance gain | MSE gain [conditional95% CI] |
| ---: | ---: | ---: | --- |
| 16000 | 4 | 2.555 | 2.676 [2.183,3.837] |
| 16000 | 5 | 1.246 | 1.247 [.947,1.927] |
| 16000 | 6 | .802 | .798 [.630,1.190] |
| 64000 | 4 | 41.413 | 43.060 [28.923,61.822] |
| 64000 | 5 | 5.045 | 5.036 [3.240,7.657] |
| 64000 | 6 | 1.692 | 1.858 [1.268,3.347] |

Threshold-free MSE decline factors: at16k, T4/T5 is2.145 [1.422,3.111]
and T5/T6 is1.562 [.985,2.592]; at64k,8.551 [4.945,14.473] and
2.710 [1.335,4.468]. Variance decline factors at64k are8.209 [5.354,11.994]
and2.982 [1.645,4.556]. These are fixed five-seed panel intervals conditional
on the explicit numerical-cost and approximate truth-uncertainty model, not
population coverage guarantees. They cannot certify a deductive rung.

Per-seed MSE gains (seeds0,1,2,3,4, no best-seed selection):

| Budget | T | Gains |
| ---: | ---: | --- |
| 16000 | 4 | 2.598,2.574,1.896,3.171,3.412 |
| 16000 | 5 | 1.911,3.864,.675,.935,.648 |
| 16000 | 6 | .598,1.447,.794,.715,.660 |
| 64000 | 4 | 36.918,59.605,80.142,29.119,28.829 |
| 64000 | 5 | 8.521,7.127,2.834,5.425,3.469 |
| 64000 | 6 | 1.620,1.652,3.380,1.873,1.309 |

All30 mean budgets lie between.96805 and1.00103 of target; none triggered the
frozen10% replanning criterion. The comparison is approximately matched, not
an exact-cost equality. The T5seed1/64k truth SE exceeds half the estimator SD;
that cell retains its unresolved-truth flag rather than switching to a better
metric. All point denominators are positive. Full uncertainty/diagnostics are
in `independent_analysis.json`; original correlated panel remains in
`analysis.json` with inference explicitly invalidated.

The30million-operation table proxy alone costs12500,10000,8333.33 FPE at
T4,5,6 respectively. This dominates16k and is explicitly declared, not a
validated instruction count. Warm costs in every row expose this assumption;
these same rows do NOT constitute a new warm-budget-matched experiment.
Changing cost assumptions would require new counts/runs for matched budgets.

Reproduce with the pinned interpreter and lock/thread/memory policy:
`python calibrate.py`; `python run_panel.py --T 4 --independent` (repeat T5,6);
`python analyse.py --independent`. Original failure uses the same commands
without `--independent`. Existing files resume completed rules, so use a clean
owned artifact directory for a genuinely cold replay. Protocol/calibration
commits882862b/09dfdf7 precede outcomes; rescue protocole629db8 precedes reruns.

## Compute accounting
Charge checkpoint loads, table construction, calibration, nodes/face mapping,
all walker counters, failed controls, reductions and bootstrap. Historical truth
generation is reused evidence with original cost separate. Actual measured CPU
and composed per-call cold operation proxies are not interchangeable.
Measured total CPU68.949012327 seconds includes calibration/controls2.092491533,
original production33.271591812, clean production33.448898910, and both
analysis passes. Max RSS265060KiB. Both panels together execute1917376 production
faces,17896062 gate cells,44109389 suffix-step vectors and79253752 CDF calls.
Their recorded walker arithmetic categories total78,559,256,192 proxy operations
(including declared CDF allowances); setup/node/pilot proxies remain separate.
The control and15 cost-pilot work is retained separately in calibration.json;
every reported estimator cost charges its own pilot/setup, even though repeated
identical diagnostic calibration work was reused across rules. No new truth
sampling; inherited T2 truth generation is not erased from the evidence cost.

## Claims
[inferred] Fixed face count does not ensure equal compute. [speculative] A
threshold-free within-seed decline persists at nominally matched budgets.
[measured] Under the explicit cold proxies the clean fixed panel retains a
large64k T4->T5 decline. [measured] Budget strongly affects its magnitude.
[inferred] Fixed threshold-crossing language obscures these dependencies.
No mechanistic solution, all-seed/noisy rung pass, portable-FLOP guarantee or
family-wide impossibility follows.

## Alternatives not ruled out
Other allocations, boundary partitions, deterministic mechanism recipes,
certified geometry, better model-free tables and larger replicate counts.
This finite corrected-quadrature audit cannot exclude them.

## Failure modes
Five seeds and16 rules give broad uncertainty; proxies are not certified costs;
calibrated counts can miss target; inherited face tables and finite probes are
numerical; reused truths may not resolve MSE. Constant-readout outer-rule bias
from T20 is relevant even with an exact inner walker.
