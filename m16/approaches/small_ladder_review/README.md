# T52: independent review of the small-model ladder

## Question
Do the small-model candidates calculate accuracy correctly, including positive-mass output ties, and beat sampling under a defensible compute comparison?

## Method
Root owns the review while six workers research their assigned cases. The earlier unpushed reviewer files were absent after the workspace restart; this is a new review record, not a claim that those artifacts survived.

Freeze the next checks before execution:
1. Compare H2,T3 trained and independently perturbed models against the octant-first enumerator in `facet_transport/probe.py`. The candidate uses the independently developed two-input tail. Check all-zero and duplicate-row readouts separately.
2. Audit the operation-count convention and compare runtime against a vectorized sampling baseline separately. Kernel-call counts alone are not forward-equivalent compute.
3. Review longer-case geometry and integration bounds before accepting their nominal estimates. A convergence check alone is not a proof of floating-point geometry correctness.

### Alternatives and cheapest falsifiers
- A seemingly exact estimate omits a positive-mass tie region: use all-zero and duplicate-row output models with independently known first-index predictions.
- Both implementations share a geometry bug: use a closed-form nonrecurrent H1 family as an independent calibration and inspect the clipping conventions.
- Apparent compute advantage comes from omitted preprocessing: inventory geometry, numerical integration, and per-model setup; report wall time separately from arithmetic cost.

## Status
Active. No new candidate independently accepted yet. Claude remains responsible for project-level confirmation.

## Results
Pending recovery-cycle checks. Existing T55 conditional-primitive reviews are preserved in its own folder and do not establish a complete estimator.

## Compute accounting
Numerical checks acquire `/workspace/work/codex-compute.lock`, use one thread, bounded chunks, and the repository memory limit. Validation sampling is reported separately from estimator cost.

## Claims
- [measured] The recovered repository contains the committed T55 primitive and its bounded prior reviews.
- [speculative] The T3 estimator may provide another complete small-model benchmark; current validation is pending.

## Alternatives not ruled out
Numerical clipping error, correlated validation implementations, unsupported operation proxies, and model-specific shortcuts remain possible.

## Failure modes
Thin regions, nearly duplicate output rows, high-degree moment cancellation, and missing compute charges can each invalidate a result.

## Recovery-cycle T3 independent result

`check_t3.py` compares the two-tail calculation with octant-first three-step
clipping on12 cases: public seed0, eight new relative-RMS perturbations
(sigma.0025/.02, independent RNG52002), zero readout, identical nonzero rows,
and duplicated first two rows. Maximum difference is1.67e-16. Both constant
readouts independently equal1/3. The check uses0.281CPU seconds,24 analytic tails
and12 independent cone enumerations, with zero sampled model inputs.
` t3_review.json` preserves all results. This checks a separate perturbation
ensemble from the author's absolute-sigma.05 copies.

The candidate's T3 polygon-vertex bound2H+7 follows from an initial triangle,
2H gate cuts, two readout cuts and two rank cuts. The proposed per-kernel scalar
allowances appear conservative for H2 under unit-cost elementary/special
operations, but are not an instruction-level FLOP certificate. The author's
full-CPU comparison separately includes Python overhead; its benchmark inputs
are not used to fit or evaluate accuracy. CPU and arithmetic-proxy costs must
remain separate. Budget is advisory and may be exceeded; compare returned cost.

This supports nominal numerical correctness on these cases. It does not prove
all weights safe under floating clipping, nor establish shared-leaderboard
acceptance. The DRI decides acceptance of this small-case benchmark.

## Next frozen check: independent T4 geometry

Use Claude's `exact_ladder/cones.py` double-description gate enumeration and
pulling triangulation, rather than the candidates' Qhull-based clipping. Split
these independently constructed cones into coordinate orthants and reuse the
positive R4 volume primitive. First test only the original H2T4 and H4T4 models,
with a45CPU-second total cap. Compare nominal probabilities and quadrature-only
bounds; verify full gate determinant sum16. Shared volume arithmetic means this
checks geometry independence, not two independent volume formulas. Any failure
in the alternate enumerator is reported rather than silently patched or taken
as disproof of the candidate.

## Independent T4 result

`check_t4_dd.py` independently reconstructs the gate and correct-output cones
using DRI double-description geometry (source SHA256 recorded in
`t4_dd_review.json`). H2T4 gives0.30773074302315223 and H4T4 gives
0.47701312599404416, agreeing with the authors within7.75e-13 and7.5e-14.
Gate determinant sums are16 within7.1e-15. GL16 integration-only bounds are
1.17e-6 and1.77e-6. The check took1.46CPU seconds, with no sampled model inputs.
The DD counters248958 and4787862 exclude the review volume calculation and
are not end-to-end costs. This validates independent geometry on the two
original models; shared volume arithmetic and floating geometry remain explicit
limitations. DRI acceptance and portable cost accounting remain pending.

Reproduce: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python m16/approaches/small_ladder_review/check_t4_dd.py`.
