# T460: independent T437 scope and rescue review

## Question

Does T437 justify abandoning halfspace charging, and is another run worthwhile? This is the bounded review initially claimed as T11599, renamed T460 by the acting DRI.

## Method

Read-only source inspection and independent arithmetic, without importing producer modules, loading weights/checkpoints, reading sealed truths, generating inputs, or running models. Producer pin: **0d29f3446c6dd8fa6e3f27983b01cd48a6b5ebce**, tip of `origin/claude/t437-halfspace-charge` fetched for this review. Inspected its `m16/approaches/halfspace_charge/{README.md,hcharge.py,t414_ibab.py,qfrac.py,diag_q.py,validate.py,cell_probe.py}`. Reproduce source inspection with `git show 0d29f3446c6dd8fa6e3f27983b01cd48a6b5ebce:m16/approaches/halfspace_charge/hcharge.py` (substitute each filename).

The frozen tree contains source and README summaries but **no raw run JSON, diagnostic logs, or checkpoints**; README says these live in ignored `data/t437/`. Therefore this is not an independent replay or verification of recorded measurements. T418 comparisons below are reported by the pinned README, not independently rerun here.

## Status

Done, bounded review. Recommend retaining **negative (unverified), scoped to the tested recipes**, with the pre-registered KILL decision intact. No confirmed/dead promotion requested. Rule 5's independent test of the strongest untried variant remains outstanding.

## Results

Reproduce all arithmetic with `python review/halfspace_charge_review/check_saved_arithmetic.py` from the repository root; inputs are explicitly rounded published numbers.

- [measured arithmetic] D1/D2 extrapolations from `N*(1e-10/charge)**(1/-.46)` give **1.073e11 / 6.436e10 boxes**, matching reported 1.1e11 / 6.5e10 and exceeding the 1e9 kill line. These are conditional extrapolations, not complexity lower bounds.
- [measured arithmetic] Primary endpoint slope from 450,000 to 3.74 million boxes is **+0.503**, consistent with the reported fitted +0.49. Charge remains above 1e-9 and rises; the saved summary supports KILL. The complete last-decade fit cannot be reconstructed.
- [measured arithmetic] Root charge/slab mass is **473.43**: the pre-registered root-tightness validation failed. D1/D2 untilted/tilted gains are **1.01035 / 1.00161**. This supports little gain from the implemented tilt on these frontiers.
- [measured arithmetic] Estimated retained fractions .998/.82/.94 imply gains **1.0020/1.2195/1.0638**, respectively. They are not certified upper limits on possible gain. D2's reported 1.3x is a changed-search comparison with D1 at equal evaluated boxes, not its remaining exact-halfspace gain.
- [measured arithmetic] Domain-tail formula gives **6.14984e-12**. D2's claimed superset bound is **22.15%** of the slab; using that upper bound, conditional probability **<=0.00806** would suffice for 1e-10 (not 0.01 exactly).

## Mass assumptions and failure modes

[inferred, algebra checked] With valid lower affine margins and nonnegative multipliers, the exponential indicator inequality is sound. Completing the square gives the source's shifted Gaussian differences; integrating eight inner coordinates leaves the stated `g/(2*pi)` factor. Rectangle-wise minimum of two upper bounds and the untilted radial tail remain valid in exact arithmetic. FBBT's coordinate inequalities also preserve the feasible intersection in exact arithmetic.

These observations do **not** certify floating execution. `margins_lin` lowers `b` heuristically but returns the original `lb`; `run` prunes using **unadjusted `lb > 0`**, including after FBBT. Thus halfspace slack does not protect pruning. FBBT uses heuristic outward padding; interval endpoints, norm minimization and elementary operations are floats. The final 1e-10 inflation and A1 libm assumption are not a supplied end-to-end rounding proof. No violation is established here, but “certified superset” needs this qualification at the point of use. If E is contained in S, an upper bound on m(S) is also an upper bound on P(E); the producer's “not a bound on P(E)” wording is mathematically misleading.

`qfrac.py` uses sampled boxes and self-normalized weighted fractions with no saved confidence bounds; `nan_to_num` can substitute zero for undefined fractions. `diag_q.py` weights its top-box fractions by **upper charges CU**, not true box masses. `cell_probe.py` counts cuts at sampled points, not by maximization over entire cells. Consequently “at most 1.002x,” “bounded above,” and “0 cells cut” must be read as empirical diagnostics. Sampled validation cannot prove soundness; source also allows a 0.1% tolerance in its near-exact-mass comparison.

## Alternatives not ruled out / exhaustion

Tried: plain CROWN with margin splits and tilt; widest splits; widest plus three-pass FBBT; finite sampled k=3/4 cell probes; finer root integration grid. Not tried: alpha-CROWN halfspaces on T418's widest/mass-ordered frontier; alternate search order; exact halfspace integration; (9,8,2); complete outward-rounded pruning and mass arithmetic. Exact integration on the existing frontier is lower priority given small estimated retained-mass gains, but is not ruled out.

Two competing explanations remain: stronger bounds could move substantial mass near cutting planes, or poor integration/search ordering could hide useful cuts. Cheapest distinguishing follow-up: on a **saved alpha-CROWN frontier**, inspect mass-dominant boxes and compare their affine ranges and tighter box charges; avoid rerunning T437's failed search. Current review cannot perform this because the frontier artifacts are absent.

## Recommendation and compute accounting

Prioritize the existing **T455** lane: rigorous mass with T418's stronger pruning and widest/mass order. Request its saved frontier and provenance before reserving a small deterministic alpha-halfspace audit; any numerical work requires its own task/claim. Do not infer the plain-CROWN constant-factor ceiling transfers to alpha-CROWN.

This review used zero model passes, LP solves, or numerical experiments; only source reads and elementary saved arithmetic. Extending unchanged D1/D2 to their projected target would take roughly **49,900 / 35,600 CPU-hours** at observed throughput, with unvalidated scaling. **No such compute is justified.** Most promising rescue is an integration with T455, not more duration on T437; cost cannot be estimated credibly before obtaining the saved frontier. Strict F<=1 and lenient F<=10 are **not evaluated**: this is a bound-search review, not an estimator with independently measured error/cost.
