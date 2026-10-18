# T285 — independent Astra/xhigh review, round 2

## Question

Can the revised own-cost/retrain challenge admit a cheap deductive M16,T10 estimator, and what should change after T262?

## Method

Actual reviewer: **gpt-6-astra, reasoning xhigh**, the existing T262 reviewer. Read/think/write only; no experimental execution, model imports, arrays, fitting, RNG or sealed truths. Neither Fable round-2 review nor its scratchpad was read. Assignment was already published in TASKS-only `a4150230`.

Exact reading snapshots (paths below are relative to these commits):

| Key | Commit | Material read |
|---|---|---|
| S | `d3f71e6847839eb25abd34de1ac5a85aab77a8ec` | AGENTS, TASKS, INDEX, DRI_HANDBOOK; DRI_HANDOFF; SCALING_OBSTRUCTIONS; own T262 review; `lit/web_arc/hilton_talk_2026-09_notes.md`; `review/retained_rate_audit/{SOURCE_FINDINGS,RESULTS}.md`; `review/residual_band_application/HANDOFF.md`; `review/blind_sector_accuracy/README.md` |
| S2 | `339cddc1bb83686cecf7112a551e728efc432f54` | Final `review/{retained_rate_audit,blind_sector_accuracy}/HANDOFF.md`; current TASKS (frozen-witness task renumbered T288→T287) |
| G | `d32895b6ef1b1bb8746d71a275f7a951222ad5d1` | `m16/approaches/capacity_ceiling/README.md` |
| W | `d8130f4d3e16a37d4994b877349850b13b58c4d9` | `m16/approaches/{rate_T,frozen_check}/README.md` |
| N | `44a3524a3ab188cc10c725c5aaf8096c7febb493` | `m16/approaches/{probe_v5,crown_tree}/README.md` |
| L | `26e9291d49505b116860aed48945deb45d57ea34` | `m16/approaches/lenient_bb/README.md` |
| H | `27788ff7d1478e789b1db9d73f3f9ad298fbbb78` | `m16/approaches/hp1_scorer/README.md` |

Reproduce documentary evidence with `git show <commit>:<path>`. Reported experimental numbers are citations, not my numerical confirmations. HP2/DEV2 construction is taken from S's directive; the advertised HP2 README was absent at G. The ARC source is an owner-supplied approximate transcript, not a formal scoring specification.

## Status

Complete independent written review. Full challenge remains unsolved. Proposed experiments below are **not executed or authorized by this review**.

## Results

### A. Audit, ranked by decision impact

1. **Own-cost scoring is a useful correction; distinguish conventions from ARC requirements.** [inferred] For fixed cost C and model distribution, write `V=E[p_m(1-p_m)]`, `R=E[(estimate_m-p_m)^2]`. Uniform-budget sampling has MSE V/C; the project's score is `F=R max(C,1000)/V`. This correctly exposes useful cheap points hidden by the former 16k minimum. Neither the 1k floor, 5948-FLOP FPE, particular retraining kernel nor lenient threshold 10 comes from the talk. Cheap constant estimates passing an unmodified comparator are mathematically possible; they still supply no model-specific explanation. Keep a constant development-prior control, without calling it a mechanistic solution.

   For variable C_m, H's `R max(E[C_m],1000)/V` compares against sampling with a common average budget. The earlier comparator `R/E[V_m/C_m]` matches each model's budget. These are different legitimate questions, especially when hard models cost more. Freeze the chosen comparator and floor placement; report both when costs vary. Use E[p(1−p)], not E[p](1−E[p]). Truth noise and amortization of model-independent tables require an explicit ledger.

2. **HP2 changes the population; it does not retroactively repair HP1.** [inferred] Fresh retrains are well motivated by the talk. A pass there establishes performance for that training distribution, while HP1 remains a robustness test. G's 40-versus-40 nonsignificant KS comparison does not establish distributional equivalence. Same-initialization correlation .11 does not prove initialization causally irrelevant: changed data and numerical trajectories can swamp its contribution. A target near the observed maximum does not establish an architectural ceiling or prove identical provenance. Freeze trainer, schedule, seed law and endpoint. Keep the two DEV2 kernels paired by initialization when splitting development/validation; never tune on HP2 feedback.

3. **“Closed” methods must be rescoped under the cheaper objective.** [inferred] H's own-cost table gives v5a lenient passage at its cheapest setting, despite older “fails everywhere” prose. It remains inductive. W's approximately 1 pp frozen-classifier bias excludes particular high-budget claims, not every cheap refinement. Neither α>1/2 nor improvement at every budget is necessary for one fixed-cost win. T286/T287 already own the appropriate reconsideration. Calling center-based branch_bound a bookkeeping artefact obscures its actual defect: uncertified extrapolation of a witness to its entire cone.

4. **The new rate evidence is narrower than its headline.** [inferred] S2/T283 independently reproduces 1,009 retained rows and 1,186 required fit fields. T277 rejects its finite dimension criterion through rank correlation .02857. Its fitted α=.667 also falsifies “all within .5±.15”; two T271 intervals exclude .5. Four T3/T4 prefactors match the exact-reference denominator rather than fit.py's MC denominator; this compatibility does not recover the absent generator's history. Reused scramble labels across budgets and adaptive allocations limit ordinary regression-CI interpretation. These facts establish neither equal asymptotic rates nor prefactor-only causation. A falling prefactor can explain the observed loss without making deductive estimation impossible.

5. **Retain the experimental confounds.** [inferred] T278's outer integrated mass exceeds one (1.003667 at GL5); cancellation with outer error prevents its failed correction from killing residual-law integration. S2/T268 tests binary correctness sectors, not constancy of predicted class: changing true labels can change correctness under a fixed prediction. H's “every control CI contains 1” also conflicts with its 16k absolute-noise interval above 1. One such interval need not invalidate a calibrated harness, but the prose is false. These are bounded corrections, not requests to restart every experiment.

6. **Freeze the actual decision rule.** [inferred] H now uses point F for strict/lenient screening, while older board text uses upper confidence limits. Report the version, point estimate and interval together; require an independently specified HP2 confidence criterion before a success claim. Passing by selecting among many development budgets is not held-out evidence. Six HP1 base clusters give limited precision. Fixed nodes do not become deductive merely because they are deterministic; constructed witnesses need a weight-derived extrapolation argument. A proved integration formula or error-controlled quadrature could supply that argument; naked node averaging does not.

### B. Strongest case on each side

[inferred] At p≈.92, strict RMS allowances are about **.85/.43/.21 pp at 1k/4k/16k FPE**. With between-model SD≈3 pp, a constant prior is far too crude; reducing its variance by roughly 92%/98%/99.5% illustrates the required model-specific information. These are scale calculations, not measured predictive R² requirements.

**For:** weights completely determine accuracy; training chaos does not leave irreducible randomness after conditioning on weights. Ten Gaussian inputs, homogeneity, exchangeability, known rank laws and one shared scalar innovation per step are substantial structure. The desired quantity is one signed integral, not faithful reconstruction of all gates or predictions. Roughly six million FLOPs can perform considerable algebra on 432 weights. Cheap success need not imply a competitive asymptotic convergence rate.

**Against:** actual lower-rank substitutions leave biases comparable to or larger than the cheap allowance. Scalar innovations create strong joint dependence, so independent-neuron closures fail. Generic region counts remove one hoped-for simplification; rank conditioning couples the remaining inputs; repeated thresholding can destroy smoothness. At 16k+, the error tolerance tightens faster than an uncontrolled approximation's bias improves. Spending more does not cure a floor.

There is **no established impossibility theorem for this trained distribution at either budget**. Worst-case volume/CAPP hardness, one finite convergence fit and large region counts do not provide one. Conversely, small parameter count and a verbal algorithm do not guarantee an inexpensive accuracy integral.

### C. Three new deductive designs

All **[speculative]**, distinct from T286's existing-method sweep and T287's frozen witnesses. Costs below are hard proposed caps, not demonstrated implementations. Any unresolved probability remains an interval; no empirical frequency is substituted. Model-free integration tables must themselves be analytically derived, with construction and amortization disclosed.

**C1 / T289 — Joint readout-error bounds.** Partition by the true maximum/runner-up positions (90 cones; split their lineality for bounded projective relaxations). Reuse T279's relaxation infrastructure, with its owner: share cheap forward affine bounds across rivals instead of full repeated back-substitution. Derive inner/outer cones for rival-margin events, retaining shared-input constraints. Integrate these Gaussian cones with certified deterministic bounds; combine rival events using first/second-order Bonferroni bounds, adding intersections in a fixed weight-derived priority order. Omitted intersections widen the interval. Return its midpoint.

The new step integrates error-event overlaps; T279 instead certifies whole cones. It also avoids v5b's independent-rival product and FORM/SORM's nearest-boundary approximation. N/T279 quotes 2–4 million FLOPs per cone: that implementation already fails the cheap cap. Require shared-bound construction within **1k FPE**, and **4k total** including intersections and masses; otherwise stop before integration. Risk: cheaper bounds become vacuous. Cheapest test: four predeclared DEV2 models, two paired initializations across both kernels. Stop this configuration if it cannot construct valid intervals within the cap, or their half-widths exceed the sampling allowance. Full-model truth is unnecessary for this first discriminator.

**C2 / T290 — Algebraically coarsened shadow network plus a coupled discrepancy certificate.** Remove common readout shifts, balance neuron scales deterministically, then round weights at a fixed sequence of relative precisions and merge only algebraically identical units. Compute the smaller shadow's accuracy with existing exact machinery. Propagate original/shadow states jointly, retaining correlated difference bounds; charge only probability where their winning classes can differ. Estimate from the shadow with that uncertainty interval.

No activation fitting or behavior-selected rounding is allowed. Evidence: homogeneity permits function-preserving normalization; many gates need not correspond to distinct output decisions. Neither implies successful compression. Risk: tiny weight changes accumulate, or no repeated algebraic structure appears. Cheapest test: the same four DEV2 models; cap **1k FPE for construction and 4k total**. Stop if the shadow remains computationally large or discrepancy plus integration uncertainty cannot fit the strict allowance. This tests compression of arithmetic, not a fitted neuron-account surrogate.

**C3 / T291 — Shared-innovation coefficient propagation with a remainder ledger.** Represent hidden-state functions in a fixed low-degree Gaussian polynomial basis, compressed by deterministic tensor factorizations; propagate the common input variable jointly across all neurons. Replace ReLU only with analytic upper/lower approximants on a declared bounded domain; charge Gaussian tails. Integrate endpoint correctness through polynomial bounds on margin indicators within known true-rank cones, using model-free truncated Gaussian moments. Bound every coefficient truncation and threshold-approximation remainder.

This differs from TT-cross on sampled function values and independent-neuron cumulants. Evidence: recurrence and exact Gaussian moments supply algebraic operations. Risk: indicator discontinuities and correlated remainders make the bounds useless—precisely the previous harmonic warning. Fix degree/rank combinations before examining truths, cap **4k FPE**, and stop on the four-model DEV2 screen if the final interval is wider than the strict allowance or the rank cap is exceeded. Do not silently replace a failed bound with Gaussian closure.

### D. Start C1 with this preregistration

[speculative] Coordinate with T279; first verify enclosure and cost accounting on the existing exact H4T3/H4T4 rung, without tuning to target truth. Enforce C1's construction-cost gate before mass work. Then use DEV2 initialization IDs 0 and 1 from both kernels for the fixed 4k screening test above. Select no alternate formulation after seeing those truths. For an affirmative screen, require valid intervals [l_m,u_m], C_m≤4000, and

`mean[((u_m-l_m)/2)^2] × 4000 / mean[min(l_m(1-l_m),u_m(1-u_m))] ≤ 0.8`.

If bounds or denominator are vacuous, stop. This is a sufficient development certificate with headroom, not a necessary condition for ARC success. Freeze the surviving estimator and run the remaining DEV2 models with all costs; only then request independent HP2 scoring. A failed screen rejects this relaxation configuration, not all joint-event estimators. No experimental execution is part of T285.

### E. Odds and top three

**Subjective judgments, not frequencies:** within three months at current resources, **30%** for a strict deductive HP2 win somewhere at 1k–4k, **10%** at a separately frozen ≥16k point; **60%** eventually for the retrain-family challenge. Requiring the same estimator to win every HP1 noise group lowers the cheap three-month estimate to **15%**. More favorable cheap odds than T262 reflect the changed objective, not a new demonstrated mechanism.

Priority: **(1)** finish T286's fully charged cheap frontier; **(2)** C1's bounded enclosure screen; **(3)** finish T287's class-refinement curve and validate it independently. Keep T278's outer-error discriminator available if these fail; do not spend on a full new propagation engine before its cheap bounds look useful.

## Compute accounting

Zero model, training, MC, fit, geometry, mass-integration or binary-array computation. Only text/Git/document operations. Proposed candidates charge input-law construction, algebra, optimization, integration, certificates and routing in native FLOPs; 1 FPE=5948 FLOPs. A cost cap is not a forecast.

## Claims

All critical conclusions above are **[inferred]** from explicitly cited published reports; no new **[measured]** scientific result. C1–C3, prioritization and odds are **[speculative]**. No claim is self-confirmed.

## Alternatives not ruled out

Cheap failure may be poor coordinates rather than intrinsic complexity: C1 tests a different representation. Apparent HP2 improvement may instead be population change: retain HP1 and kernel-stratified DEV2 reporting. Strong signed cancellation can beat my sufficient intervals; failure of these certificates cannot rule out such estimators.

## Failure modes

Loose bounds, hidden table costs, unproved closure assumptions, adaptive panel reuse, inconsistent cost comparators and treating finite negatives as impossibility would invalidate the proposed conclusions. No Fable-r2 or sealed truth access is needed to test them.
