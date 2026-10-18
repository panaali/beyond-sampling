# T262 — Independent Astra/xhigh review

## Question

What actually obstructs a deductive accuracy estimate for M_16,10, and which inexpensive tests could change that diagnosis?

## Method

Source review and first-principles reasoning only, by **gpt-6-astra, xhigh**. The requested README was absent when checked; T262 specified it as this output. I did **not** read Claude Fable's parallel review/scratchpad.

Source snapshot: shared `61b30b17` (claim `f0bf6b77ffaa103c75fbdefe4c65cd53670307ea`; report update `5240d464`). Read AGENTS, TASKS, INDEX's requested ledger/experiment entries, DRI_HANDBOOK §§4/7/8/10, FINDINGS and handover QA. Branch-only sources below were read with `git show`; none of these five heads is an ancestor of that shared snapshot. Their reported measurements are **not newly verified here**.

| Reference | Exact source head; files under `m16/approaches/` |
|---|---|
| G | `f8bbb07b3f6ac56813d728c48ef2f9ae77eee8d9` — `gpu_cost_law/README.md`, `prefix.py`, retained k6 JSON |
| W | `7e066e51e996342471b34049064856e9af1ffcbf` — `width16_probe/README.md`, `random_control/README.md`, `shatter_test/README.md`, `tie_blind/README.md` |
| A | `daff5713ddebad320dabbd3e2ec42db72e3539cb` — `split_attribution/README.md`, pruning report |
| L | `182f16c0011a0b8d0e35962ee96ecaa36c450a6f` — `tie_tail_labels/README.md` |
| N | `e3d0a0de79c1dd37b7574889c3157fd3b0f44c74` — `neuron_account/README.md` |

Shared review precedents include T219/T227/T228/T231/T233/T243/T254/T255 and the current T230 protocol. These precedents are credited, not independently rerun.

## Status

Done: independent review, eight proposals, six new backlog rows T263–T268. Existing lanes retain T230/T233/T243/T256/T258/T259/T260. Scientific acceptance remains Claude's.

## Results

### A. Critical review, ranked by impact

1. **The scaling headlines outrun the evidence.** [inferred] “Per-region methods cannot scale,” “constant-factor savings,” and “noise does not change this” in [SCALING_OBSTRUCTIONS](../../m16/SCALING_OBSTRUCTIONS.md) should be conditional, implementation-specific judgments. Independent exact history counts reach k=4; the documented lower bound is 1,346,048 histories at T=10, not 10¹³ minimal affine maps or operations. T222's target coverage growth decreases 18.6→16.1→14.9; an expensive continuation is plausible, not proved. T232 compares one trained network against selected random ensembles at k≤4; it neither isolates training causally nor establishes an asymptotic law. Noise costs were measured only at T=4.

   G's k6 “unbiased” estimate is unsupported: shuffled dispatch, unordered completion and a wall deadline select completed subtrees by runtime. [T255 already establishes this](../../m16/approaches/feynman_fundamentals/t192_latest_evidence/HANDOFF.md); its correction has not reached the report. The 22.9M estimate is uncalibrated; completed histories remain numerical lower-bound evidence. No bias direction or magnitude is established.

2. **The surrogate question is being confused with the estimation question.** [inferred] Let I and J denote network and surrogate correctness. Then
   `p_network − p_surrogate = P(I=1,J=0) − P(I=0,J=1)`.
   Behavioral disagreement bounds this difference but need not approximate it tightly. Conversely, matching accuracies can hide wrong mechanisms. N's +0.11 versus +0.82 pp accuracy differences demonstrate unstable cancellation, while .955–.966 agreement consistently misses its own target. Neither proves that accuracy estimation requires 99% agreement. The report's “99% needs 1−R²≈10⁻⁵” applies to **independent Gaussian neuron perturbations**, not all structured formula errors. Recurrent injection through step 7 is promising evidence about suffix sensitivity, not proof that the mechanism is understood everywhere else.

3. **The reported oracle saving is not an upper bound.** [inferred] W/T206 sorts leaves by error and uses a conservative fixed sampling threshold. With unequal integration costs this is not an optimal cost/error allocation; reducing cost also increases the permitted sampling error. Counterexample: errors .1/.2, costs 1/100, bias allowance .2. Error-first skipping saves 1; skipping the second saves 100. Thus “at most 2–4×” is unsupported even within the oracle setup. Recompute its self-consistent frontier before dismissing precision allocation. Also distinguish T196 nominal FLOPs, implementation completeness, and wall time: G's 1.37 multiplier is a nominal audit, while geometry dominates wall time despite masses dominating counted operations. Setup, routing and fitting cannot become free advice.

4. **Several negative experiments test the wrong sufficient statistic for a universal conclusion.** [inferred] T217's modal *gate word* need not predict the conditional accuracy integral. T218 rejects one fine-label rule; its tail-conditioned within-label analysis does not reject coarse-family regularity (T227/T230). T195's fixed-piece intervention is not joint split removal; its pruning bound excludes recipe bias, so `B²/SD²<1` is not a complete accuracy certificate. T239 assigns whole-leaf mass by its center: that is not mass actually inside a gap stratum (T254). Fragmentation near ties neither proves randomness nor prevents a compact integrated law.

5. **Mechanistic prose still contains concrete errors and overstatements.** [inferred] “All gates start off” on all-negative inputs is false: `h1[j]=ReLU(Wi[j] x0)` activates negative-Wi neurons; L explicitly notes this. The blind probability 11/1024 is not its error probability. FINDINGS' canonical-spacing success excludes crucial magnitude/sign configurations and cannot establish algorithmic sufficiency. T241's large cell tables are not compact formulas until T260 counts their construction and description. Conditional donor resampling preserves selected strata, not the full hidden-state joint law; its damage establishes sensitivity more readily than a unique causal role. T228 also rejects T216's claimed gauge invariance of row cosines.

6. **Evaluation hygiene remains load-bearing.** [inferred] No direct train/eval overlap is established for N; its independent review reports disjoint sets. Nevertheless, repeated development on public target truths creates adaptive selection risk. Freeze the estimator, budgets, cost convention and noise panel before final scoring; retain every model/budget, independent truth streams and uncertainty. Old nine-model rung wins do not certify the current ≥8-copies-per-seed, three-noise-setting protocol. Scramble-averaged quadrature gains on the clean target remain quadrature evidence, not mechanistic target success. Near-unity ratios need more caution than the large failures.

### B. First principles

[inferred] Positive homogeneity removes radius: accuracy is a Gaussian/angular measure of **correctness**, not a census of hidden gates. The estimator must exploit relations between weights, task order statistics and the input law. For clean target p≈.953, sampling at C T196 FLOPs has SD approximately `16.3/sqrt(C)`. A deterministic point estimator needs sufficiently small **integrated signed error**; a uniform certificate is sufficient, not an ARC requirement.

The useful state may therefore be much smaller than an exact gate history: competing logit differences, top-order values/positions, record ages, and the **joint conditional law** of the residual contributions. Readout common shifts do not matter. Replacing omitted inputs by their means discards threshold-crossing probability; independently Gaussianizing correlated neurons discards the shared scalar innovation. Carry distributions or controlled signed corrections instead.

What remains insufficiently tested is weak equivalence—different histories with nearly the same *future success probability*—and integration regularity after removing known discontinuities. Exact all-futures equivalence is much stronger. Heavy tails under one face parametrization are not an invariant obstruction to every coordinate system. Conversely, small parameter count, damped eigenvalues in one gate set, or a short verbal explanation do not guarantee a cheap integral. Bulk-error noisy copies require a fallback beyond clean-model tie asymptotics.

### C. Eight concrete ideas

All are **[speculative]**. Each is a proposal, not an executed result. Development diagnostics may use reference data; an ARC candidate must construct its estimator from weights without random model inputs or fitted per-model advice.

1. **Integrate the residual band, retaining its law** (T233/T243/T256). Cheapest test: on one exact H4T4 reference, fix eight top-order/position strata and compare zero/first/second-order signed boundary corrections for both 1–2 and 2–3 ties. Fail if total `cost × squared error` does not halve or an explicit remainder overwhelms sampling SD. Avoids mean-input/Jensen failures. Risk: multiple crossings and residual–gap dependence defeat a local expansion; rank conditioning changes the Gaussian law.

2. **Test full coarse-family regularity** (T230/T235). Recover or capture one bounded full-face panel, including zeros and capped walks; freeze cheap pre-walk labels on development data. Test held-out residual second moments and within-family quadrature refinement, charging label and conditional-law construction. Fail if held-out residual variance times total cost is not below the ungrouped baseline, or deterministic refinement fails to improve. Avoids fine-label singleton and global-tail conditioning artifacts. Risk: useful labels are outcome-derived or have intractable probabilities; low variance alone does not prove deterministic convergence.

3. **Backward suffix-success functions** (T265). At H4T4, integrate the last one/two inputs exactly and compare prefix cells using a fixed small weight-derived summary of `q_t(h, rank-state)=P(future correct | state)`. Fail if a ≤sampling-SD error needs more than half the exact states or costs as much as the exact suffix calculation. Avoids demanding identical futures for every suffix, as exact bisimulation does. Risk: rank/hidden-state coupling carries the complexity into q.

4. **Optimize the decision-relevant objective** (T259, dependent on T258/T260). Under the existing frozen fit budget, compare activation MSE with competitor-margin/sign loss using the same feature family and an untouched evaluation set; retain both signed correctness transitions. Fail if the frozen .99 agreement target fails on the untouched set or a fresh noisy copy; separately report signed accuracy bias. Avoids high-R² but wrong-boundary models. Risk: this is a sampled diagnostic, not a deductive estimator; richer fitting may merely memorize residuals.

5. **Find the cost/error frontier before further precision work** (T264/T212). On one retained T206 leaf table, compute feasible selections and an optimistic relaxation with fixed geometry charged, unequal leaf costs and the sampling threshold recalculated at remaining cost. Fail if even the optimistic frontier cannot halve total cost at bias below its recomputed sampling SD. Avoids a false oracle ceiling and uniformly overprecise masses. Risk: discovering error bounds can consume the apparent gain; an oracle cannot be deployed.

6. **Pair signed residuals under a measure-preserving input involution** (T266). Freeze time reversal, transform the true label consistently, and evaluate paired residuals on an exact H4T3→H4T4 control. Fail if reduced absolute remainder cannot pay for both evaluations. Avoids requiring a neuron-weight symmetry: Gaussian measure preservation alone justifies pairing. Risk: residuals reinforce; finding a favorable pairing on test data leaks selection.

7. **Cancel internal classification facets before mass integration** (T267). On existing small exact complexes, cancel oppositely oriented facets within the same correctness event; compare remaining boundary work across T3→T4. Fail if remaining facets cannot halve projected total work after discovery/matching costs, or their T-growth matches cells. Avoids paying separately for internal ReLU boundaries irrelevant to correctness. Risk: shared-facet degeneracies and discovering the boundary cost as much as cells; H-U3 already warns that tracing alone can fail.

8. **A decision quotient inside blind sign sectors** (T268). At H4T4 split all-negative and single-positive-position sectors, preserve their active negative-Wi paths, and seek equal conditional predicted-class laws rather than equal gate words. Fail if sector construction plus integration gives no tighter cost-matched error bound. Avoids pretending known blind mass gives known blind error. Risk: negative inputs still carry a complicated rank-dependent state; success on this sector would only solve one contribution.

### D. Diagnostics: where to look

[inferred] Retain these eight quantities per model, not only pooled:

- Signed accuracy bias versus total prediction disagreement, stratified by both top gaps and blind status.
- Conditional residual tails **and cross-covariances**, alongside readout-margin sensitivity.
- Full-family refinement error versus label-construction cost; no conditioning on an already selected tail.
- Accuracy-relevant boundary count/cost versus gate-history count.
- Weak suffix-function approximation error versus required state count.
- Joint classification-bias, integration, truncation and rounding budget versus sampling SD.
- End-to-end cost/error frontier, including setup and an efficient matched sampler.
- Transfer across trained seeds and noise, with positive diagonal neuron-rescaling and permutation invariance controls.

### E. Top three next experiments and odds

[speculative] **First:** C5, the retained-data cost/error frontier: cheapest way to test whether a major exclusion was justified. **Second:** C2, one proper full-family regularity test, using the existing T230/T235 design. **Third:** C1, a single small-rung residual-law application connecting the proved T233/T243 primitives to actual margin error. Give each one a fixed falsifiable cutoff; a failed variant stays recorded, with its untested alternatives open. Avoid another broad sweep or more source audits without a concrete decision they unlock.

**Subjective probabilities, not measured frequencies:** about **60%** that some practical, frozen weight-derived algorithm can eventually beat sampling on the required M_16,10 family; about **20%** that this project demonstrates it within three months at roughly current resources; **under 5%** for straightforward continuation of today's per-region/cone-mass pipeline. These are deliberately rough judgments, conditional on the stated ARC interpretation and usable compute budgets—not a prediction of an unpublished ARC result. A cheap conditional-residual law transferring to noisy copies would raise my odds substantially; state complexity tracking regions on the proposed discriminators would lower them. The full challenge remains unsolved.

## Compute accounting

Zero model forwards, training, fitting, MC, RNG, LP, geometry, integration or binary-array replay. Only text/JSON source inspection, Git metadata, writing and document checks. Empirical numbers above are attributed reports; no reproduction command was executed. This review has no estimator FLOP claim.

## Claims

`[inferred]` marks source/algebra-based conclusions, `[speculative]` proposals and subjective odds. Reported measurements retain their source's verification scope; this review confirms no empirical hypothesis.

## Alternatives not ruled out

Weak distributional compression, improved boundary coordinates, cheaper mass algorithms and hybrid certified/approximate integration remain open. The actual k6 selection bias may be small. Alternative parameterizations or noise regimes may change every empirical scaling trend.

## Failure modes

Source reports can be stale or incomplete; no runtime reproduction occurred. Small-rung diagnostics may transfer poorly. Cancellation may be unstable across models. A useful sampled explanation or quadrature improvement may still violate ARC's mechanistic restriction.
