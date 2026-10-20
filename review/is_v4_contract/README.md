# T462: IS-v4 cost and reliability contract controls

## Question
Does the defensive implementation pass independent analytic controls, and does its finite weight cap make the normal-SE interval or observed-weight gate reliable when a proposal misses a mode?

## Method
The [protocol](PROTOCOL.md) was published in `376ffd4b` before execution. Root approved the frozen run. [run.py](run.py) extracts the literal sampler, mixture, SE and gate code by AST from T456 `a52da0d6a57d29aa2f76daa74c3360222f33aec2`; references use independent scipy distributions and exact multinomial category probabilities. It never imports a trained model. [results.json](results.json) records source hashes, all controls and resources; [replications.csv.gz](replications.csv.gz) retains every replicate. The source normalisation deliberately omits the common Gaussian constant; the independent density comparison restores it.

Two analytic missing-mode fixtures use nominal Uniform[0,1], event A union B of probability 1e-6, and proposal .2 nominal + .8 Uniform(A). B carries 10% or 50% of the event probability. A positive control covers both modes. Each has 2000 independent fixed-n repetitions at n=100000, followed by the source's optional extension rule on the same initial draws. A low-hit control tests the extension branch. Exact category-count sampling has exactly the same law as IID draws for these piecewise-constant weights; it avoids generating hundreds of millions of irrelevant uniforms. These are **sampling-class synthetic controls**, not trained-network replays or leaderboard results.

## Status
Bounded phase complete; [independent mathematical cross-review](../late_bound_soundness/T462_REVIEW.md) passes the scoped construction and interpretation. Root separately checked raw summaries. Sampler and density screens pass. The generic normal-interval/gate reliability claim fails on the preregistered missing-mode controls (**negative supported by independent cross-review; root ledger decision separate**). This does not invalidate T456's recorded trained-panel observations. No claim is promoted to confirmed here.

## Results
[Measured] All three defensive-sampler screens pass at thresholds 1.25/1.75/2.25 (100000 draws each): third-order-statistic proportions against independent integration, conditional top/low PITs and largest-coordinate positions; no invalid gaps or nonfinite samples. Independent quadrature agrees with source Z within the frozen tolerance. The 100000-draw mixture test passes: maximum log-density discrepancy 1.07e-14, maximum weight .00962853 below cap .00965083, normalisation z=-.392.

| Fixed-n analytic control | Normal interval coverage | Gate rate | Unflagged noncoverage (95% Wilson) | Draw-cost F | Strict / lenient |
|---|---:|---:|---:|---:|---|
| Missing 10% mode | 0.25% | 0.25% | 99.75% [99.42,99.89]% | .6235 | pass / pass |
| Missing 50% mode | 0.90% | 0.90% | 99.10% [98.58,99.43]% | 2.2300 | fail / pass |
| Both modes covered | 95.80% | 0% | 4.20% [3.41,5.17]% | 2.502e-7 | pass / pass |
| Low-hit nominal | 92.40% | 99.40% | 0.60% [.34,1.05]% | 1.0020 | fail / pass |

[Measured] Both missing-mode fixtures fail the frozen reliability screen (Wilson lower bound on silent noncoverage >5%); the positive/low-hit controls do not. The mean reported SE squared approximately matches empirical variance even in the two failing cases (variance/mean SE squared .998/.991): agreement of averaged variance alone does not establish per-run coverage. Most runs see no B hits and report a very small SE around the incomplete mode mass; the rare B hits carry cap-sized weights and the gate fires only after observing them. The fixed-n estimators remain unbiased in theory. At this rarity, empirical mean and MSE depend strongly on only 5 and 18 observed B hits across 2000 runs; F is not a precisely calibrated efficiency result. The 50% fixture has 1004 silent >2x underestimates, because sampling fluctuation places roughly half of the no-B estimates below .5p; the threshold is deliberately near the construction's boundary.

[Measured] Source optional extension triggers on .25%/.90% of the missing-mode cases, leaving the same silent failure fractions. Mean estimate/truth is .9126/.5525 after stopping versus 1.0251/.9500 at fixed n. The descriptive MSE*mean(n)/[p(1-p)] ratios are .00716/.05783 (strict and lenient point bars both pass); those cheap, biased estimates still have poor coverage. The covered control is unchanged. The low-hit control always extends, reaches 96.5% normal coverage and F=.9454 (strict/lenient point passes). This experiment illustrates stopping selection, not the trained estimator's empirical bias.

[Measured] The separate stage1 no-B-discovery diagnostic is 99.3%/94.5%, with analytic probabilities 99.005%/95.123%. It models only defensive discovery exposure, not the source descent or CE update after a discovery. We do not imply that the source could never repair such a mode.

### Constructive reporting alternative
[Inferred, analytic guarantee] For independent fixed-n draws with a correct bounded weight W in [0,b], the Chernoff/KL confidence set in the protocol is a valid conservative 95% interval. The proof uses the Bernoulli upper bound on the bounded-variable moment generating function and a two-tail union bound. It also holds conditional on an independently fitted, then frozen proposal. This is separate from any exactness assertion for the source's grid sampler and is not automatically valid after optional stopping.

[Measured] Fixed-n KL coverage is 100% for both missing-mode cases and the covered control, 99.3% for the low-hit control. The cost is precision: median interval widths are 190.2p/187.9p/190.7p/1.753p respectively. The nominal cap alone therefore supports honest but weak reporting; a silent normal gate is not a substitute. No post-outcome tighter cap or interval was selected.

Reproduce numerical results, from the repository with the pinned history:
```
ulimit -v 2500000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python review/is_v4_contract/run.py
/workspace/clean-neurons/.venv/bin/python review/is_v4_contract/cost_inventory.py
```
The environment is Python 3 with numpy/scipy; the existing virtual environment requires no installation. The one run passed without code fixes or reruns. After execution, only the replicate file's storage was changed from CSV to gzip CSV and the runner updated accordingly; the numerical path and outcomes were unchanged.

## Compute accounting
Measured suite CPU after imports 3.354s, peak RSS 150768 KiB (147.2 MiB), one BLAS/OpenMP thread; 400000 generated 10D vectors; 8000 category-count replications; zero model passes, learned geometry solves, weight loading or new trained truths. Cost inventory took <.03s wall time. Import overhead is additional and small; no whole-process CPU counter was retained for the original run. Phase comfortably below600CPU-s and2GB. Analytic draw-cost F uses represented IID trial counts, **not actual FLOPs or FPE**; random-stopping F is only descriptive. The trained estimator's full-FLOP strict/lenient status remains unestablished by this task.

### Source cost inventory
C includes pilot network passes, timed pilot/discovery descent work converted at4microseconds/FPE, and stage network passes. C_draws uses draws instead of passes. Discovery timing includes its seed ranking/selection. Neither is a full operation inventory.

| Operation | Cost dependence | Accounting status |
|---|---|---|
| Defensive CDF grid/quadrature (200001 nodes per threshold) | Input law only | T312 permits exemption if actually precomputed once, listed and reused identically; current code rebuilds, so distinguish runtime from charged FLOPs |
| RNG, truncated-normal inversion, permutation and gap filtering | Draw/proposal processing | Draw counts alone do not instrument actual cost; model-free pieces and weight-dependent mixture choices must be distinguished |
| Gaussian component transform | Selected learned mean/covariance; O(n*d^2) | Not explicitly charged by C |
| Mixture density on hits | Learned components; O(h*K*d^2) | Not explicitly charged by C; every component evaluated on every hit |
| Nearest centre / weighted covariance / eigendecomposition | Learned hits and centres; O(h*K*d), O(h*d^2), O(K*d^3) | Not explicitly charged by C |
| Cholesky/inverses, merge/selection and bookkeeping | Learned proposal/points | Not separately instrumented; some timed descent work is already included |

[Measured saved-record counts; inferred conventional FLOPs] [cost_inventory.json](cost_inventory.json) reconstructs 127740531 stage2 hit/component pairs over the15 saved points. Direct dense 10D quadratic evaluation costs299 multiply/add FLOPs per pair (200 multiplies,99 additions). That contraction alone averages428093FPE at5948FLOPs/FPE, from2.86% to28.31% of each published C. This is a source-level operation estimate, not a hardware instruction measurement or revised panel F: it excludes stage1 densities, log/exp costs, transforms and fitting. Saved summaries are insufficient for a complete weight-dependent replay. The law-only exemptions in T312 must not be mistaken for omitted chargeable work.

## Claims
- [measured] The frozen sampler/density controls pass at the tested scale; common implementation errors large enough for these screens were not observed.
- [inferred] A defensive finite cap does not make a normal SE interval or a gate based only on observed weights uniformly reliable. The analytic missing-mode controls construct a counterexample with an exact target and exact proposal.
- [inferred] Fixed-n bounded-weight reporting can be conservative and honest but may be too wide to estimate the rare probability usefully.
- [inferred] T456 saved performance remains an observed panel result; neither a universal reliability guarantee nor a full-operation score follows.

## Alternatives not ruled out
The trained panel may have sufficient discovery and benign tails; those outcomes remain intact. This does not test how CE fitting/descent repairs missing modes, a different defensive fraction, larger budget, more modes, trained-weight perturbations, reference stopping bias, or the full network cost. The source inverse-CDF uses trapezoids on a finite grid and interpolation: a passing coarse screen is not proof of exact sampling and cannot itself inherit the mathematical KL guarantee without accounting for sampling error.

## Failure modes and exhaustion checklist
- Tried: all frozen thresholds and analytic mixture controls, fixed/optional SE rules, independent density references, and fixed-n bounded-KL reporting. No cases skipped, no failed run discarded.
- Not tried: trained replay (public weights not available in this sparse task environment; no substitutes), full pilot/descent/CE fitting, repeated trained-panel coverage, rare-event sampler discretisation error certificates, or full FLOP instrumentation. These need separately scoped work; this phase did not stop for compute.
- The KL numerical endpoints are ordinary floating-point roots, not outward-rounded certified numerical enclosures; the mathematical construction is valid, and a production implementation should round conservatively. Statistical coverage here tests the implementation at the frozen scale only.
- Negative scope is the generic observed-weight gate/normal-interval guarantee under the stated analytic proposals; it does not establish broad impossibility for defensive IS or dominating-point methods (Bai/Hesterberg), and it is not a new trained-panel failure.
