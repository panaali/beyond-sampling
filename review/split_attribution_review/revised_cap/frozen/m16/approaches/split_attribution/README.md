# split_attribution — T195: which recipe108 splits drive cell growth along T
↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [estT_fix](../estT_fix/README.md)

**Question.** recipe108 (frozen per-cell recipe, eps .1, `estT_fix`) is accurate at (8,4)/(8,5) but its cells grow
x13-250 per extra step. T187: an input's influence on the final margin follows its rank, not its age. **H:** the growth
comes mainly from gate splits triggered by low-rank inputs (rank >= 4 in the sorted input list), which a rank-conditioned
bound could prune.

**Alternatives written down before measuring (2026-10-17, commit 1 of this folder).**
- (a) splits (and class-changing mass) are spread evenly over trigger ranks (no rank structure to exploit);
- (b) growth is driven by the top-3 near-tie geometry itself (thin tie bands, H-E12): triggers are mostly ranks 1-3 and
  the extra splits sit in ever thinner cells;
- (c) growth is driven by a few "clock" neurons (time-since-max modes) regardless of trigger rank.
Separating tests: (a) vs H: share of splits / class-changing mass by trigger rank vs the uniform share; (b): mass share
with trigger rank <= 3, and splits by cell mass; (c): concentration of splits by neuron, and clock score vs share.

**Status:** STATUS_PLACEHOLDER

**Phase 2 (DRI request 2026-10-17): budgeted refinement.** Rule: a cell whose mass (bound) < tau is not refined; it
takes the true network's class at its ray centroid and its mass bound goes into a certified bound B. Alternatives /
risks written down before the (8,4)/(8,5) grid (after only the (8,3) pilot): (p1) pruned mass concentrates in the
error band (thin cells sit at tie walls), so the centroid class is biased and B is the realistic error, not ~0;
(p2) cells are numerous because of moderate-mass cells, not tiny ones, so pruning by mass barely slows growth;
(p3) the cheap cap bound is too loose to prune what exact masses would, and exact masses cost as much as the leaves;
(p4) cost is dominated by retained leaves' orthant masses, so cell counts understate/overstate savings.

## Method
Deductive instrumentation of the unchanged recipe (no estimator change). `estT.py` = byte-identical copy of
`estT_fix/estT.py`; `attr.py` re-writes `_finish` / `_margin_split` (recipe108 branch) line by line with hooks that use
separate `Ops()` and restore `STATS`, so estimates, ops, cells and resolved counts are **bit-identical** to the reference
(`ref.py`, the unmodified code; checked for all 11 runs, `analyze.py` column `identical`). Per resolved split (cell c,
gate k = neuron, step t = input index 1..T-1):
- **rank, exact**: every recipe cell lies in one ordering chamber (`halves`: permutation x sign of x0), so the rank of
  each input is known exactly (no interval needed). Rank 1 = largest input; class = rank 2.
- **driver input**: the gate row a = A[k] (pre-activation of neuron k at step t as a linear form in x_0..x_t). Contrast
  attribution con_i = a_i (mean_{pos rays} R_i - mean_{neg rays} R_i), sum_i con_i > 0: how much input i's variation
  across the cell drives the gate's sign change. Driver = argmax_i con_i; robustness variants: fractional
  (max(con_i,0)/sum) and gradient-magnitude (argmax |a_i| mean|R_i|).
- **exact class effect (within the recipe)**: for every step-t piece on k's minority side, error mass of the recipe's
  downstream subtree with the gate true (the normal run) minus with gate k forced to its frozen value (counterfactual
  subtree). `absdE` = sum over pieces of |difference| (net `dE` also kept). This is the first-order change in the
  estimate if that single split were pruned (joint pruning is not additive). "Class-changing" = absdE > 1e-13.
  **Label (review T55, 2eef376c):** this is a *fixed-refined-piece intervention* (the step-t pieces produced by all of
  that cell's splits are kept; only gate k's value is overridden on its minority pieces), not the literal effect of
  removing the split from the recipe (which would also change the later partition of the merged cell).
- **masses**: Gaussian mass of the split cell and of its minority side (`estT.mass`).
- **sampled DIAG** (T164 test, labelled sampled): 64 Dirichlet points + minority rays, true network vs gate k frozen.
- **cumulative-shift log** (DRI request, `attr.py H T s guard`, pure log): at each terminal margin test with frozen
  gates, max over the union of the per-gate candidate test points of sum_k (frozen shift_k) / |frozen margin|.
  **Label:** these are sampled candidate-point ratios (a finite-point lower bound on violations), not measured extra
  splits of a fixed recipe.
- **clock labels** (`clocks.py`, sampled, diagnostic only): per neuron R^2 of final h_j on one-hot argmax position
  minus R^2 on top-3 values.

Files: `attr.py` (instrumented run, resumable per half), `ref.py`, `analyze.py` -> `analysis.json`, `report.py` ->
`report.md` (pooled tables), `clocks.py` -> `clocks/`, `run34.sh`, `run5.sh`; records in `out/H8_T{T}_s{seed}.npz`.

## Results
RESULTS_PLACEHOLDER
