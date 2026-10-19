# walk_bound_check (T451): independent re-check of T449's NO-GO

↑ [TASKS](../../../TASKS.md) · [INDEX T449, H-E16, H-E17, T429](../../../INDEX.md) · [OWNER_QA 4.7, 4.13, 4.14](../../../OWNER_QA.md) ·
re-checks `claude/t449-walk-bound` (m16/approaches/walk_bound/README.md; its claims read, its code not used) ·
frozen line set: `claude/t429-walk-budget` @ 79ecae63 `frozen/` (copied verbatim to `frozen/`, sha256 prefixes asserted:
core 48dbba49ab13, estimator fb7902e12240, hom a91f019fca21, walk a38258f586b4)

## Question
T449 asked whether a bound computed from the weights can certify the error of the frozen T = 10 walk quadrature
(the average over a fixed line set) within ~10x of the actual error, and answered NO-GO: (a) the switch-point
sensitivity is heavy-tailed and unbounded at grazing lines, so the Lipschitz bound is infinite; (b) Koksma-Hlawka /
sphere-Sobolev bounds are infinite because of creases; (c) any cell enclosure stays >= 330x even at 8e13 cells.
Per AGENTS rule 5, is that NO-GO right: (1) do its two key numbers reproduce with different code, and (2) does the most
promising untried family also fail?

## Pre-registration (written 2026-10-19, before any computation in this folder)

**Panel.** DEV3 retrains dev3_s500 .. dev3_s504 (weights from `claude/t192-gpu-cost-law` `gpu_cost_law/dev3/`,
sha256 62c95203... asserted). The 5 zoo seeds T449 also used are not run: this container has no torch / alg_zoo (stated
as a scope limit, not a choice made after seeing data). HP5 (sealed) is not touched.

**Line set.** The frozen T10 rule exactly: 5 strata (23,0) (23,1) (23,2) (12,0) (12,2), stratum j uses
`core.sobol(10, n, 31415 + 101 j)` and `estimator.face_points_strat` with the radial (Student-t_9) weight. Only the
model-free geometry is taken from the frozen files; every model-dependent quantity is computed by my own code.

**My own walk (`plwalk.py`), algorithmically different from both the frozen walk and T449's `sens.py`.** Instead of
walking cell by cell along the line, it refines a breakpoint grid through time: the hidden state is a continuous
piecewise-linear function of s on a sorted grid; at each RNN step the pre-activations are evaluated at the grid points,
every sign change of a unit inside an interval adds its exact root to the grid, then ReLU. After T steps the logits are
linear on each grid interval and the wrong set is solved per interval. Gate: per-line wrong masses equal the frozen
`walk.walk` (radial, `hom.tcdf`) to <= 1e-10 absolute on every line used in step 1a; if more than 0.1% of lines fail,
I stop and debug before scoring anything.

**Step 1a: tail of the weight-derived switch-point sensitivity.** Lines: the first 2,048 nodes of each stratum
(10,240 lines per model). Switch points inside the window: (E) wrong-stretch endpoints that are decision-boundary
roots (L_k = L_j; not window ends, not the model-free target switch at s = 0); (G) gate switch points (z_{t,i} = 0).
Sensitivity rho = |P c| / |c . d|, c = input-space gradient of the switching function in the local linear region
(forward-mode Jacobian through the region's gate pattern), d the line direction, P the projection orthogonal to d
(T449's definition). Reported per model and kind: median, 99th percentile, max, and the Hill tail index at k = 2% of
the points (also k = 1%, 5%); the local gradient contribution k(s*) rho for E points (Student-t_9 weight at s*);
max rho over the first 256 vs all 2,048 nodes per stratum (growth = no finite sup).
- Reproduced if: per-model Hill indices (k = 2%) lie in T449's ranges widened by 0.3 (E: 0.9-2.7, G: 1.0-2.0), and
  max rho over 2,048 nodes exceeds max over 256 nodes in >= 4/5 models for both kinds.
- Theory check [to be stated as inferred]: if c . d has a positive density at 0, P(rho > x) ~ C/x (index 1); finite
  samples bias Hill upward, so I predict indices in 1-2.

**Step 1b: oracle zeroth-order enclosure width (T449's family (c)).** Partition per stratum: the configuration cells
in u_0 (360 for R23 strata, 45 for R12; 1,170 in total) times a dyadic box partition of u_1..u_9 with k splits applied
round-robin (u_1 first). Levels k = 0, 6, 12, 18, 24, 36 (1.2e3, 7.5e4, 4.8e6, 3.1e8, 2.0e10, 8.0e13 cells). Per model,
stratum and level: 400 random cells (numpy default_rng(451)), 64 uniform points per cell (u_0 at the configuration
centre). r0 = sampled range of g in the cell, a lower bound on that cell's width for any zeroth-order (per-cell
min/max) enclosure. Width W0 = sum_h M_h mean_cells r0. Reported: W0 per model, mean over models, and the ratio
(W0 / 2) / error.
- Error denominators: T429's DEV3-100 RMS 1.0e-3 / 2.0e-4 / 9.0e-5 at 64k / 1e6 / 4e6 FPE (the stated, generous
  choice T449's README names); T449's own 10-model RMS 9.5e-5 at 1e6 reported alongside for comparability.
- Reproduced if: my 5-model mean W0 at k = 12 and k = 36 is within +-30% of T449's 10-model means (.262, .0977), and the
  ratio stays > 100x at every level. Observation made while reading T449's log (before computing): its "330x" equals
  min-model half-width .0313 / 9.5e-5, not / 2.0e-4 as its README says; against 2.0e-4 the same numbers give 157x
  (min model) / 244x (mean). Both are > 100x, so the verdict would not change; I report both.

**Step 2: the most promising untried family = crease-aware cells with first-order (affine) enclosures.**
Why this one: family (c) came closest (hundreds of x, the others infinite), and T449's oracle width bounds only
zeroth-order enclosures, whose width shrinks like diameter x |grad g| ~ N^{-1/9}. A cell on which g is smooth (no crease
inside) admits an affine model integrated exactly plus a remainder interval, whose width shrinks like diameter^2.
The QMC alternatives are weaker candidates: worst-case bounds in reproducing-kernel spaces of dominating mixed
smoothness need smoothness r > 1/2 for point evaluation to be bounded, and certifying ||g|| is itself an integral of g
over the whole cube; bounds for functions with a known discontinuity set (Brandolini, Colzani, Gigante, Travaglini,
"On the Koksma-Hlawka inequality", J. Complexity 29 (2013) 158-172; Harman 2010; Pausinger & Svane, J. Complexity 31
(2015)) need the isotropic or polyhedral discrepancy of the 10-D node set, for which no computable upper bound exists
and whose known rate N^{-2/(d+1)} is slow. These are argued, not computed (marked [inferred]).

Measurement (oracle, so it can kill the family or leave it open; it cannot certify a GO): on the same sampled cells and
points as step 1b,
- r1 = min over affine l(u_1..u_9) of the sampled range of g - l (an LP; a lower bound on the remainder width of any
  affine-model enclosure of that cell);
- smooth flag: all 64 lines in the cell share one combinatorial signature (for each wrong-stretch endpoint: its type
  window / zero / decision root, and for decision roots the competing logit and the gate pattern of the region holding
  the root; plus the model-free creases: which low input is the maximum, and which term of the R23 window minimum is
  active). Different signatures inside a cell mean a crease or a grazing event lies between sampled lines.
- Scores per level: W1_box = sum_h M_h mean_cells r1 (affine models on box cells); W1_CA = sum_h M_h x
  median over smooth cells of r1 (stand-in for a perfect crease-aware partition with the same cell count: every cell
  assumed as easy as a typical smooth cell, extra cells for cutting along creases not charged; generous to the method).
  The smooth fraction is reported; a level with < 20 smooth cells of 400 (summed over strata, per model) has no W1_CA.
- Compute allowed for the bound: cells <= 10 B (each cell costs >= 1 line walk >= 1 FPE; generous, since a certified
  affine model needs more). Measured levels within that: k <= 6 at 64k, k <= 12 at 1e6 and 4e6.

**Decision rule for step 2 (T449's lines, applied to the oracle):**
- NO-GO for the family: W1_CA / 2 > 100x the error at every measured level within 10 B, for every budget, in the
  5-model mean; if it is also > 100x at 8e13 cells, the family is dead at any reachable compute.
- In between: 10x-100x within 10 B, or <= 10x only beyond 10 B (record the cell count and a COMPUTE_FRONTIER line).
- Not killed: <= 10x within 10 B. Not a GO, because the oracle is not a bound; it would make a real construction
  (rigorous Hessian / grazing control per cell) the next task.

**Prediction (written now):** step 1 reproduces (indices 1-2; W0 within 30%). Step 2: NO-GO within 10 B (smooth cells
are rare at reachable cell counts because lines cross many regions, and the inverse-CDF faces of the cube are steep),
with W1_CA at 8e13 cells somewhere in 10x-300x.

**Limits stated in advance.** Oracle quantities are sampled lower bounds per cell (64 points); a misclassified creased
cell counted as smooth enlarges W1_CA (against the method), which the median dampens. The u-cube parametrisation and
box cells are one choice; crease-aware cells in another geometry could do better than any box statistic, which is why
W1_CA charges nothing for crease cutting.

**Compute.** CPU only, <= 2 processes, each under `ulimit -v 2500000`. Estimated < 1 CPU-h total. Not used: HP5,
Fable, any contact with ARC. T449's code is not imported or copied.
