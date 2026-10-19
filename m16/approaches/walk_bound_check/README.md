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

## Method
- `plwalk.py`: my own exact line walk (grid refinement through time, numba; described in the pre-registration), the
  line data of the frozen rule (face points, windows, targets, model-free crease labels), the combinatorial signature
  of a line, and the sensitivity rho from forward-mode Jacobians through the local gate pattern.
- `gate.py` -> `gate.log`: my walk vs the frozen `walk.walk` on all 51,200 step-1a lines.
- `step1a.py` -> `step1a.log`, `step1a.json`: step 1a.
- `cells.py MODEL...` -> `cells_<model>.npz`, `cells_a.log`, `cells_b.log`: per sampled cell r0, r1 (scipy HiGHS LP)
  and the smooth flag (steps 1b and 2).
- `analyse.py` -> `analysis.log`, `analysis.json`: the pre-registered scores. `diag.py` -> `diag.log`: exploratory
  checks of the smooth-cell statistic (not pre-registered, labelled as such below).

## Status
**Done (2026-10-19).** Pre-registration 12f7e1df, pushed before any computation; nothing in it was changed afterwards.
Step 1 reproduces T449's key numbers with different code, with one sub-check missed (below). Step 2, the untried
family, lands **in between** by the pre-registered rule: NO-GO within 10 B, but its oracle reaches <= 10x at 8e13
crease-free cells. T449's NO-GO therefore holds at reachable compute, but its "METHOD-limited, ~1e29 cells" verdict is
wrong for first-order crease-aware enclosures.

## Results
All numbers are from `analysis.log` and `diag.log`. Panel: dev3_s500-s504.

**Gate.** My walk equals the frozen walk on all 51,200 lines (5 models x 5 strata x 2,048 nodes). Worst |mass
difference| 2.2e-13; 0 lines over 1e-10 (`gate.log`).

**Step 1a: sensitivity tail. Reproduced, with one sub-check missed.**

| model | E: median / p99 / max | E Hill 1/2/5% | G: p99 / max | G Hill 1/2/5% | max k(s*) rho |
|---|---|---|---|---|---|
| s500 | .17 / 2.9 / 22 | 1.37 / 1.71 / 2.01 | 8.6 / 388 | 1.48 / 1.40 / 1.31 | 26 |
| s501 | .16 / 2.9 / 24 | 1.76 / 1.84 / 2.11 | 11.8 / 979 | 1.53 / 1.37 / 1.09 | 27 |
| s502 | .09 / 2.6 / 13 | 2.16 / 2.15 / 2.32 | 5.2 / 228 | 1.26 / 1.51 / 1.58 | 15 |
| s503 | .13 / 2.8 / 30 | 2.05 / 2.44 / 2.54 | 7.1 / 170 | 1.58 / 1.51 / 1.53 | 35 |
| s504 | .20 / 3.1 / 58 | 1.78 / 1.78 / 1.84 | 8.5 / 312 | 1.34 / 1.42 / 1.30 | 64 |

- Hill (2%) lies inside the widened T449 ranges for 5/5 models, for both E and G. T449's other figures also match:
  E median .06-1, p99 2.2-8, max 15-78, gates max 140-2,000, local |grad g| 17-91.
- Max growth from 256 to 2,048 nodes per stratum: E 4/5 (pass); **G 3/5 (fails the pre-registered >= 4/5)**. In s501
  and s504 the overall gate maximum already falls in the first 256 nodes. One extreme draw is compatible with a heavy
  tail, but this sample does not show divergence for gate points.
- Theory [inferred]: within one linear region a decision root is s* = -c.w / c.d. Large rho needs a small c.d, and a
  root inside the window then needs |c.w| <= tau |c.d|. So P(rho > x, root in window) ~ x^-2, which predicts an E tail
  index of ~2 (measured 1.7-2.4). That makes the gradient of g square-integrable only at the margin, and its supremum
  infinite: the Lipschitz bound of family (a) is infinite. This rests on the theory and the E tail, not on the G
  growth check.

**Step 1b: zeroth-order oracle width. Reproduced.**

| k | cells | W0, 5-model mean | min model | T449 (10 models) | (W0/2)/2.0e-4, mean (min model) |
|---|---|---|---|---|---|
| 0 | 1.2e3 | .411 | .336 | (.358) | 1,028x (840x) |
| 6 | 7.5e4 | .365 | .301 | - | 912x (752x) |
| 12 | 4.8e6 | .287 | .212 | .262 | 716x (531x) |
| 18 | 3.1e8 | .246 | .186 | (.223) | 614x (465x) |
| 24 | 2.0e10 | .196 | .148 | - | 489x (370x) |
| 36 | 8.0e13 | .108 | .086 | .0977 | 270x (214x) |

- At k = 12 and 36, W0 is within +9% and +11% of T449's means (pre-registered tolerance: 30%).
- Denominator: against T449's own 9.5e-5, the k = 36 ratio is 569x mean and 452x min, versus T449's "330x". T449's
  README says it divides by 2.0e-4 but actually divides by 9.5e-5; against 2.0e-4 its own numbers give 157x min.
  Either way the ratio is > 100x, so the verdict stands.
- At 64k FPE (error 1.0e-3) the W0 ratio is 182-206x within 10 B but drops to 54x at 8e13 cells. "> 100x at every
  compute" holds only at 1e6 and 4e6.

**Step 2: first-order (affine) enclosures, box cells and crease-aware cells.** Ratios are (statistic / 2) / error,
given as 5-model mean (min model). "n/a" means some stratum had no smooth cell.

| k | cells | smooth fraction | W1_box at 1e6 | W1_CA at 64k | W1_CA at 1e6 | W1_CA at 4e6 |
|---|---|---|---|---|---|---|
| 0 / 6 | 1.2e3 / 7.5e4 | 0 | 750x / 634x | n/a | n/a | n/a |
| 12 | 4.8e6 | .001-.004 | 478x (355x) | n/a | n/a | n/a |
| 18 | 3.1e8 | .002-.024 | 390x (298x) | n/a | n/a | n/a |
| 24 | 2.0e10 | .02-.05 | 290x (222x) | 3.8x (1 model) | 19x (1 model) | 43x (1 model) |
| 36 | 8.0e13 | .14-.24 | 135x (109x) | 1.1x (.5x) | **5.6x (2.3x)** | 12.4x (5.0x) |

- **Within 10 B (cells <= 6e5 / 1e7 / 4e7):** no model has a smooth cell in every stratum at any of these levels.
  Only .1-.4% of the 4.8e6-level box cells are crease-free. So W1_CA cannot be scored there. Affine models on box cells
  (W1_box, a lower bound for any affine-model enclosure on those cells) stay > 100x at every budget: 64k k = 6:
  127x (105x); 1e6 k = 12: 478x (355x); 4e6 k = 12: 1,062x (788x). **Within 10 B the family is NO-GO.**
- **Beyond 10 B:** at 8e13 cells, 14-24% of box cells are crease-free. On those cells, affine models leave a residual
  only 3-9% of the zeroth-order range (median r1 / r0), and the crease-aware stand-in drops below 10x at 64k and 1e6.
- Exploratory checks (`diag.log`, not pre-registered) on how optimistic W1_CA is:
  - The two heaviest strata (M = 4.37 and 2.71) have only 6-41 smooth cells of 400.
  - Their smooth cells have a median r0 1.1-6x below that of all cells, i.e. they are the easy cells.
  - Using the mean instead of the median raises W1_CA 2-3x: 2.3e-3 to 7.9e-3, i.e. 5.8x-20x at 1e6.
  - Applying the smooth-cell median r1/r0 (.03-.09) to all cells' mean r0 gives 8.0-15.4x at 1e6, i.e. ~10x.
- **Pre-registered verdict for step 2: in between ("<= 10x only beyond 10 B").** The crease-aware first-order oracle
  first reaches ~10x somewhere in 2e10-8e13 cells. That is the exact-region scale: COMPUTE_FRONTIER item 1 counts
  2e13-1e14 regions. The prediction (NO-GO within 10 B: right; W1_CA at 8e13 in 10x-300x: wrong, it is 2-20x) is
  half right.

**What this means for T449's NO-GO (reviewer's reading):**
- (a) holds: the switch-point sensitivity is heavy-tailed, and the Lipschitz bound is infinite.
- (c) zeroth order is reproduced: > 100x at every level at 1e6 and 4e6.
- Within compute <= 10 B, no family tried by T449 or here gets below 100x. For equal-compute use, the NO-GO stands.
- T449 calls its oracle width "a lower bound for any enclosure, T398 LPs included". That is too strong: it bounds only
  zeroth-order (per-cell min/max) enclosures. First-order enclosures on crease-free cells beat it by 20-50x at 8e13
  cells.
- T449 calls the 10x point METHOD-limited at ~1e29 cells. That does not hold for this family. If an oracle-tight
  first-order enclosure could be built, it would reach ~10x at ~1e13-1e14 cells: COMPUTE-limited, at roughly the
  exact-region cost.
- Even that route would not be cheap. Its per-cell remainder must be certified, and the heavy-tailed sensitivity
  (near-grazing lines) makes second derivatives unbounded inside some cells, so the real width would be larger.

## Compute accounting
- What counts: exact line walks (my walk). Nothing here is a bound computed at estimator cost; all cell statistics are
  oracle measurements on random points inside cells (the random cells and points are measurement only).
- Step 1a: 51,200 lines (plus root Jacobians), ~1 CPU-min. Gate: 51,200 lines x 2 walks, ~10 s.
- Cells: 5 models x 5 strata x 6 levels x 400 cells x 64 lines = 3.84e6 lines (~170 us each with signatures) +
  60,000 LPs: ~2.5 CPU-min per model, ~13 CPU-min in total.
- Total ~15 CPU-min. At most 2 processes at a time, each under `ulimit -v 2500000`.
- Reproduce (from this folder, `source ../../../.venv/bin/activate`; DEV3 files from `claude/t192-gpu-cost-law`
  `m16/approaches/gpu_cost_law/dev3/` into `data/dev3/`): `python gate.py`; `python step1a.py`;
  `python cells.py dev3_s500 dev3_s501 dev3_s502`; `python cells.py dev3_s503 dev3_s504`; `python analyse.py`;
  `python diag.py`.
- Compute frontier (standing rule) for the in-between family: ~1e13-1e14 crease-aware cells per model. Each cell needs
  its region found (item 1: 1e9-2e11 CPU-h) and a certified affine model plus a remainder bound, >= ~10 walks per cell
  (~80 FPE, ~5e5 FLOPs), so >= 4e19-5e20 FLOPs on top (1e7-1e8 CPU-h). Total ~1e9-2e11 CPU-h per model, ~$3e7-6e9
  today, /2-3 in 5 years. More compute would help S1 (a certified error bar) but not S2: the bound costs ~1e13x the 1e6
  FPE estimate it certifies.

## Claims
- [measured] My walk (a different algorithm) equals the frozen T10 walk to 2.2e-13 on 51,200 frozen lines.
- [measured] The weight-derived switch-point sensitivity is heavy-tailed on 5 DEV3 models: Hill index (2%) E
  1.71-2.44, G 1.37-1.51. This reproduces T449. The sample maximum grows with the line count for E (4/5) but not
  clearly for G (3/5).
- [inferred] P(rho > x, root in window) ~ x^-2 for decision roots, so g has no finite Lipschitz constant (T449 (a)).
- [measured] The zeroth-order oracle width reproduces T449 within 11% and is > 100x the error at 1e6 / 4e6 at all
  levels up to 8e13 cells. T449's "330x" uses 9.5e-5, not the 2.0e-4 its README states.
- [measured] Affine-model enclosures on box cells stay > 100x within 10 B at all three budgets.
- [measured] Crease-free box cells are rare (<= .4% at <= 4.8e6 cells, 14-24% at 8e13). On them an affine model
  leaves 3-9% of the range.
- [inferred] A crease-aware first-order enclosure is oracle-tight to ~2-20x the error at 1e6 FPE only at ~1e13-1e14
  cells. The family is compute-limited at the exact-region scale, not method-limited; this is an oracle, not a bound.
- [inferred, not computed] Fractional or mixed-smoothness QMC bounds and bounds for known discontinuity sets have
  no computable, certified constant here (pre-registration, step 2 rationale).

## Alternatives not ruled out
- The smooth-cell stand-in is optimistic: smooth cells are easier than average, and the cells needed to cut along
  creases are not charged. It is also pessimistic: a creased cell whose 64 lines all share one signature counts as
  smooth. The exploratory corrections move it 2-3x either way; the conclusion (10x only at ~1e13-1e14 cells) survives.
- Cells shaped in face-direction (w) space rather than u-cube boxes, or second-order (quadratic) models, could shift
  the cell count needed; not measured.
- Panel: 5 DEV3 retrains only (no zoo seeds, noisy copies or the target). At 64k the min-model W0 ratio within 10 B
  is 150-168x, near the 100x line.
- (b) Koksma-Hlawka / sphere-Sobolev was not re-measured numerically (not a key number in T451's brief).

## Failure modes
- Oracle statistics are sampled lower bounds per cell (64 points), and the cell means come from 400 random cells per
  stratum and level (no CI computed; model-to-model spread shown instead).
- Hill indices depend on k; reported at 1%, 2% and 5%.
- At k <= 18 W1_CA is undefined (too few smooth cells), so the cell count where crease-aware enclosures first reach
  10x is bracketed (2e10-8e13), not fitted.

## Exhaustion checklist
Tried:
- Own-code reproduction of (a) (sensitivity tail, plus a theory check) and of (c) zeroth order (oracle width at 6
  levels).
- First-order (affine) oracle on box cells.
- Crease-aware first-order oracle (smooth-cell stand-in), with exploratory bias checks.

Not tried, with reasons:
- A real certified crease-aware enclosure: it needs the exact-region enumeration (item 1).
- Second-order models; w-space cells.
- Zoo seeds (no torch here); HP5 (sealed).
- Numeric re-check of (b) (outside the brief).
- QMC fractional-smoothness and known-discontinuity-set bounds: argued only, since their constants are not computable
  with certification.

## Verdict for the DRI
T449's NO-GO is right where it matters for H-E17. Within compute <= 10 B, no weight-derived bound family tried
(T449's three, plus first-order box enclosures here) gets below 100x. Two corrections:
1. "330x" should read 157x (min model) / 244x (mean) against the stated 2.0e-4. My code gives 214x / 270x.
2. "METHOD-limited, ~1e29 cells" is wrong for first-order crease-aware enclosures. Their oracle reaches ~10x at
   ~1e13-1e14 cells: COMPUTE-limited at the exact-region scale.

Suggested ledger status: "dead (verified)" scoped to bounds at compute <= 10 B FPE. The crease-aware first-order
family stays open as COMPUTE_FRONTIER (S1 only).
