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

**Status:** done (negative, unverified): H is not favoured; mass-threshold pruning does not certify below sampling with slower growth.

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
### Phase 1: split attribution (`run34.sh`, `run5.sh`, then `python analyze.py; python report.py > report.md`)
Identity: all 11 instrumented runs ((8,3), (8,4) seeds 0-4; (8,5) seed 0) reproduce the unmodified recipe's acc, ops,
cells and resolved counts bit for bit (`analysis.json` "identical"; (8,5) reference = T185's run5/zoo0_recipe108.json:
acc .8358034749864574, 2.177e11 ops, 114155 cells, 84491 resolved). Full tables: [report.md](report.md).

**Resolved splits per model by driver rank** (mean over seeds; T5 = seed 0 only), growth per step in brackets:

| T | rank 1 | rank 2 | rank 3 | rank 4 | rank 5 | all |
|---|---|---|---|---|---|---|
| 3 | 64.6 | 54.0 | 64.8 | - | - | 183 |
| 4 | 1477 (x22.9) | 1146 (x21.2) | 1435 (x22.1) | 735 | - | 4793 (x26.1) |
| 5 | 22433 (x15.2) | 18371 (x16.0) | 18623 (x13.0) | 18459 (x25.1) | 6605 | 84491 (x17.6) |

**Decisive number: share of exact class-changing mass (absdE) from rank >= 4 drivers:** T3 0 (no such inputs),
T4 **0.160** pooled (per seed .354/.029/.157/.083/.022), T5 **0.207** (seed 0). Their share of splits: 0.153 / 0.297,
below the uniform-over-available-inputs baseline 0.257 / 0.406. Ranks 1-3 carry 84% (T4) / 79% (T5) of the mass;
rank 3 alone 42% / 41%. Robust to the attribution choice: rank >= 4 share of absdE by fractional attribution .26 / .22
(seed 0 at T4: .255), by gradient-magnitude driver .337 (T4 s0) / .189 (T5).

By step (T5 s0): splits 2% / 8% / 26% / 65% at t = 1..4, absdE 36% / 19% / 26% / 20%; median split-cell mass
2.9e-3 / 7.4e-5 / 2.3e-6 / 9.5e-8. **Size:** 56% of T5 splits sit in cells of mass < 1e-6 and carry 0.1% of absdE
(T4: 19% / 0.0%); cells < 1e-4: 87% of splits, 4.9% of absdE. 94% of T5 splits (80% at T4, 68% at T3) have their own
absdE < 1e-6, and those negligible splits are rank-mixed (rank >= 4 share .307 at T5, .155 at T4, the same as overall).
**Age:** the newest input (index t) is the driver of 37-45% of splits but 66-89% of absdE (within step 1 at T4/T5: 97-99%).
**Neurons:** splits are spread evenly (max neuron share 0.14-0.18 per model); absdE is concentrated (max 0.27-0.50),
with no consistent relation to the sampled clock score (Spearman -0.48..+0.67 across models; the zoo (8,T) models have
no clear clock/value separation: R^2_pos 0.1-0.6 for every neuron, `clocks/`).

**Cumulative-shift log** (`attr.py 8 T s guard`, pure log, sampled candidate-point ratios): summed frozen shift >=
|frozen margin| in 92/517 terminal tests at (8,3) (10% of resolved count) and 522/9041 at (8,4) (5.8%; 23% of the 2233
tests with >= 2 frozen gates; 2.2% of resolved count). 18 (8,4) tests have a single frozen gate with ratio > 1 at
another gate's test point (the per-gate test points are not exhaustive).

**Verdict on H / (a) / (b) / (c) [measured, unverified]:** H is not favoured. Low-rank-driven splits are fewer than
uniform, carry 16-21% of the class-changing mass, and removing all of them would still leave ranks 1-3 growing
x13-23 per step. (a) is partly right for counts: split counts are near-even over ranks 1-3 (and grow alike), not over
all ranks. (b) is favoured: growth is splits in ever thinner cells (median cell mass /30-/40 per step) driven by
ranks 1-3, whose individual effects are mostly negligible. (c) is not supported: splits are not neuron-concentrated.
**Separating tests not done:** direct tie-band geometry of the thin cells (distance to x_(1)=x_(2), x_(2)=x_(3) walls);
literal removed-split replay (vs the fixed-piece intervention); seeds 1-4 at T5; H = 16.

### Phase 2: budgeted refinement
`prune.py H T s tau cap|mass [oracle]`, grids `run3prune.sh` ((8,3)) and `runprune.sh` ((8,4), (8,5)), table `python prune_report.py > prune_report.md` ([prune_report.md](prune_report.md), all rows).
Cost convention T196: estimator FLOPs = 2 x m43 ops (all pruning tests, centroid forwards and masses of wrong-centroid
pruned cells charged); sampling charge 511 / 682 / 853 FLOPs per input at (8,3) / (8,4) / (8,5); SD = sqrt(p(1-p)/N)
at the retained cost; ratio_B = B^2/SD^2 (B = certified pruning bound; recipe108's own error is not certified),
ratio_act = (acc - exact)^2/SD^2. Exact references: (8,3) `exact3.py`, (8,4)/(8,5) estT_fix (T131/T185).
Bound validity: `capcheck.py` -> [capcheck.txt](capcheck.txt), 834 controls (narrow simplices 1e-1..1e-15 in d = 3,4,5,
repeated and antipodal rays, 519 recipe sub-cones), 0 violations. (An earlier cap version computed 1 - cos^2 and
returned 0 on narrow cones (review T55, 2eef376c); all results here use the fixed version.)

**Pooled (mean retained cells per model; max ratio over seeds):**

| test | tau | T3 cells | T4 cells | T5 cells (s0) | x(3->4) | x(4->5) | max ratio_B T3 / T4 / T5 | max ratio_act T4 |
|---|---|---|---|---|---|---|---|---|
| none (recipe108) | - | 233 | 6276 | 114155 | 26.9 | 18.2 | - | .52 |
| cap | 1e-4 | 159 | 3498 | - | 22.0 | - | 2.7 / 1.6e5 / - | .25 |
| cap | 1e-5 | 168 | 4269 | 78917 | 25.4 | 18.5 | .045 / 3.8e3 / 6.4e6 | .40 |
| cap | 1e-6 | 184 | 5135 | 89186 | 27.9 | 17.4 | 7e-4 / 40 / 7.5e4 | .47 |
| cap | 1e-7 | 207 | 5808 | 99988 | 28.0 | 17.2 | 6e-6 / .13 / 410 | .51 |
| cap | 1e-8 | 226 | 6103 | - | 27.1 | - | 9e-9 / 3e-4 / - | .52 |
| mass | 1e-4 | 110 | 686 | - | 6.2 | - | 13 / 1.9e5 / - | 258 |
| mass | 1e-5 | 142 | 1519 | - | 10.7 | - | .048 / 5.6e3 / - | 1.7 |
| mass | 1e-6 | 165 | 2606 | - | 15.8 | - | 5e-4 / 116 / - | .69 |
| mass | 1e-7 | 184 | 3808 | - | 20.7 | - | 7e-6 / 1.7 / - | .98 |
| mass | 1e-8 | 211 | 4923 | - | 23.3 | - | 6e-8 / .011 / - | 1.14 |

**Decisive answer [measured, unverified]: no tau gives ratio_B < 1 with retained cells growing clearly slower than
x13/step.** With the cheap outward cap test, ratio_B < 1 on all (8,4) seeds needs tau <= 1e-7, where 89-96% of cells
are retained and growth T3 -> T4 is x27-28 (unpruned x27); B ~ n_pruned x (cap bound), so it only passes when almost
nothing is pruned. Exact masses prune more (x23 at tau 1e-8 with ratio_B <= .011), but computing the mass of every
tested cell costs 2.3x the FLOPs of the unpruned recipe. At tau scaled to SD/10 (T3 1e-4 -> T4 1e-5) ratio_B is
1e3-6e3. The actual pruning error is far smaller than B (oracle: pruned mixed-correctness mass is ~1% of B for cap),
(8,5) s0 (cap, 1e-5/1e-6/1e-7): retained 78917 / 89186 / 99988 of 114155 cells (FLOPs -9% / -5% / -2%), growth
T4 -> T5 x17-18.5 (unpruned x18.2), ratio_B 6.4e6 / 7.5e4 / 410 against SD 1.7e-5 (ratio_act .018-.019, the recipe
itself stays accurate); ratio_B < 1 at T5 would need tau ~ 1e-9 or below, i.e. essentially no pruning.
At fixed tau the certified bound worsens along T because SD shrinks (6e-4 / 1e-4 / 1.7e-5) while the number of tiny
cells grows. The actual pruning error stays far below B (oracle: pruned mixed-correctness mass ~1% of B for cap),
i.e. the failure is certification, not bias: (p1) is not what happens at tau <= 1e-6, (p3) holds (the cap is
10-100x loose), (p2) holds at T4 (most retained cells have moderate mass).

## Compute accounting
Deductive throughout (no random model inputs) except the labelled sampled diagnostics (DIAG test, `clocks.py`).
Estimator ops (`m43.Ops`, multiply-add = 1) are reproduced exactly by the instrumented runs: (8,3) 3.6-4.0e5,
(8,4) 6.3e8-1.07e9, (8,5) s0 2.18e11. Instrumentation (counterfactual subtrees, masses, DIAG) is uncharged and cost
wall time: (8,4) ~90 s/model, (8,5) s0 2.9 h (vs ~25 s and ~1 h for the bare recipe here). Phase 2 FLOPs are in the
tables (charged as above); oracle diagnostics uncharged. Total CPU about 7.5 h ((8,5) attribution 2.9 h, (8,5) pruning 1.5 h, (8,4) grid 0.8 h, rest < 1 h), one process at a time, single thread,
`ulimit -v 2500000` (peak RSS ~330 MB).

## Claims (all unverified; nobody marks their own claim confirmed)
- [measured] The instrumented recipe108 is bit-identical to the unmodified one on 11 runs ((8,3)/(8,4) x 5 seeds,
  (8,5) s0).
- [measured] Splits whose driver input has rank >= 4 carry 16% (T4, 5 seeds; 2-35% per seed) and 21% (T5, s0) of the
  exact fixed-piece class-changing mass, and are under-represented among splits relative to a uniform driver.
- [measured] Split counts driven by ranks 1-3 alone grow x21-23 (T3 -> T4) and x13-16 (T4 -> T5) per model.
- [inferred] A rank-conditioned bound that pruned all rank >= 4 splits could remove at most ~30% of T5 splits and would
  leave >= x13/step growth: H as a remedy for the growth is not supported.
- [measured] Most splits are individually negligible (own absdE < 1e-6: 68% / 80% / 94% at T3/T4/T5) and rank-mixed;
  most sit in small cells (T5: 56% of splits in cells of mass < 1e-6, carrying 0.1% of absdE).
- [measured] Mass-threshold pruning with an outward cap bound gives ratio_B < 1 only when it retains ~90%+ of cells
  (T4); exact-mass pruning cuts cells but costs 2.3x FLOPs; no tau certifies below sampling with growth < x13/step.
- [measured, sampled candidate points] the cumulative frozen-shift condition fails at 5.8% of (8,4) terminal tests.
- [speculative] Splits are needed for exactness but individually negligible; a saving would need a joint (summed)
  certificate over many small pieces, which a per-cell mass bound cannot provide.

## Alternatives not ruled out
- The driver definition (contrast attribution) may misassign: a gate's sign change could be "caused" by a low-rank
  input entering through the running max. The fractional and gradient variants agree within ~0.1, but a causal
  intervention (perturb one input within the cell) was not done.
- The effect measure is a fixed-refined-piece intervention, not a literal removed-split replay; the latter could give
  larger effects (merged cells re-split differently).
- (b) was inferred from cell masses, not from direct distance to the tie walls; thin cells could be thin for another
  reason (e.g. gate planes nearly parallel to chamber walls).
- T5 rests on one seed; H = 8 only (H = 16 is T194's scope).
- Phase 2: a tighter outward bound (e.g. interval-arithmetic exact mass, or the chamber-restricted cap) could change
  the cap row; the exact-mass row bounds what any mass test can prune, but its cost could be lowered by caching
  masses computed later anyway (not tried).

## Failure modes
Splits with no rays strictly on one side (degenerate gates) are not recorded (they are never resolved). The cap bound
returns 1 for cones not within cos(theta) > 1e-6 of a half-space (never pruned). Thresholds on absdE (1e-13 "class-
changing") are near float noise for tiny cells; shares by mass are insensitive to it.

## Exhaustion checklist (rule 5: this is a negative (unverified) for H and for mass pruning)
Tried: driver = argmax contrast / fractional contrast / gradient-magnitude; effects exact (fixed-piece) and sampled
(DIAG); T = 3, 4 (5 seeds), 5 (seed 0); pruning by outward cap bound and by exact mass, tau 1e-4..1e-8, with oracle
single-correctness bound at T3/T4. Not tried, and why: a literal removed-split replay (cost ~ one recipe per split);
causal per-input perturbation; T5 seeds 1-4 (~3 h each); H = 16 (T194); joint pruning certificates (summed bounds over
siblings, which the speculative claim above points to); pruning rules not based on mass (e.g. by absdE upper bounds).
An independent re-check with different code (e.g. Codex lane h4's audit) is needed before "dead (verified)".
