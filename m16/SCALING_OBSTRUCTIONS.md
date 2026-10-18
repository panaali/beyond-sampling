# Scaling obstructions for deductive accuracy estimation of M_16,10 (2026-10-18) — board task T225

Status of every claim: **[measured]** or **[inferred]**, and **verified** (independently re-checked, DRI-confirmed) or
**unverified**. Sources are `branch@commit, file`. Branches: shared = `ccr-6a61f25c-wn80ii`; T194 = `claude/t194-width16-probe`
(this file); T192 = `claude/t192-gpu-cost-law@e5a40128`; T218 = `claude/t218-tie-tail-labels@8645f749`.

## Executive summary (DRI, 2026-10-18)
1. **Not solved.** No deterministic method we know beats sampling on M_16,10. The best deterministic number is ~1.1-1.3x better than sampling, on the clean target only, with a fixed node set (quadrature class) (§4).
   **[Corrected 2026-10-18, T263]** That 1.1-1.3x is the MSE averaged over 16 random scrambles of the node set. Averaging
   over random scrambles is randomized quasi-Monte Carlo (RQMC), which is sampling by ARC's definition; one fixed rule is
   deterministic, but its error is a single draw from that spread. Fixed nodes are also *inductive* under ARC's definition
   (a step that relies on chosen input points rather than being derived from the weights; TASKS Directive 8). The result
   predates the Directive 7 panel. So it is quadrature-class evidence on the clean target, not a deductive win (§4).
2. **Per-region methods cannot scale.** The target's exact prefix regions are 34 / 792 / 21,032 for k = 2-4, **confirmed by an independent exact-arithmetic recount**, then 632,394 at k=5 and ~2.3e7 at k=6, growing ~x23-30 per step. Covering all but 1e-5 of the probability needs ~x15-19 more regions per step (confirmed for k <= 4). Extrapolated to T=10: ~1e10-1e13 regions (§3.2-3.4).
   **[Corrected 2026-10-18, T263]** Measured versus extrapolated. Measured and verified: region counts at k = 2-4, and
   the coverage step k = 3 -> 4 (x16.1). Measured, unverified: the k = 5 count and the coverage step k = 4 -> 5 (x14.9).
   The k = 2 -> 3 coverage factor (x18.6) is capped, because at k = 2 all 34 regions are needed. The k = 6 figure is an
   estimate from a completion-censored sample, not a count (§3.4). Everything at T = 10 is a conditional extrapolation of
   these 1-2 measured steps: it holds only if the per-step growth stays in the measured range for six more steps. The only
   rigorous bound is N_10 >= 64 N_4 = 1.35M gate histories (T55). Read this item as: region-by-region methods look
   infeasible for the target *if* the measured growth continues. T270 (complete k = 6 masses, running) tests the next step.
3. **Training does not reduce the region count** (it matches norm-matched random weights); it *moves* regions from the bulk to the tie walls, where the trained net is finer than random (§6-7).
   **[Corrected 2026-10-18, T263]** Scope: measured at k <= 4, on one trained network against 10 random draws and 2
   scrambled controls (§6). It is not a law for larger k, and it does not isolate the causal effect of training.
4. **Constant-factor savings** (cheaper start, pruning, merging, symmetry, event words, labelled tails, certificates) do not change the growth (§3.5).
   **[Corrected 2026-10-18, T263]** Scope: this holds for the variants and models tested, mostly at T <= 5. One cited
   ceiling, T206's oracle "at most 2-4x", is not an upper bound (§3.5).
5. **What a solution must do:** describe the probability of the tie-wall band compactly without resolving its regions. That is untested ground (§5).
   **[Corrected 2026-10-18, T263]** The band alone is not enough. Measured per input (T269, 2M samples per model), 14% of
   the target's error lies outside the band (gap >= .05, not blind), and 19-63% on the 5 zoo seeds and 2 noisy copies
   (§5). T239's tie-wall picture (§7) assigns each leaf's whole mass by the leaf's centre, so its stratum masses are not
   the mass of inputs inside each gap band.
6. **[Added 2026-10-18] The cheap regime (§9).** Scored at its own cost over a panel of models, as ARC frames the
   challenge, no deductive estimator comes within 7x of the lenient bar on 80 retrains (best F 76.5 at 3.9k FPE, T286)
   [measured, unverified]. The cheap approximations fail for four reasons: frozen gates erase the model-specific signal,
   Gaussian closures break by steps 7-9, relaxation bounds widen ~50x per step, and exact branching grows 5-19x per step
   [inferred]. Kept: deductive errors on retrains are near-constant offsets (corr .96 with the truth), and R0's oracle
   floor is .51 pp. Open: R0's integral (T297, T298), the shadow screen (T290), k = 6 masses (T270). No impossibility
   proof exists.

## 1. The question and the bar
Estimate the accuracy of the 432-parameter 2nd-argmax RNN M_16,10 (`alg_zoo.example_2nd_argmax()`, accuracy
0.95313575 ± 3.7e-5 from 3.2e7 samples). **Answer key: 0.9531174789 ± 3.9e-7** from 3e11 float64 samples in independent
Philox blocks (between-block SD 3.4e-7) [measured, unverified; T192@2ea6766d gpu_cost_law/results/mc_target/blocks.jsonl, summary.json] without random model inputs, with MSE
no worse than plain sampling at equal compute. Also on the 5 trained seeds and noisy copies.
Compute is in T196 FLOPs (multiply-add = 2, other scalar op = 1; sampling costs c_s = 5948 FLOPs per input at (16,10))
[shared@bf9f0ead, review/cost_convention/README.md; DRI ruling]. At cost C FLOPs, sampling's standard error is
sqrt(p(1-p) c_s / C) ≈ **16/sqrt(C)**: 1.6e-4 at 1e10 FLOPs, 5e-6 at 1e13. A method beats sampling only if its
error is below that.

## 2. What works
- **M_4,3 (32 params):** a mechanism-derived estimator (E36) and one frozen generic recipe (E37) beat sampling on all 5
  trained seeds under ARC's noise protocol (MSE ratios 0.0002-0.25) [measured, **verified** (scoped): H-U1/H-U2, INDEX].
  Cost ≈ exact enumeration there.
- **T = 3 ladder (H = 4..8)** and **(8,4)**: recipe108 (resolve-by-impact plus a multi-step margin certificate) passes.
  (8,4): MSE ratio <= 0.27 on 9 models, <= 0.36 under the T196 op audit [measured, unverified: shared, estT_fix/README.md
  E46; T192 README §3-4].
- **(16,3) and (16,4)** (target width): the frozen recipe108 beats sampling on all 5 seeds + 4 noisy copies. Ratio <= 4e-5
  at (16,3) (1.7e6 ops); <= 0.017 (0.035 T196) at (16,4) (2.5-3.0e9 ops) [measured, unverified; T194@a594da1,
  width16_probe/README.md "Results"; audit T199 pending].
- **[Corrected 2026-10-18, T263]** The rung wins above (T = 3 ladder, (8,4), (16,3), (16,4)) and the (8,5) panel in §3.1
  predate Directive 7 (TASKS, DRI ruling 2026-10-18). Under it, a rung passes only if one frozen method beats sampling on
  all 5 trained seeds and on >= 8 noisy copies per seed at each of ARC absolute sigma .05 and relative sigma .005 and .01,
  with every model and budget reported with a CI. These panels were 5 seeds plus 4 noisy copies of seed 0 (relative sigma
  .005 at (8,4), (16,3) and (16,4)). So they are evidence at the panel tested, not Directive 7 passes.

## 3. Where it breaks
**3.1 The recipe stops winning at T = 5.** (8,5), 9 models, MSE ratio in estT ops / T196 FLOPs (estT ops x1.37 audited):
seed 0 0.0097/0.013, seed 1 0.0057/0.0076, **seed 2 3.4/4.6**, seed 3 1e-5/1.3e-5, seed 4 3e-7/4e-7; fresh-noise copies of
seed 0: **1000 1.0/1.4**, 1001 0.0019/0.0026, 1002 0.0025/0.0034, 1003 0.027/0.036. Under T196 that is **7 wins, 2 losses**
(seed 2 and copy 1000; copy 1000 ties in estT ops) [measured, unverified; T192@e5a40128, gpu_cost_law/results/summary.md]. (16,5) zoo0 loses (ratio ~35, error -3.2e-5 vs exact) [measured,
unverified; T192@2930bc92]. From T = 4 the recipe saves only 1.2-1.6x ops over exact enumeration [measured,
unverified; T192 §3; T194@45e1f6c width16_probe/README.md "T206"].

**3.2 Exact enumeration is infeasible.** Gate-pattern regions of the target's first k steps: **34 / 792 / 21,032 /
632,394** for k = 2..5, growth **x23.3, x26.6, x30.1** per step and rising. That extrapolates to ~1.5-3e13 regions at
k = 10, and ~1000 CPU-years of geometry before masses. k = 2-4 counts are [measured, **verified**]; k = 5 is [measured,
unverified]; everything beyond is [inferred] (T192 README §9, results/prefix/k2-k5.json; INDEX H-E13). Reproductions with different code: k = 2-4 by T194's numpy port
(identical; T194@ee01b49 width16_probe/prefix.py) and k = 2-3 by T218's LP counter (T218 tie_tail_labels/README.md).
Neither is independent exact arithmetic. **Independent check:** Codex T55 recounted k = 2/3/4 = 34 / 792 / 21,032 with a
sequential plane-arrangement count in exact arithmetic over the stored weights (reviews 13e3a70e, ad1d458f, 895fb566;
shared@619436ad m16/approaches/feynman_fundamentals/t219_k4_saved_review.json). The DRI marks k <= 4 CONFIRMED (H-E13).
Caveat (T55): the only rigorous bound is N_10 >= 64 N_4 = 1.35M gate histories. The ~1e13 is an extrapolation of the
measured growth, not a bound. Zoo seed 0 grows alike: (16,3) 776 -> (16,4) 26,742 -> (16,5) 786,114 regions (x34.5, x29.4). (k <= 4: T194@95ed88e width16_probe/regions/; k = 5: T192 §8. Regions are start-independent, T192 §5; §8's 838 / 28,200 are 5-seed means.)

**3.3 Cost is cone masses.** Exact-mode ops are 95% (16,4) / 97% (8,4) cone-mass quadrature. A leaf that needs a
mass needs ~6 simplices at ~3e4 ops each (d = 4: 440x the d = 3 closed form; d = 5: 5.2e4-1.7e5) [measured, unverified;
T194@45e1f6c "T206"; T192 §5]. 34-56% of leaves are fully correct and need no mass.

**3.4 Mass concentrates, but not fast enough for the absolute size.** The number of regions needed to cover 1 - 1e-5 of the prefix mass grows
**x18.6 then x16.1** per step on the target (x17.5-21.6 on zoo (16,4)/(16,5)). Even N_1e-2 grows x8.6-13, at 65-93% of the
count's log-rate. Projected N_1e-5 at k = 10 is ~2e11 [measured k <= 4, rule pre-registered; inferred beyond;
T194@57bad7b width16_probe/README.md "T222", coverage.json]. At **k = 5** (T192@f02482d6, gpu_cost_law/results/prefix/k5_regionmass.npz; 632,394 regions, mass sum within
1.4e-9; numbers recomputed from the file by T194), N_delta for delta = 1e-1..1e-6 is 2,548 / 14,658 / 42,499 / 89,216 /
152,956 / 226,080. Growth k4 -> k5 is x6.4 / 8.4 / 10.6 / 12.7 / 14.9 / 17.2 (k3 -> k4: x6.0 / 8.6 / 10.4 / 13.3 / 16.1 /
18.6). The inverse-Simpson effective count is 397 (x3.0, then x3.5) [measured, unverified]. k = 6 (T192@2ea6766d README §11; capped random-subtree sample, 245 of 3,811 subtrees): 1,473,541 cells,
estimated **2.3e7 cells (x30.6 over k = 5)**. The sample holds 1,459,727 distinct regions (a lower bound) and only 8.8% of the mass, so its N_delta is
not meaningful. Growth stays ~x30 per step through k = 6 [inferred, ~~unbiased~~ subtree estimate; unverified].
**[Corrected 2026-10-18, T263]** The estimate is not shown to be unbiased. Subtrees were dispatched in random order, but
only those that finished before a wall-clock deadline were kept, so the 245 completed subtrees are selected by runtime
(completion censoring). The 2.3e7 (22.9M) is therefore a censored, uncalibrated estimate; the direction and size of its
bias are unknown, and the completed counts are only lower-bound evidence (T255,
[t192_latest_evidence/HANDOFF.md](approaches/feynman_fundamentals/t192_latest_evidence/HANDOFF.md)). T270 (k = 6
complete masses, running) will replace it.

*Precision scaling vs absolute size.* Per decade of delta at k = 5, N grows x5.75 / 2.90 / 2.10 / 1.71 / 1.48, far
below sampling's x100 per decade of eps (cost ~ eps^-2). But the factors **rise with k**: the deepest decade (1e-5 -> 1e-6)
costs x1.12 / 1.29 / 1.48 at k = 3 / 4 / 5, and the first (1e-1 -> 1e-2) x3.05 / 4.39 / 5.75. The delta exponent drifts
from ~0.05 to ~0.17, so it is not stable, and by k = 10 the deep decades may cost x3-5 each [inferred; T192@f02482d6
k*_regionmass.npz; k = 3-4 factors also from T194 coverage.json]. So at the depths measured, precision is still much cheaper
than for sampling, but the advantage shrinks with depth, and the absolute size dominates: ~1e10 regions at T = 10 if the
per-step growth keeps falling, ~2e11 at a constant x15-17, each needing d = 10 cone masses. T192's measured throughput
on the (16,5) run: geometry 4.2e6 counted ops/s per CPU core (3.2e7 on 11 workers); GPU masses ~7.7e9 counted ops/s
(7.1e4 simplices/s, d = 5); wall time geometry-bound at T >= 5 [measured, unverified; T192]. That puts the target far
beyond our budget, and it cannot be demonstrated. Independent
replication (T218 child: LP cone-splitter + robust-angle masses; claude/t218-tie-tail-labels@a498f53d, tie_tail_labels/
README.md Part 2) gives regions 34 / 792 / 21,032 and k = 4 N_delta 1,737 / 4,018 / 7,021 / 10,233 / 13,165 (T194: 13,164).
The DRI marks the k <= 4 coverage CONFIRMED [measured, **verified**].

*Which precision is required (review A1).* N_1e-5 is the wrong yardstick. A region-by-region method with N regions and
d = 10 cone masses costs >= ~1e6 FLOPs per region (d = 5 is already 5e4-1.7e5 per simplex, ~6 simplices per leaf; §3.3). At
N ~ 1e10 that is >= 1e16 FLOPs, where sampling's SE is ~1.6e-7. So the uncovered remainder must be bounded to ~1e-7, and the
deep decades are the ones whose growth rises with k. An *exact* method meets "MSE <= sampling at its own cost" trivially,
so the exact route is purely a feasibility question: the full count with a remainder < 1e-7, at d = 10 mass cost. The
evidence for the growth is thin in both directions. The verified coverage factor is one unsaturated step (k = 3 -> 4,
x16.1; 2 -> 3 is capped because N = regions at k = 2) plus the unverified 4 -> 5 (x14.9), falling ~8%/step, so six-step
extrapolations are weak (3e10 to 2e11 at N_1e-5). The deciding numbers are g(5 -> 6) and g(6 -> 7) at delta ~1e-7. k = 6 full
masses (2.3e7 regions, d = 6) are feasible on the GPU already used and were not computed [inferred; review A1].
**[Update 2026-10-18, T263]** T270 is now computing them (running). The 2.3e7 here is the censored cell estimate above.

**3.5 Structural savings found so far are constant factors.**

| idea | saving | status, source |
|---|---|---|
| truth-cone start (2T(T-1) cones instead of 4·T! chambers) | lossless; 0.67x / 0.76x ops at (8,4) / (16,4); chambers ~10% of log-growth | [measured, unverified] T192 §5; T194 "T206"; shared@ab532a27 review/chamber_split |
| oracle skip of small-error leaves (bias <= 1 sampling SE) | <= 1.9x (8,4), 3.8x (16,4); 4.8x with truth start. ~~An upper bound for a *certified* method, which must bound each cell (but ARC's bar is empirical MSE, not a certificate; see §5)~~ **[Corrected 2026-10-18, T263]** Not an upper bound. The oracle drops leaves in order of their exact error, ignores what each leaf costs, and fixes the allowed bias at the full-cost SE. A cost-aware choice can skip more: with errors .1/.2, costs 1/100 and an allowance of .2, error-first skipping saves 1, but skipping the second leaf saves 100 (astro review A3). Read it as a heuristic ceiling under error-first ordering. (A certified method must also bound each cell's error without computing it, but ARC's bar is empirical MSE, not a certificate; see §5.) | [measured, unverified] T194 "T206" |
| merging same-history leaves / same-future cells (T188) | exact merged cells 0.59-0.82x at T = 3-4 (T = 10 extrapolation 2-5x at H = 16); T188 counts with no savings claim | [measured, unverified] review/chamber_split; shared TASKS T188 handoffs b9a6b5df/3ee0667b |
| budgeted pruning by cell mass (T195 phase 2), (8,4) 5 seeds, tau 1e-4..1e-8 | cap bound: bound^2 ratio < 1 on all seeds only at tau <= 1e-7, where 93-97% of cells are retained and T3 -> T4 growth stays x27-28. Exact-mass pruning: growth x20.7-23 but ratio_B >= 1.71 at 1e-7 and ~2.3x the FLOPs. No certified setting slows growth | [measured, unverified] claude/t195-split-attribution@5d58cd51, split_attribution/prune_report.md |
| weight symmetry (T216) | none: no permutation symmetry; projecting the near-collinear incoming rows {1,6,7,8; 4 anti} costs 79 pp (8.9 pp for the closest pair) for <= 2.7x fewer regions, growth still >= x16/step | [measured, unverified] T194@ee01b49 "T216" |
| event words record/2nd/low (T217) | modal gate word holds 21% of an event word's mass at k = 4 (go: 99%) | [measured, unverified; DRI: kill] T194@050bfaa "T217" |
| mechanism-labelled tie tails (T218) | top-100 labels cover 2-4% of out-of-sample tail mass. Codex review T227 (791b2d5f) caveats: the random-label null does not preserve label sizes, the half-size check thins an already-selected tail, and within-label stats condition on the global tail. So the kill holds only for the tested label families; there is no universal exclusion | [measured, reviewed with caveats; kill scoped] T218 README; T227 |
| rank-coordinate certificate (T189): fixed zero-anchor / global-absolute-envelope Chernoff family | exact width floors .0853 (H4T4) / .0426 (H8T4 seed 0); sampling's maximal SD is smaller at every budget >= 7630 / 94116 FLOPs, even with the certificate free. A limitation of this family only, not an MSE lower bound | [measured, unverified] shared@8269963f TASKS T189 backlog row; conditional_order_bounds/box_mass/HANDOFF.md |
| T189 local affine follow-up (same models) | reported, final saved review pending: enclosures [.071, .866] (H4T4) and [.068, 1] (H8T4), > 10x sampling SD at 1e7 FLOPs; crossover 8720 / 19778 FLOPs | [measured, unverified, review pending] shared@f18320b2, conditional_order_bounds/local_certificate/README.md |

**3.6 Noise does not change the region count, but it changes where the error is.** At T = 4 the noisy copies (relative
sigma .005-.02, ARC absolute .05) have 0.94-1.21x the clean regions and 0.94-1.33x the ops [measured, unverified; T192 §7;
not tested at larger T]. Counts are the wrong invariant, though (review A3). Every exploitable structure found (tie-wall
concentration, the blind region's 60% of the face second moment, the 1.1-1.3x of §4) is a clean-model fact. On the relative
sigma .01 copy the blind share is 8% and 65% of the tie-slicing tail sits at u > 0.5 (T218). At ARC's absolute sigma .05
(its M_4,3 panel level) six copies of M_16,10 have accuracy **0.12-0.28** (2e5 samples each; T271 check), so the errors are bulk errors.
No tie-band method addresses that regime [measured, unverified].
**[Corrected 2026-10-18, T263] Noise scope.** The region-count result above includes absolute sigma .05, but only at T = 4.
Leaderboard v3 (LB3), the source of the estimator ratios in §4, has noisy groups at *relative* sigma only (.0025 / .005 /
.01 / .02 times each matrix's RMS; `SIGMAS` in [leaderboard.py](leaderboard.py)). Directive 7 (TASKS) also requires ARC's
absolute sigma .05, where the copies' accuracy is .12-.28. No LB3 number speaks to that regime. The planned sealed panel
HP1 (T276) includes it.

**3.7 Region counts in earlier tables were understated.** The 2e5-sample lower bound missed up to 92% of regions:
(8,5) zoo0 has 42,322 exact regions, against 3,433 sampled. Restated against exact counts, recipe cells/regions are 0.99 / 1.99 /
2.70 along T at H = 8 (published 2.8 / 5.4 / 33) and 2.44 -> 0.46 along H at T = 3 [measured, unverified;
T194@95ed88e "T205"].

## 4. The best deterministic number
sci_tao's tie-slicing with the blind region (<= 1 positive input, P = 11/1024) split off exactly. Over 16 independent rule
scrambles on the target: MSE ratio **0.75 (16k FPE) / 0.89 (64k)**, i.e. **1.1-1.3x better than sampling**, on the
clean target only. Quadrature class, not mechanistic. On leaderboard v3 pooled over 30 models it
does not beat sampling at any budget (3.4-6.5; plain sci_tao 1.4-3.6, CIs reaching below 1 only at 1k and 256k). The
summary's "sci_tao 0.03 @ 256k on orig" is one lucky run (0.99 / 1.56 over scrambles)
[measured, unverified; T194@2809650d m16/approaches/tie_blind/README.md].
**[Corrected 2026-10-18, T263]** Three qualifiers. (1) The 0.75 / 0.89 are averages over 16 random rule scrambles. That
average is the MSE of RQMC (quasi-random nodes with a random scramble), which is sampling by ARC's definition. One fixed
rule is deterministic, but its MSE is one draw from that spread (compare the 0.03 vs 0.99 / 1.56 above). (2) Fixed nodes
are inductive under ARC's definition (TASKS Directive 8). (3) The result predates Directive 7, and LB3 has relative-sigma
noisy groups only (§3.6). So this is quadrature-class evidence on the clean target, not a deductive or a Directive 7 win.

## 5. What a successful method must do, and what is untested
**First, the regime** (review A3): the clean target and a noisy panel are different problems. Tie-band structure exists only
on the clean model and at small relative noise. At ARC's absolute noise level the copies make bulk errors (accuracy 0.12-0.28).
A method must state which regime it targets.

**Must** [inferred]: aggregate across regions rather than visit them. ~1e13 linear regions, and an exact route needs the full
count with a remainder < ~1e-7 (§3.4). The error is not a few families but a continuum of rare tie-band events (T218), spread
over many gate words per mechanism-level event word (T217). It is also not confined to the band (review A2). ~86% of the error
is within tie distance 0.05 or in the blind region, but **~14% is a soft tail** in mixed leaves far from ties: 3.1e-4 of
2.13e-3 at (16,4) zoo0 (T239), and at T = 10 the far 80% of mass errs at 0.83%, i.e. 0.67 of 4.7 pp (T241). The gap-binned
error rate decays smoothly, with no band edge. A band-only method that leaves the far part to a bound leaves ~0.67 pp, about 40x
the sampling SE at 1e10 FLOPs.
**[Corrected 2026-10-18, T263]** The T239 figure assigns each leaf's whole mass and error by the leaf's centre, which is
not the mass of inputs actually inside a gap band (T254). A per-input check confirms the tail (T269,
`claude/t241-neuron-account@8e1da8ac`, gap_formula/README.md, 2M fresh inputs per model; gap = min(M1 - M2, M2 - M3) of
the top three input values, so these thresholds are in input units, not T239's angular distance). Non-blind inputs with
gap >= .05 carry 14.1% of the target's error (0.66 pp), 19-45% on the 5 zoo seeds (1.0-5.4 pp) and 38-63% on the relative
sigma .005 / .01 copies (4.1-15.2 pp). The weaker models differ from the target mostly in this tail [measured, unverified].
**Design rule** (review A6): ARC scores empirical MSE on a panel, not certificates. Certified bounds have been 10-100x and more
above the actual error (T195: cap tau 1e-5 at (8,5), bound ratio 6e6 vs actual 0.019). Uncertified truncation and pruning rules
should be judged by their measured MSE. The answer must therefore come from a quantity that is smooth across
regions (a bound or an expansion in the order statistics, the clock phase or the margins) and be certified to ~16/sqrt(C).
Region-by-region methods inherit >= x16/step.

**Not ruled out** (from the READMEs' "alternatives" sections):
- precision-matched cone masses: quadrature order set by each cell's needed precision (T206; deprioritized as T212);
- richer event words: top-2 gap, age of the max, signs of all inputs (T217);
- intermediate-resolution or continuous-feature tail labels (clock phase) and tail-aware importance rules (T218);
- symmetries beyond neuron permutation and sign: general linear maps mixing neurons, task symmetries such as time
  reversal (T216);
- the coverage curve at k >= 5 (pending), per cell rather than per region, and after same-future merging (T222);
- exact treatment of the all-negative orthant (1/1024, ~~all gates start off~~ **[Corrected 2026-10-18, T263]** not all
  gates start off: the model has no biases and h_0 = 0, so h_1[j] = ReLU(Wi[j] x_0), and a negative x_0 switches on every
  neuron j with negative input weight Wi[j]; [common.py](common.py), and T218 notes these neurons keep branching inside
  the blind region) inside the blind region; fixed rather than
  pilot-based allocation; skipping the blind middle of R23 walks (T223);
- certificates with rank/domain/reachable-state restrictions, other anchors, one-sided envelopes, and cancellation-aware point
  estimators (T189 handoff: "remain open, not dead");
- whether the noise insensitivity of region counts holds beyond T = 4 (T192);
- a genuinely independent region recount in exact arithmetic (T219, pending).

## 6. Trained vs random weights (T232)
**Measurement.** Exact prefix regions of the target vs 10 random draws (5 Frobenius-, 5 row-norm-matched) and 2 scrambled
controls (W_hh row-permuted; sign-flipped). The random draws grow x22-26 (k = 3) and x25-30 (k = 4) per step, the same law as the trained x23.3, x26.6. At k = 4 they have
19,354-26,074 regions vs 21,032. At k = 2 every model has the generic maximum, 34. Training concentrates mass only at coarse
resolution: N_1e-2 is 1.7-6x and the inverse-Simpson count 2-8x smaller for the target, N_1e-4 ~1.9x, and N_1e-6 0.8-1.7x
[measured, unverified; T194 branch, m16/approaches/random_control/README.md, pre-registered at 1d6f4db9]. The trained
M_16,10 is **generic in region count**: whatever training builds is in what the regions compute, not in how many there
are [inferred].
**Theory note** [literature; not tested here]. ARC's no-coincidence conjecture concerns exactly this contrast. For random
networks no short explanation should exist; for trained ones a mechanistic explanation should make the accuracy cheaply
estimable. Our data rule out "few linear regions" as that explanation. Worst-case impossibility for white-box
deterministic estimation is out of reach with current tools: it ties to derandomizing circuit acceptance probability (CAPP)
and circuit lower bounds (Impagliazzo-Kabanets-Wigderson; Williams). Unconditional lower bounds exist only for black-box
query access, e.g. deterministic convex-body volume (Elekes; Bárány-Füredi). The obstructions above are empirical, not theorems.

## 7. Is the error part "shattered"? (T239)
Owner's hypothesis: the correct part is simple; the error part is shattered, random-like, and makes the counts. Pre-registered tests (T194 branch, m16/approaches/shatter_test/README.md, rules at 1cd671cc; unverified):
- **Error vs correct coverage** (final exact leaves, zoo seed 0): at T = 4, the cells covering 99% of the error mass are
  1.17x (H = 8) and 0.75x (H = 16) those for 99% of the correct mass. So not more spread at T = 4 (rule: >= 3x; not supported).
  But from T = 3 to 4 the error coverage grows x24.9 / x44.3 vs x8.3 / x18.4 for correct, 2.4-3x faster [measured].
- **Where the cells are:** cells per unit mass are 26-27x higher within angular distance 0.05 of a class-changing tie than
  beyond 0.1, at T = 4 (14-19x at T = 3) [measured; supported].
- **Against random weights** (prefix k = 4): the trained net concentrates regions at ties (near/far 44x) while 10 random
  draws and 2 scrambled controls do not (1.3-6.6x). Near ties it is 2-11x *more* fragmented than random, and far from ties
  4-9x coarser [measured; supported].
- **Is the function simple on the correct part?** At (16,4) the fully correct leaves (72% of the mass) form 78 connected groups,
  and 10 groups hold 90% of that mass. But each group is 1.5k-4k linear pieces. Fully wrong leaves occur only near ties
  (d < .05) or in the blind region, and those strata hold 84% of the leaves, 36% of the mass and 76% of the error [measured].
  **[Corrected 2026-10-18, T263]** The 84% / 36% / 76% are for the d < .02 and blind strata, as in the source table
  (43,176 of 51,552 leaves, mass .359, error 1.62e-3 of 2.13e-3), not for all of d < .05 plus blind.
- **[Corrected 2026-10-18, T263] Scope of all T239 strata.** Each leaf's tie distance is taken at its centre (the
  normalized mean of its rays), and the leaf's whole mass and error are put in that stratum. That is not the mass of
  inputs actually inside each gap band (T254), so "cells per unit mass" and stratum shares above are centre-assigned
  proxies. The per-input check (T269, §5) finds 14% of the target's error at gap >= .05 outside the blind region, and
  19-63% on other models.
Verdict: a large coherent correct part and a thin, fragmented error layer at the tie walls, created by training and growing
with T. It is not random-like: training *moves* regions from the bulk to the ties at about the same total count (T232). The
correct part is coherent only as a union of many linear pieces, so certifying it needs a bound on a nonlinear map. A method that beats enumeration must handle the
fast-growing tie-wall layer without resolving it cell by cell [inferred].

## 8. How far the neuron-level account reaches (T241, added by the DRI 2026-10-18)
Source: `claude/t241-neuron-account@e3d0a0de`, m16/approaches/neuron_account/README.md. Independent review: `claude/t218-tie-tail-labels`, review/neuron_account/README.md. Core confirmed, details qualified.
- **Causal roles.** Every neuron except n9 is individually necessary late. On-manifold conditional resampling at t=10 costs 48-54 pp (comparators), 24-35 (clocks), 27 (n2) and 18-26 (n3/n13/n15), while n4, n5 and n9 matter little at the last step [measured, unverified].
- **Formulas.** Comparators, n4 and n2 fit at held-out R^2 >= .999 from hinge cells on the top-2 ages; clocks need the ages of the top 2-3 (.93-.99) [measured; reproduced by the review within .003].
- **Replacing all 16 at t=10.** Agreement with the network is .955-.966 (implementation-dependent), against a pre-registered .99 target, which is missed. 99% agreement would need 1-R^2 ~1e-5 per neuron [measured, **confirmed** by independent review]. **[Corrected 2026-10-18, T263]** That figure holds *under i.i.d. errors*: i.i.d. Gaussian noise added to every neuron's true pre-activation (T241 README). Real formula errors are correlated and reach .955 agreement at a median 1-R^2 ~3e-3, so it is not a requirement for every kind of formula error. The accuracy match is not robust (+0.11 vs +0.82 pp).
- **Recurrent variant.** Set h_t0 to the formulas, then run the true recurrence: agreement is 1.000 for t0<=2, >=.990 through t0=7, then .983/.978/.955 for t0=8/9/10 [measured, unverified]. The description is faithful except over the last 2-3 steps, where state errors reach the near-tie readout (the blind region degrades first).
- **Where the disagreements are.** About 79-85% sit in the ~20% of mass near ties (gap < .05) or in the blind region [measured, **confirmed**].
- **Reading.** Quantitatively, the mechanism is understood except exactly where accuracy is decided. This matches section 7 (complexity at the tie walls) [inferred].
- **[Corrected 2026-10-18, T263] Fitted, not derived.** The formulas are fitted to sampled inputs of the target (1M
  training inputs), so the account is inductive: its coefficients come from samples, not from the weights. They were
  fitted and tested on the target only. A related fitted surrogate, T269's formula for P(error | gaps), failed to
  transfer: fitted on the target, it is off by +0.4 to +19.5 pp on 5 zoo seeds and 2 noisy copies [measured, unverified;
  INDEX T269]. So the account cannot yet serve as an estimator [inferred].

## 9. The cheap regime (2026-10-18)
DRI, 2026-10-18: the cheap-regime results of T269-T299. Source: the INDEX registry, numbers checked against the child
READMEs. A result is unverified unless marked **verified**; verdicts follow each task's pre-registered rule.

*Terms.* **FPE**: one sampled forward pass, 5,948 FLOPs (T196). **RMS**: root-mean-square error over a panel's models,
in percentage points (pp). **Deductive**: every step is derived from the weights. **Inductive**: some step averages the
network's outputs over sampled or chosen inputs; fixed quadrature nodes count (Directive 8). **Frozen gates**: one ReLU
on/off pattern for a whole class of inputs. **Soft tail**: inputs whose top-3 gaps are all >= .05 (T269).

### 9.1 Why the cheap regime
ARC asks for an estimate that matches random sampling in MSE per unit of computation, over a distribution of models: "a
random seed or maybe adding noise to the weights". It names no budget (Hilton talk, ~Sept 2026, approximate transcript;
[notes](../lit/web_arc/hilton_talk_2026-09_notes.md); a post-January-2026 ARC source, logged in lit/web_log.md). So a
frozen estimator is one (cost, MSE) point, compared with sampling at the same cost [inferred].

Our score (TASKS Directive 8, amendment 2) is the own-cost figure of merit
F = mean(e_m^2) x C_eff / mean(p_m(1-p_m)), where e_m is the error on model m, p_m its accuracy, C the charged cost in
FPE and C_eff = max(C, 1k FPE). Strict: F <= 1; lenient: F <= 10. Equivalently, the estimator wins iff
C_eff <= n* = mean p(1-p) / RMS^2. At p ~ .92 the strict bar allows an RMS of about .85 pp at 1k FPE, .43 at 4k and .21
at 16k. Earlier deductive work aimed at 16k-256k FPE (allowance .05-.2 pp).
The cheap allowance is closer to existing methods' 1-3 pp structural biases.

Two choices are ours, not ARC's: the 1k floor (a constant at the panel mean has RMS ~3 pp, so the principled "beat the
prior" floor is ~80 FPE; Fable r2 A1; this section uses 1k), and the comparator and confidence rule, to be frozen before
any claim (Astra r2 A1, A6).

*Panels.* **LB3** (development): the target, 5 zoo seeds and 24 copies at relative sigma .0025-.02. **DEV2** (T284): 80
open retrains of the zoo config, 1e8-sample truths. **HP1** (T276): 150 sealed models, the target and 5 seeds plus 8
copies of each at absolute sigma .05 and relative .005 and .01, scored only by T281. **HP2** (T284, being built): >= 100
sealed retrains, the primary panel. LB3's seed group predicts DEV2's F within ~20%; its heavy-noise groups mis-rank
methods (T286) [measured].

### 9.2 Results
D = all steps deductive ("approx." = with a declared approximation); I = some step inductive. DEV2 unless marked; costs
as implemented. "Oracle" = the bias left by perfect integration, measured by diagnostic sampling. Every row is
[measured].

| family (task) | class | cost (FPE) | RMS (pp) | F | verdict, status |
|---|---|---|---|---|---|
| best deductive settings (T286): probe v1 / cone_enum / rank_exact Q=8 | D | 3.9k / 1.1k / 25.8k | 3.6-3.8 / 8.4-8.8 / 1.5-1.9 | 76.5 / 119 / 131 | none within 7x of lenient |
| frozen witness, classes D0-D3 (T287) | I (verdict at the witness) | 990-15.8k | 5.5-6.3 | 47.9-829 | NO GO; overestimates by +3.3 to +10.2 pp (LB3) |
| frozen gates + exact last m <= 4 steps (T274b) | D | perfect integration | best m: .44-1.11 per LB3 group | - | dead (**verified**, scoped, T282) |
| 180 root cones, model-free gates (T274b) | D | >= 2.0k | 10.6-10.8 | >= 313 | negative |
| lin low-input closure, oracle (T292) | D approx. | - | 4.77 | - | DEAD by rule; 2 models reproduced independently |
| observer E21 v4 (T292) | D | 123k-880k | 1.18-.63 | 238-476 | G2 (F <= 3) fails |
| R0 point closure, oracle (T292, T296) | D approx. | integral open (T297) | .51; LB3 seeds .61; all LB3 1.39 | - | NO-GO on LB3; retrain-only scope open |
| BLIND-11 (T293) | D | 2.8k | .234 of the blind part (fixed 8/9 rate: .265) | component | uninformative: corr -.30 with the truth |
| mean-field closure, blind part (T294, T294b) | D | 11.9k-392k | .19-.23 of the blind part | component | inconclusive; readout negative (T295); T299 re-check running |
| relaxation bounds (T289, T279) | D (certified) | 632-32k at T = 10 | vacuous: 0/180 sectors decided | - | T289 dead (**verified**, scoped, by T279's code); T279 negative |
| convergence rates (T271, T277) | D and I | 4k-1M | - | - | probe, FORM/SORM alpha ~0: closed; sci_tao (I) .47-.73; H-E11's gain falls through the prefactor, not the rate |
| HP1 held out (T281): rank_exact Q=8 | D | 25.8k | 1.1-1.8 | 19-108 (pooled 32.7) | fails both bars (sealed truths) |
| HP1 held out (T281): v5q / v5a | I | 1k-256k | - | v5q at 4k: .70-1.35 | strict nowhere; lenient v5q 1k-64k, v5a 1k-4k |
| inductive references (T286): v5q / v5a / rank_exact stride 16 / reliability L0 | I | 1.0k / 1.0k / 1.8k / 2.7k | .9 / 1.4-1.5 / 2.5-2.6 / 1.9-2.2 | 1.2 / 3.0 / 16.9 / 17.3 | v5q ~ sampling (it is fixed-node QMC) |

### 9.3 The common failure
Every cheap deductive approximation tried fails for one of four reasons [inferred from the rows above].

1. **Frozen gates erase the model-specific signal.** Frozen at one input per class, every model of this architecture
   gives nearly the same map; models differ through gates switching inside each class. BLIND-11 predicts a spread of
   .06 pp across models against a true .14 (T293). A canonical witness is a typical, well-separated input, so it is
   classified correctly even when its class has many errors, and finer classes do not help (T287). Exact last steps do
   not rescue frozen early ones (T274b, **verified** by T282).
2. **Gaussian state and readout break by steps 7-9.** Margins pile up near 0 at near-ties and have hard edges where
   gates switch, so their law is a mixture. Fed the exact sampled moments, a class-level Gaussian readout is still off
   by -10.95 pp on DEV2, against -.87 for frozen gates (T286's oracle). Per blind unit, the exact Gaussian orthant
   passes 36% of units, and a mean-field state plus exact last m = 0-3 steps 16-30%: the state is already wrong by steps
   7-9 (T295).
3. **Relaxation bounds widen ~50x per step.** Width over margin grows 10.6 -> 514 (forward) and 5.8 -> 229 (CROWN) from
   T = 3 to 4 at H = 4 (T289). A sound CROWN tree at (16,4) leaves .35 of the mass uncertified at 1e5 cones (T279).
4. **Exact branching grows 5-19x per step.** The regions needed to cover all but 1e-5 of the prefix mass grow x15-19 per
   step (H-E13, **verified** for k <= 4); a branch mixture over the last steps grows ~5-10x per step (T295). T287 reaches
   F <= 10 only with 4 exact first and 4 exact last steps: the region problem again. T270 measures the next step.

*Why the same four* [inferred]. H-E15 (**confirmed**, scoped k <= 4, T <= 4): training puts the finest pieces of the
function at the tie walls, 37-44x finer near ties than far, against 1.3-6.6x for random weights. Each approximation
fails on those pieces: frozen gates miss the switching, Gaussians miss the hard edges, relaxations must shrink toward
every tie face (T279), and exact methods must resolve every piece. ANAT-80 (T292) [measured]: weak retrains differ by a
diffuse, model-specific widening of the soft tail (2.48 +- 1.94 pp), which carries 60% of the cross-model error
variance. In the weaker half, half of a model's excess tail error is spread over a median of 44 triples. Gross-error
triples carry 13% of the variance and the blind region 1%. So the model-specific signal is not in a few places, and R0's
bias grows with this tail (T296).

### 9.4 What remains open
- **R0 integral and variants** (T297, T298; running, retrain-only scope). R0 stays viable only if its skeleton integral
  adds <= ~.3 pp at <= ~1.8k FPE (T296); T298 must beat R0's .51 pp by >= 20%.
- **Coarsened shadow network** (T290, running): can rounding and merging shrink the network at <= .4 pp accuracy change?
- **k = 6 complete masses** (T270, running): coverage growth g_1e-7 >= 12 ends the exact route; <= 7 and falling keeps
  it alive, at its own large cost.
- **Codex lanes** (TASKS). T267 (facet cancellation): bounded H4T3/H4T4 phase done, construction costs more than it
  saves, no general negative. T273 (ARC estimator A calibration): blocked on a DRI target definition. T283: an
  independent audit reproduces the 1,009 T271/T277 rows, finds two false CI statements, and notes that the rows
  establish neither equal asymptotic rates nor prefactor-only causation.
- **Untried variants** (from the READMEs): value-dependent gates or exact comparator splits within a class (T286);
  weight-derived witnesses, finer classes (T274b, T287); sub-case conditioning (T293); Gaussian-mixture states (T295);
  alpha-CROWN, multi-neuron relaxations (T279, T289); lin over several anchors (T292).

**No impossibility proof exists.** Every negative above is scoped to the methods and settings tested, as both round-2
reviews say (Fable r2 B, Astra r2 B). Their subjective odds of a strict cheap win on HP2 within three months are 15% and
30% [speculative].

### 9.5 Positive facts worth keeping
- **Deductive errors on retrains are nearly constant offsets** (T286) [measured]. rank_exact is off by -1.39 pp on
  average (76/80 models negative, sd 1.06), probe v1 by +3.28 pp (80/80 positive, sd 1.73), and both correlate .96 with
  the truth. A DEV2-fitted constant would give F ~40 / ~16, but it is advice and not allowed, and the residual spread
  still exceeds the 1k allowance. The methods rank the models; a weight-derived offset is missing [inferred].
- **R0's oracle floor on retrains is .51 pp** (DEV2: mean +.22, slope -.05 pp per pp; LB3 seeds .61; T292, T296)
  [measured]. It was not pre-registered in T292, and T296's retrain-only scope is post hoc. It fits inside the strict 1k
  allowance if the integral is cheap (T297) [inferred]. It fails on noisy copies (1.39 pp on all of LB3).
- **The mean-field closure tracks between-model variation** of the blind-region contribution (corr .73 on DEV2, .80 on
  LB3; frozen gates -.30), but its response is compressed about 2x (predicted sd .049 vs true .139 pp) (T294)
  [measured]. A post hoc calibration fitted on LB3 transfers to DEV2 (RMS .138 pp), so the compression is systematic
  (advice, not counted).
