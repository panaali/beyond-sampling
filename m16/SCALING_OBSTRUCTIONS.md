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
   floor is .51 pp, but its integral needs >= 20-40k FPE (T297), which closes the R0 route in the cheap regime
   [measured, unverified]. Open: R0 variants (T298, T300), the shadow screen (T290), k = 6 masses (T270). No
   impossibility proof exists.
   **[Updated 2026-10-18 09:30]** Superseded in part by T296-T313 and the HP2 scores (§9 headline). The best
   deductive estimator is now R1W V1, at HP2 F 14.92 (official; other counts 13.5-16.8): a lenient fail (T306, T281,
   T313).
   The R0 variants (T298, T300) and the shadow screen (T290) are closed. Strict looks out of reach for every deductive
   design built so far [inferred].
   **[Updated 2026-10-18 12:00 UTC]** Superseded in part by the T317 family (§9 headline). It computes R1W V1's
   estimates at less than half the cost (T317 -> T317e) and is the first deductive entry (with declared
   approximations) to pass the lenient bar with its upper CI <= 10, on two sealed retrain panels: fresh HP3 F
   5.07-5.08 [3.4, 7.0], HP2 6.72-6.76 [4.8, 8.8], at C ~5.4k FPE, RMS .805 pp, bias +.415 pp [measured, unverified;
   T281, H-L1]. It passes only under the project's lenient convention: plain sampling at equal compute has ~5x lower
   MSE, strict fails, and it fails on weight-noised copies (HP1 noisy F 39-206), so it is in scope for retrains only.
   Strict at 5.4k FPE would need RMS <= .37 pp, below the closure oracle floor (~.46 pp). The cost lane T326 found
   the family's floor at ~2.2k FPE (strict F ~1.9 at unchanged RMS; negative for knob cuts, unverified), so strict
   needs an estimate-preserving screen (T330) plus an error fix to ~.58 pp (T324 budget, T327 closure) [inferred; open].
   **[Updated 2026-10-18 14:00 UTC]** Superseded in part by T317f and the lanes T324-T353 (§9 headline). The best
   sampling-free estimator is now T317f, T317e plus a declared frozen-gate margin screen (T330 tier C1; T335): on
   fresh sealed HP4, scored once, F 2.63 [1.93, 3.39] at C 3,302 FPE, RMS .749 pp, bias +.436 pp [measured,
   unverified; T281, H-L1]. It is a lenient pass and a strict fail (the CI excludes 1). Plain sampling at equal
   compute still has ~2.6x lower MSE, and T317f breaks down between .1% and .25% relative weight noise (T337). The
   error budget (T324; product and tmax terms **verified** by T334) puts 66% of the MSE and +.59 pp of bias in the
   walk approximations (the line-set product ~40%, tmax ~20%) and 34% in the closure [measured]. Apart from T330's
   declared screen (T317f itself), no strict lane produced a frozen variant. T326, T330/T330b and T331 are negative on
   cost. T327's spread-lows oracle GO did not survive as an estimator (T339: error sd .667 -> .455, but +.27 pp more **[T343, 14:15: does NOT replicate on fresh DEV3: sd ratio .912 [.76, 1.11] vs DEV2 .68; the variance cut was DEV2-specific]**
   bias). The product fix is unaffordable or not derived (T333, T347), and T351 kept T317f [measured, unverified]. The
   one new DEV2 GO, T353 (spread plus tilt, cost-only, 5,079 FPE, DEV2 F 2.60 vs T317f's 3.14), is a lenient-track
   gain awaiting a pre-registered DEV3 check. Strict at ~5k FPE still needs RMS <= ~.38 pp, against its .609
   [measured, unverified; inferred].

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
**Headline [Updated 2026-10-18 14:00 UTC].** The best sampling-free estimator is now **T317f** (T335): the T317 family's
T317e plus a declared frozen-gate margin screen (T330 tier C1). On the fresh sealed panel HP4 (100 retrains, scored
once) it scores **F 2.63 [1.93, 3.39]** at C 3,302 FPE, RMS .749 pp, bias +.436 pp (T281 84466121) [measured,
unverified]. That is a lenient pass and a **strict FAIL** (the CI excludes 1). Plain sampling at equal compute still
has ~2.6x lower MSE (H-L1), and the family fails on weight-noised copies (T337). So it does not solve the challenge
[measured]. No estimator of any class passes strict (T281) [measured, unverified]. Since 12:00 every strict lane has
reported (strict-bar arithmetic below): only T330's declared screen produced a frozen variant (T317f itself), and
none comes near strict. The one new development result is T353, a cost-only
spread-plus-tilt arm that is GO on DEV2 (F 2.60 vs T317f's 3.14, at 5,079 FPE) and awaits a pre-registered DEV3 check
[measured, unverified; development panel only].
*[12:00 UTC headline, kept; superseded in part]* One deductive estimator (with declared approximations) now passes the
lenient bar with its upper CI <= 10, on two sealed retrain panels: the T317 family, which computes R1W V1's estimates
at less than half V1's cost (T281) [measured, unverified]. It fails strict; plain sampling at equal compute
has about 5x lower MSE; and it fails on weight-noised copies. So it does not solve the challenge [measured]. No
estimator of any class passes strict (T281) [measured, unverified].
- **[Updated 2026-10-18 14:00 UTC] Best sampling-free: T317f** (T335; T218 branch @88453669, r1_cont/estimator_f.py
  -> cont_f.est; §9.2 step 13). The C1 screen stops a walk once the frozen-gate affine margin stays positive to tmax.
  Its declared error against T317e on DEV2 is RMS .077 pp (mean +.045, max .34). Self-contained, no random draws
  (RNG-patched probe clean); C as executed 3,326 FPE on DEV2, where RMS is .827, bias +.489 and F 3.14 (T317e 4.78).
  It was selected among ~6 DEV2 variants (forking-paths note in its README) and submitted once, to HP4 [measured].
- **[Updated 2026-10-18 14:00 UTC] Sealed HP4** (T328 panel: seeds 300-399, mean truth .92265; T281 84466121)
  [measured, unverified]: official C 3,302 FPE, **F 2.63 [1.93, 3.39]**, RMS .749 pp, bias +.436 pp, n* 1,255.
  Sensitivities: screen-charge audit (+1.5% C) 2.67 [1.96, 3.45]; stacked x1.09 x1.04 (C 3,743) 2.98 [2.19, 3.85].
  Secondary, on spent panels: HP3 3.22 [2.19, 4.42], HP2 4.41 [3.21, 5.66]. The HP4 plain-MC control gives .83-1.25,
  with every CI covering 1. HP4 is now spent; HP5 (T336) is the reserve.
- **Best deductive: the T317 family** (claude/t218-tie-tail-labels, r1_cont/README.md; §9.2 steps 8-12). Each step
  is cost-only or a model-free clean-up: T317 (continuity restarts plus an executed B2 prefix tree; estimates equal V1
  to 6.7e-16) -> T317b (G = 20 Gauss-Legendre nodes) -> T317c (telescoping merge of touching correct pieces before
  the Phi profile, model-free Phi tables) -> T317d (model-free tables precomputed once and listed, exact-law
  transverse points; estimates move <= .0115 pp) -> T317e (self-contained, no MC anywhere; estimates and FLOPs
  bitwise identical to T317d). T317e is the version to cite (INDEX T317) [measured].
  *[Updated 2026-10-18 14:00 UTC]* T317f (above) adds one declared approximation to T317e. T317e remains the version
  to cite for the estimates scored on HP2 and HP3.
- **Sealed scores** (T281 scorer) [measured, unverified]. Fresh HP3 (T321; 100 retrains never used before): T317c
  **5.08 [3.42, 7.04]**, T317d **5.07 [3.41, 7.03]** (T317e has T317d's scores), at C ~5.4k FPE, RMS .805 pp, bias
  +.415 pp; sensitivity counts 5.05 and 5.29. HP2: T317c 6.72 [4.84, 8.73] at the official count of the frozen code
  and **6.76 [4.87, 8.79]** as executed (official); T317d 6.75 [4.86, 8.78]. HP1 clean 3.75; HP1 noisy copies 39-206.
  The plain-sampling control gives F ~1 on both panels.
- **What the pass means.** It is a lenient pass under the project's convention (point F <= 10 and upper CI <= 10;
  T281 ruling); strict fails on both panels. The 10x bar is a project convention, not ARC's, whose word is "match".
  Sampling at equal compute has ~5x lower MSE. It fails on weight-noised copies (HP1 noisy F 39-206), so it is in
  scope for retrains only [measured]. H-L1 as restated by the DRI after the red-team (INDEX): "a sampling-free
  estimator (exact piecewise-linear ReLU-gate walks along the two gap directions of each position triple, combined
  under five declared approximations whose validity was established by Monte Carlo on 80 separate development
  retrains) estimates each retrain's accuracy with RMS .80 pp (bias +.41 pp; corr with truth ~.98) at a counted cost
  of ~5.4k FPE (5.3-6.8k under alternative FLOP conventions). Plain sampling at the same compute has about 5x LOWER
  MSE [...]. It does NOT match sampling, and it fails on weight-noised copies." Status: measured on two sealed panels
  (DRI 2026-10-18).
  *[Updated 2026-10-18 14:00 UTC]* H-L1's 12:33 update carries the same reading to T317f on HP4 with ~2.6x in place of
  ~5x: "Plain sampling at equal compute still has ~2.6x lower MSE; strict fails." T324's audit adds that the scheme is
  sampling-free at runtime but not deductive in the strict Hilton sense (below) [measured].
- **Red-team caveats** (review/candidate_redteam/README.md; as INDEX H-L1 and T281 state them) [measured unless
  marked].
  - *Scope:* retrains of the zoo config only; the 10x bar is a project convention, not ARC's ("match"; above).
  - *Selection:* ~30 DEV2 variants and 5 HP2 submissions in the lineage. HP3 scored the frozen T317c once, then
    T317d.
    *[Updated 2026-10-18 14:00 UTC]* T317f: selected among ~6 further DEV2 variants; HP4 scored it once.
  - *Model-free MC:* T317c sets two model-free constants by seeded MC at import; T317d uses exact quadrature (HP3
    5.07), and T317e has no MC anywhere.
  - *Information, not the bar:* a panel-mean constant scores F ~13-14 under the 1k floor (HP3: the constant .920
    scores 18.9 [14.1, 24.2]), so the positive is information content: n* ~1.1k FPE against ~73 for the prior.
  - *Cost latitude:* stacked (x1.09 x1.04 plus T317c's per-call table grid, 807 FPE), HP3 T317c 6.51 [4.38, 9.02]
    and T317d 5.75 [3.86, 7.97] stay candidates; on HP2 T317c is 8.66 [6.23, 11.25] (not a candidate) and T317d 7.65
    [5.51, 9.95] (T281 1515eb10).
  - *Accuracy dependence:* F uses the panel's p(1-p) (mean accuracy ~.92), not the target's (.953). The error falls
    with accuracy (slope -.065 to -.079 pp/pp). T317c's subset F for p > .95 is HP2 1.99 [1.13, 3.01] (n = 23) and
    HP3 4.17 [1.31, 7.67] (n = 22); for p < .90 it is HP2 9.41 and HP3 6.90, with upper CIs above 10 (T317d within
    .03). On the target M_16,10, T317c is off by +.087 pp against m16/truth.txt (.95317945, 2e7 samples) and T317d by
    +.084 pp: one model, anecdotal (T281 1515eb10). Against §1's answer key (.9531175) T317c's error is +.093 pp
    [inferred, arithmetic].
  - *Independence:* the estimates were reproduced by a second implementation (T312, T320), but by the same agent
    both times; a check by a second, different agent is pending (T324). The cost recount agrees within ~9%.
    *[Updated 2026-10-18 14:00 UTC]* Done. T324, a second agent, itemised the error and audited the scheme, and T334
    (own code, pre-registered) reproduced its product and tmax terms; the DRI confirmed those two terms [measured].
  - *Justification [inferred, red-team]:* "deductive" describes the runtime, not the justification. The declared
    approximations were chosen and validated by sampling on DEV2, and no bound is derived from the weights for any
    of them. From T318's .72 pp walk-vs-closure error, roughly 80% of the .8 pp error and the whole bias come from
    them, and the two net biases partly cancel (-.16 from the closure, +.55 from the walk approximations), which is
    fragile. T324 measures this budget.
    *[Updated 2026-10-18 14:00 UTC]* Measured (T324, 80 DEV2): the walk approximations carry 66% of the MSE and
    +.59 pp of bias, the closure c 34% and -.15 pp (§9.2). The audit: sampling-free at runtime, but the scheme
    (closure c, latest-position rule, tmax .5, K = 1 v4 node, blind rule) was selected and validated by MC on DEV2,
    so it is not deductive in the strict Hilton sense [measured].
- **Earlier best deductive: R1W V1** (T304, frozen by T306): the same estimates at 12.1k FPE. HP2 F 16.35 [11.77,
  21.24] as frozen; **14.92 [10.74, 19.38] official** under the written cost convention (T281 recount); 16.83 and
  13.51 [9.72, 17.55] under T313's counts. Every count fails lenient [measured]. Its estimates were reproduced within
  .012 pp on all 80 DEV2 models by a second implementation (T312) [measured]. Earlier on 2026-10-18 (before T306) the
  best deductive HP2 entries were at F 98-107 (rank_exact, probe v1; T281) [measured].
- **Best inductive: fixed-node QMC (v5q)**, about equal to sampling. HP2 F is 1.27 / 1.06 / 1.48 at 1k / 4k / 16k
  FPE, so strict fails narrowly at 4k (T281) [measured].
- **Strict looks out of reach for every built deductive design**, and inductive v5q misses it narrowly (above). The
  limit is a floor, not a budget [inferred]. Strict allows an RMS of about .43 pp at 4k FPE, .37 at 5.4k, .24 at 13k
  and .17 at 26k (§9.1; strict-bar arithmetic below).
  - The closures that have been integrated have oracle floors of .46-.52 pp on DEV2: R0 .515 (T298), R1W's .471
    (T304), .461 with B5 (T307). Each floor alone is about 2-4x the strict allowance at the cost of the deductive
    integrals built for it (R0: 22-47k FPE, T302; R1W: 11-13k, T306, T313) [inferred].
    *[Updated 2026-10-18 12:00 UTC]* The T317 family integrates R1W's closure at ~5.4k FPE, where strict allows
    .37 pp, so that floor is now ~1.25x the allowance rather than ~2x. It still binds [inferred].
  - Better floors cost far more to integrate: .389 needs ~4e5 FPE (T298, T304); .244-.248 (post hoc) needs
    0.3-1.5M FPE, where strict allows .02-.05 pp (T298, T303) [measured cost model, inferred verdict].
  - The integration-error curve does not close the gap. Threshold designs: 2.75 pp at 3.9k FPE, 1.16 at 13k and
    2.96 at 23k (more Newton steps make it worse; T297). Exact walks: .28 pp at 22-47k, where strict allows .13-.19
    (T302) [measured].
  - The information ceiling agrees. A 5-parameter predictor fitted on 40 retrains reaches .477 pp held out, but
    only with rank_exact's 25.8k-FPE output (strict there: .17). From cheap features alone (~6.4k FPE) it reaches
    .72 pp (strict there: .34) (T308) [measured; fitted and inductive, so a diagnostic, not an estimator].
    *[Updated 2026-10-18 12:00 UTC]* Richer fitted functions of cheap per-class features (<= 4k FPE) reach at best
    .855 pp held out, about 2x the ~.43 that strict needs at 4k (T314) [measured, unverified; scoped to those feature
    families and learners].
  - *[Updated 2026-10-18 14:00 UTC]* At T317f's 3.3k FPE strict allows ~.47 pp (at p(1-p) .074), against its HP4
    RMS of .749. At T353's 5,079 FPE it allows ~.38 pp, against .609 on DEV2. The lanes that tried to close this gap
    are in the strict-bar arithmetic below [inferred, arithmetic].
- No impossibility proof exists (§9.4).

*Strict-bar arithmetic [Updated 2026-10-18 14:00 UTC; first written 12:15 UTC].* Taking the panel mean p(1-p) as .074
(TASKS T326), F = RMS^2 x max(C, 1k)/.074, with RMS as a fraction and C in FPE; F is linear in C at fixed RMS. At the
T317 family's RMS of .80 pp, C = 1k FPE would give F ~.87 and C = 2k ~1.7. At the family's ~5.4k FPE, strict needs
RMS <= .37 pp, below the closure oracle floor (~.46 pp; T307), so at today's cost even perfect integration of the
current closure would miss it [inferred]. **T326 (cost), done, negative (unverified):** 80 pre-registered knob
configs (tmax x later-cell cap x line subset) on DEV2; none beats T317e (DEV2 strict F 4.68); the best at C <= 2k is
F 13.2 (742 FPE, RMS 3.13 pp), because dropping a line costs 1.35-3.1 pp and caps 2.3-14.5 pp [measured]. Its cost
table: later-cell tangents 36.5%, B2 prefix tree 31.5%, later-cell readout 10.7%; with all three lines and no later
cells the floor is ~2.2k FPE. Oracle floors: 92-95% of the non-blind later-cell cost comes after the last wrong
piece, so an estimate-preserving cap would reach ~2.2k FPE, i.e. strict F ~1.9 at unchanged RMS [measured, oracle].
So cost can buy ~2.5x, and strict needs both halves. *The 12:15 plan:* **T330** builds a deductive safe-to-tmax
screen (go line: estimates within 1e-4 pp of T317e at C <= 3.0k FPE), and the error side must bring RMS to ~.58 pp at
2.2k, through **T324** (the error budget, attributing the +.41 pp bias, which is about a quarter of the MSE, to
the declared approximations, plus a second-agent audit) and **T327** (closure: re-check of T319 plus two untried
variants; go line oracle RMS <= .30 pp at <= 1.5k FPE added). **T329** (backlog) is a structurally cheaper
geometry. A variant that passes is a new estimate variant, and HP2 and HP3 are spent, so it goes to the reserved
fresh sealed panel HP4 (T328, being built).
*Outcomes [Updated 2026-10-18 14:00 UTC]* (INDEX rows T324-T353; unverified unless marked; §9.2 rows):
- *Cost, rigorous: negative.* T330's Tier A gate box fires on 0 rows. Tier B is exact, but its best schedule costs
  5,525 FPE against T317e's 5,490; T330b (per-gate-pattern operator norms) costs 5,561. Rigorous certificates do not
  pay; zonotopes are untried [measured].
- *Cost, declared: the one gain.* T330's Tier C1 (a declared heuristic) brings C to 3,328 FPE on DEV2 at a declared
  error of RMS .076 pp (DEV2 F 3.14 vs 4.78), short of its tier go line (C <= 2.5k). Frozen as T317f: HP4 F 2.63
  [1.93, 3.39], strict fail [measured].
- *Range: negative.* T331 (tmax raised to 1-2, with or without a zero-cost affine extension, and combined with C1):
  GO = NO. The best MSE x C is 1.08 x T317e (tmax 1 plus extension, RMS .669) [measured].
- *Error budget: done.* T324 puts 66% of the MSE and +.59 pp of bias in the walk approximations (product -.57 pp and
  -40% of the MSE; tmax -.18 and -18%) and 34% and -.15 pp in the closure. T334 reproduces the product and tmax terms
  (**verified**, DRI). With tmax, the blind rule and the product removed, RMS is .583; T332's fitted ceiling from T317e's
  own per-row outputs is .577 [.46, .68]. So what this geometry measures is worth ~.58 pp even when fitted
  [measured; reading inferred].
- *Closure: oracle GO, estimator no.* T327 reproduced T319 (dead, **verified**, scoped) and stopped both untried
  variants on the deployable closure ((a) .553 at +6.1k FPE, (b) .440 at +18k). Its arm B, one-pattern spread lows,
  was GO as an oracle (floor .482 -> .343, reverse pattern .255, no added cost). Built into T317f (T339), it does not
  freeze: s = -1 cuts the error sd .667 -> .455 but moves the estimates up +.27 pp (RMS .884, F 4.04), exposing the **[T343, 14:15: does NOT replicate on fresh DEV3: sd ratio .912 [.76, 1.11] vs DEV2 .68; the variance cut was DEV2-specific]**
  walk approximations' positive bias. Post hoc, its variance alone gives F ~1.06 at 3.7k FPE [measured; post hoc
  figure inferred]. T338 (lows chosen from W equal recency) and T341 (the running-max reading refuted) close the
  obvious mechanism-led alternatives [measured].
- *Product: unaffordable or not derived.* T333: no zero-cost rule works; the exact joint set J2 adds ~21k FPE at the
  line-cell price, ~17x the ~1.2k strict allowance; the net overstatement (+.574 pp) lives in the L-band near both tie
  walls. T347: the overstatement has mean +.568 but sd .275 per model, and the first-principles tilt formula misses its
  go line (mean |error| .230 vs .1). As a post hoc correction it cuts bias +.439 -> -.013 and RMS .796 -> .667 at
  ~1.4k FPE extra [measured]. A no-compute review note on T347 (DRI branch) [inferred]: no model-free derivation of the
  offset exists; its sign is model-free, its magnitude is not, and its constancy is a corner/outside cancellation.
- *Combination: a lenient gain only.* T351 (T317f plus spread plus tilt, four arms) keeps T317f by its pre-registered
  rule; spread plus tilt has the family's best RMS (.609) but costs 6,543 FPE. **T353** reimplements that arm
  cost-only: estimates bitwise equal on 80/80, C 5,079 FPE, DEV2 F 2.60 against T317f's 3.14 (MSE x C 1,883 vs 2,276;
  go line 2,048). It is GO on DEV2 and frozen; its pre-registered, no-retune DEV3 check against T317f is **open**, and
  only a confirmed check makes it the HP5 candidate [measured, unverified].
So at ~5k FPE strict needs RMS <= ~.38 pp (T351), against the best DEV2 arm's .609 (bias +.413, sd .451). Even with the
bias removed exactly, an sd of .451 at 5,079 FPE gives F ~1.4 [inferred, arithmetic]. Strict now needs a deductive
removal of the walk approximations' positive bias (T339's reading) and a further cut in the model-to-model spread.
T329 (a cheaper geometry) is still in the backlog.

DRI, 2026-10-18: the cheap-regime results of T269-T300. Source: the INDEX registry, numbers checked against the child
READMEs. A result is unverified unless marked **verified**; verdicts follow each task's pre-registered rule.
**[Updated 2026-10-18 09:30]** Now also T296-T313 and T281's HP2 scores, including the cost-convention recount and
the T313 recount (INDEX T281 row). T313's own result is cited from its branch. Fable round 3: pending publication.
**[Updated 2026-10-18 12:00 UTC]** Now also the T317 family (T317, T317b-e; T322, T325), T318-T321, the red-team
(review/candidate_redteam/README.md) with T281's follow-ups, and T309, T314, T315 (INDEX rows T309-T321, T281, H-L1;
TASKS rows T323-T327). T317e's scorer identity check is cited from r1_cont/README.md.

*Terms.* **FPE**: one sampled forward pass, 5,948 FLOPs (T196). **RMS**: root-mean-square error over a panel's models,
in percentage points (pp). **Deductive**: every step is derived from the weights. **Inductive**: some step averages the
network's outputs over sampled or chosen inputs; fixed quadrature nodes count (Directive 8). **Frozen gates**: one ReLU
on/off pattern for a whole class of inputs. **Soft tail**: inputs whose top-3 gaps are all >= .05 (T269).
*[Updated 2026-10-18 09:30]* **R0**: the point closure that puts the low inputs at their conditional mean given the
top-3 skeleton (T292). **R1W V1**: R0's integral by exact segment walks along the gap lines (T302), with v4 at the
triple's latest free position (one Gauss node) and the blind-region g12 walk dropped (T304); frozen by T306
(r1_walk/estimator.py @684a22b9, DEDUCTIVE=True). **B2**: exact sharing of the first-cell evaluations across the 720
triples (T307, T313). **B5**: x9 given by its exact conditional law instead of a point (T307).
*[Updated 2026-10-18 12:00 UTC]* **Continuity restart**: at a gate flip only the tangents of the later steps are
recomputed, since the values are continuous (T317, Fable r4 D1). **Telescoping merge**: touching correct pieces of
consecutive cells are merged before the Phi profile, since their inner endpoints cancel exactly (T317c).
**Candidate**: a point F <= 10 with the upper 95% CI <= 10 on a sealed panel (T281 ruling).

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

Two choices are ours, not ARC's: the 1k floor (a constant at the panel mean has RMS ~3 pp, so the principled "beat the
prior" floor is ~80 FPE; Fable r2 A1; this section uses 1k), and the comparator and confidence rule, to be frozen before
any claim (Astra r2 A1, A6).

*Cost convention [Updated 2026-10-18 09:30]* (TASKS Directive 8, written out after T312). Charge every operation
whose inputs depend on the weights, at T196 rates (multiply-add = 2). That includes special functions at
weight-dependent points (Phi, Phi2, exp, log: 20 FLOPs each unless instrumented). Do not charge functions of the input
law alone (order-statistic densities at fixed nodes, Gauss-Legendre nodes and weights, model-free cell masses), if
they are precomputed once, listed with their size, and the same for every model. Charge the estimator's own forward
passes at their instrumented FLOPs (5,760 for one M_16,10 forward, T281); the sampling side stays at 5,948 FLOPs per
FPE. Under this rule the T281 scorer lowers R1W V1's HP2 cost from 13.3k to 12.1k FPE, and T300's from 1,697 to
1,551 (claude/hp1-scorer@78a722c9, hp1_scorer/README.md, cost_audit.py) [measured, unverified].

*Panels.* **LB3** (development): the target, 5 zoo seeds and 24 copies at relative sigma .0025-.02. **DEV2** (T284): 80
open retrains of the zoo config, 1e8-sample truths. **HP1** (T276): 150 sealed models, the target and 5 seeds plus 8
copies of each at absolute sigma .05 and relative .005 and .01, scored only by T281. **HP2** (T284, being built): >= 100
sealed retrains, the primary panel. LB3's seed group predicts DEV2's F within ~20%; its heavy-noise groups mis-rank
methods (T286) [measured].
*[Updated 2026-10-18 09:30]* HP2 is built and scored: 100 sealed retrains (seeds 100-199), one cluster per retrain.
The plain-sampling control gives F .95 / 1.00 / .79 / .99 / 1.06 at 1k / 4k / 16k / 64k / 256k FPE (T281,
claude/hp1-scorer@8917f0a9), so the harness is sound [measured].
*[Updated 2026-10-18 12:00 UTC]* **HP3** (T321): 100 fresh sealed retrains (seeds 200-299, data ids disjoint, never
used before), truths held by the T281 scorer only. It had exactly one pre-registered submission (T317c @1c742430, with
later cost-only recounts), then T317d as a second submission by DRI ruling. Plain-MC control ~1 (T281 67e57f00).
HP2 and HP3 are now spent for new variants (TASKS T326) [measured].

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
| R0 point closure, oracle (T292, T296) | D approx. | integral: next row | .51; LB3 seeds .61; all LB3 1.39 | - | NO-GO on LB3; retrain-only scope open. *[Updated 2026-10-18 09:30]* The retrain-only route was then built out by T297-T313 (rows added below) |
| R0 skeleton integral, threshold designs (T297) | D | 3.9k-23k | integration error 2.75 at 3.9k, >= 1.16 up to 23k | - | DEAD for threshold designs |
| BLIND-11 (T293) | D | 2.8k | .234 of the blind part (fixed 8/9 rate: .265) | component | uninformative: corr -.30 with the truth |
| mean-field closure, blind part (T294, T294b) | D | 11.9k-392k | .19-.23 of the blind part | component | inconclusive; readout negative (T295); T299 re-check running. *[Updated 2026-10-18 09:30]* T295 dead (**verified**, scoped, by T299's own code): Gaussian-state closures with an orthant readout on the blind units, up to 4 exact last steps. Open: Gaussian-mixture states, skew readouts |
| relaxation bounds (T289, T279) | D (certified) | 632-32k at T = 10 | vacuous: 0/180 sectors decided | - | T289 dead (**verified**, scoped, by T279's code); T279 negative. *[Updated 2026-10-18 09:30]* Unchanged: T279 itself stays negative (unverified); its T = 10 sector numbers match T289's independently. Scope: single-neuron linear relaxations on whole sectors |
| convergence rates (T271, T277) | D and I | 4k-1M | - | - | probe, FORM/SORM alpha ~0: closed; sci_tao (I) .47-.73; H-E11's gain falls through the prefactor, not the rate |
| HP1 held out (T281): rank_exact Q=8 | D | 25.8k | 1.1-1.8 | 19-108 (pooled 32.7) | fails both bars (sealed truths) |
| HP1 held out (T281): v5q / v5a | I | 1k-256k | - | v5q at 4k: .70-1.35 | strict nowhere; lenient v5q 1k-64k, v5a 1k-4k |
| inductive references (T286): v5q / v5a / rank_exact stride 16 / reliability L0 | I | 1.0k / 1.0k / 1.8k / 2.7k | .9 / 1.4-1.5 / 2.5-2.6 / 1.9-2.2 | 1.2 / 3.0 / 16.9 / 17.3 | v5q ~ sampling (it is fixed-node QMC) |

*Rows added [Updated 2026-10-18 09:30].* Same columns. HP2 is the sealed claim panel; DEV2 is development. Each row
carries its own tag; all are unverified unless marked.

| family (task) | class | cost (FPE) | RMS (pp) | F | verdict, status |
|---|---|---|---|---|---|
| **R0 lineage, step 1:** R0 on all 30 LB3 models (T296) | D approx. (oracle) | - | 1.39; group means +.04 (target) rising to +2.51 (sigma .02); retrain-only (post hoc): DEV2 .51, LB3 seeds .61 | - | [measured] NO-GO by rule: R0 overestimates, and its bias grows with the noise level and with the soft-tail error (DEV2 corr .37) |
| **step 2:** R0 threshold integral (T297) | D | 3.9k / 13k / 23k | integration error vs a 1e7-sample MC of R0: 2.75 / 1.16 / 2.96 | - | [measured] DEAD for threshold designs; [inferred] a passing design must resolve the boundary in 3-D per triple, >= 20-40k FPE |
| **step 3:** cheap R0 design, stage 0 (T300) | I (Newton extrapolation within a triple) | 1,752 (DEV2); HP2 1,697, or 1,551 under the written convention | DEV2 1.57; HP2 2.13 (bias +.82) | DEV2 5.97 [4.28, 7.78]; HP2 9.72 [7.0, 13.0], or 8.89 [6.42, 11.86] recounted | [measured] HP2 point pass, but the upper CI is above 10: not a candidate. Part of its accuracy is cancellation: adding the omitted tau12 and blind classes back gives F ~90. Stage 1 (boundaries shared across the max's position) stopped: 1.40-1.87 pp error |
| **step 4:** R0 by exact segment walks (T302) | D (declared transverse point) | 22-47k (DEV2 ~26k) | integration .28-.29 vs MC of R0, no cancellation; total .57-.66 | DEV2 20.6 [15.0, 27.3] | [measured] FAIL by its line (lenient at <= 4k). The first cells alone cost 5.8k: the walk crosses ~7 gate cells in the first .25 of gap (H-E15 as cost) |
| **step 5:** R1W V1 on DEV2 (T304, T306) | D (declared approximations) | 13.5k | .795 (bias +.45) | 11.7 [8.5, 15.4] (T304); 11.95 under T306's selection rule | [measured] narrow FAIL (development). T306's two pre-registered refinements were worse (F 13.7-44.1). Its closure floor is .471 (T304) |
| **step 5, sealed:** R1W V1 on HP2 (T306, scored by T281) | D | 13.3k; 12.1k under the written convention | .99 (bias +.53) | 16.35 [11.77, 21.24]; 14.92 [10.74, 19.38] recounted (official); HP1 clean 9.1 (8.30 recounted), HP1 noisy 93-425 | [measured] FAILS lenient; the best deductive entry on HP2. *[Updated 2026-10-18 12:00 UTC]* Superseded: the same estimates at ~5.4k FPE pass lenient (steps 8-12 below) |
| **step 6:** R1W V1 reproduced (T312) | D | - | estimates agree within .012 pp on all 80 DEV2 models (RMS difference .004) | cost over-charged: DEV2 point F 11.58 -> 9.63, HP2 ~13.6 [inferred] | [measured] reproduced by a second agent with its own code; not yet marked verified in the INDEX. The frozen code charged 2.1e7 FLOPs of model-free Phi/GL work (the reproduction counts 7.5e6) and its forwards at 5,948 FLOPs instead of 5,760. This led to the written cost convention (§9.1). DEV2 is development; the claim is HP2 |
| **step 7:** R1W V1-B2 recount (T313; claude/t218-tie-tail-labels@daeebf9a, r1_v1b2/README.md; HP2 recount by T281, claude/hp1-scorer@7dea5111) | D | DEV2 10,971 (6.53e7 FLOPs, -17% vs frozen V1) | estimates identical to V1 (max difference 0 on all 80 DEV2) | HP2 16.83 (instrumented, no B2); 13.51 [9.72, 17.55] with B2 | [measured] still a lenient fail: at RMS .99, lenient needs C <= ~8.1k. DRI ruling: B2 counts only once re-frozen with the sharing performed; the official HP2 F stays 14.92. *[Updated 2026-10-18 12:00 UTC]* T317 executes B2 as a prefix tree, so it now counts (step 8) |
| R0-family oracle floors (T298) | D approx. (oracle) | - | R0 .515; top-4 exact + 6 lows at their conditional mean .389 (pre-registered PASS, 24% better); post hoc top-4 + spread lows .244 | - | [measured] better floors add a skeleton dimension, so they do not rescue the cheap integral [inferred] |
| re-check of the post hoc closure (T303) | D approx. (oracle) | one verdict per class 10,080; boundary resolution 0.3-1.5M | DEV2 .248; LB3 .386 | - | [measured] reproduced within 3% (same author, fresh code and draws; unverified by a second agent). A per-class integral cannot fit the cheap regime [inferred] |
| class sharing (T305) | D (map of the error mass) | - | closest misses: 8x fewer classes at .69 (R0), 7x at .313 (top-4 + spread lows); line: >= 4x at <= .25 | - | [measured] negative: no scheme passes. Main effects carry .18 of the variance, main + pairwise .46-.51: the class structure is high-order |
| B5 and B2 on R1W (T307) | D | V1 + B2 ~10.7k (projected) | B5 floor .503 -> .461 (8.2%; stop line 10%) | projected HP2 ~13.2 [inferred] | [measured] stopped at the pre-registered B5 check. V1's excess error comes from the walk-level declared approximations, not v4's point node |
| adaptive skipping of walks (T311) | D | a free, perfect certificate saves <= 3.35k of 13.7k; with B2, ~7.8k at best vs the 8.1k target | - | - | [inferred, from measured line counts] killed by arithmetic before building. The weight-derived Lipschitz bound is 2.3e8-1.2e9 per unit length (useless); a CROWN segment certificate costs 2-3 FPE per line, more than skipping saves (~1.6) |
| information ceiling (T308) | I (fitted on 40 retrains; diagnostic) | rank_exact 25.8k; cheap features ~6.4k | held out .477 with rank_exact; .72 cheap only (post hoc); probe v1 alone .85; the two full estimators together .631 | - | [measured] GO with a scope caveat: the model-specific information exists, but the cheap regime is not shown to reach .6 pp. Not an estimator |
| training stationarity (T310) | D (identities) | - | - | - | [measured] STOP by rule: the retrains are not readout-stationary (residual median .41 vs stop line .10, 11-60x its SE) |
| coarsened shadow network (T290) | D | - | - | - | [measured] KILL: over 80 DEV2 models and 16 rounding settings, no model both shrinks and stays within +-.4 pp. Rounding W_hh alone takes fused_s0 from 87% to 40% |
| future-aware mixture propagation (T309) | D (diagnostic) | - | - | - | running; partial results on claude/t292-oracle-floor, not yet registered. *[Updated 2026-10-18 12:00 UTC]* Done (f4b5b76c, mix_sketch/README.md): [measured] STOP by the pre-registered rule. A generous oracle (exact step-8 moments, exact push-forward through steps 8-9, 5 merge rules, K = 1..128) on 5 DEV2 models: non-blind classes need median K* > 128 (1/25 pass at K <= 50), and the error of the error rate falls only ~K^-0.2. Dead in the cheap regime (scoped): >= 36k components, >= 1e5 FPE. Untried: EM or truncated-Gaussian mixtures, a step-9 representation |
| HP2 control (T281) | sampling | 1k-256k | - | .79-1.06 | [measured] harness sound |
| HP2 inductive references (T281): v5q / v5a | I | 1k / 4k / 16k / 64k / 256k | v5q 1.00 / .46 / .27 / .21 / .18 | v5q 1.27 / 1.06 / 1.48 / 3.53 / 10.5; v5a 2.9 / 5.3 / 14.1 / 53 / 219 | [measured] v5q is a lenient candidate at 1k-64k (upper CI <= 10) and about equal to sampling; strict fails narrowly at 4k. v5a is lenient at 1k-4k only |
| HP2 deductive references (T281): rank_exact Q=8 / probe v1 (post hoc) | D | 25.8k / 3.9k | 1.74 / 4.66 | 98 / 107 | [measured] fail both bars |

*Rows added [Updated 2026-10-18 12:00 UTC].* Same columns. Steps 8-12 continue the R0 lineage; each is cost-only or a
model-free clean-up of R1W V1, so the estimates stay V1's (to <= .0115 pp). Sources: claude/t218-tie-tail-labels,
r1_cont/README.md (pre-registered); sealed scores by T281 (claude/hp1-scorer, commits as listed). HP3 is the fresh
sealed panel (T321). All are unverified unless marked.

| family (task) | class | cost (FPE) | RMS (pp) | F | verdict, status |
|---|---|---|---|---|---|
| **step 8:** R1W V1 by continuity restarts plus an executed B2 prefix tree, G = 48 (T317; f5998902; Fable r4 D1) | D (declared approximations) | DEV2 7,775; HP2 7,765, or 7,880 official (correct_interval 229) | estimates = V1 (max diff 6.7e-16 on 80 DEV2 and on HP2); DEV2 .795; HP2 .99 (bias +.53) | DEV2 6.78; HP2 9.57 [6.89, 12.43], 9.71 [6.99, 12.61] official (T281 bbf321c8, 379498b6) | [measured] the first deductive point lenient pass on HP2, but the upper CI is above 10: not a candidate. G = 16 rejected (moves the estimate 2.5e-4 pp > 1e-4). HP1 clean 5.3; noisy 52-228 (out of scope) |
| **step 9:** minimal Gauss-Legendre G = 20 (T317b; 0f933fba, estimator_b.py) | D | DEV2 6,293; HP2 6,290, or 6,405 official | estimates within 1.4e-6 pp of V1 | HP2 7.75 [5.58, 10.07], 7.89 [5.68, 10.25] official (T281 d956eee4, 379498b6) | [measured] point lenient pass; the upper CI misses 10 (10.05-10.16 over 20 bootstrap seeds): not a candidate under the fixed rule. Ruling: later submissions must reproduce T317b's estimates to <= 1e-12 (cost-only), or they count as new variants. HP1 clean 4.3 |
| **step 10:** telescoping merge of touching correct pieces before the Phi profile, plus model-free Phi tables at a = 0 (T317c; 1c742430, estimator_c.py; c70e22cd masks before ndtr) | D | DEV2 5,435 (-15%), 5,488 as executed; HP2 5,455 official, 5,489 as executed, 5,713 with every executed ndtr charged | estimates = T317b (max diff 0.0 on DEV2, <= 2.2e-16 on HP2); HP2 .99 (bias +.53), as V1 | HP2 6.72 [4.84, 8.73]; **6.76 [4.87, 8.79] as executed (official)**; 7.04 [5.07, 9.14] with every ndtr (T281 9b4e41df, 7dd19ff5) | [measured] LENIENT PASS WITH UPPER CI <= 10 on HP2: the first deductive entry to meet the candidate condition. Strict fails. HP1 clean 3.75; noisy copies 39-206 (out of scope). The third HP2 scoring of identical estimates (cost-only, pre-registered); the savings were conceived after T317b's near miss (red-team, issue 6) |
| **step 10, fresh sealed:** T317c on HP3 (T321; T281 67e57f00) | D | 5,427 (c70e22cd as executed) | .805 (bias +.415) | **5.08 [3.42, 7.04]**; sensitivity counts 5.05 (1c742430, official count) and 5.29 [3.56, 7.33] (every ndtr) | [measured] lenient PASS with upper CI <= 10 under all counts; strict FAIL. Plain-MC control ~1. HP3's one pre-registered submission |
| **step 11:** "ARC-ready" version: model-free tables precomputed once and listed (266 scalars plus 43,200 input/direction entries), exact-law transverse points (T317d, T322; 5b4d4806, estimator_d.py) | D | DEV2 5,490; HP3 5,431 | within .0115 pp of T317c on DEV2 (.0076 on HP3, .0068 on HP2); DEV2 .794 (bias +.444) | DEV2 4.78; HP3 **5.07 [3.41, 7.03]**; HP2 6.75 [4.86, 8.78] (T281 29bef882) | [measured] closes T320's three convention gaps. Its DEV2 gate (<= .01 pp) failed narrowly (.0115 on one of 80 models); the DRI accepted it as a model-free change fixed in advance. HP3's second submission |
| **step 12:** final clean-up, self-contained, no MC (T317e, T325; 01906d33, r1_cont/estimator_e.py -> cont_e.est) | D | DEV2 5,490 | estimates and FLOPs bitwise identical to T317d (0.0 / 0.0 on 80 DEV2) | = T317d's (no new sealed run) | [measured] imports only numpy, scipy.special.ndtr and itertools; the dead walk/r1w2 imports that built seeded MC tables at import are gone; est runs with numpy's random generators patched to raise. The scorer found estimates and FLOPs bitwise identical on all 100 HP2 and 100 HP3 models (claude/hp1-scorer e8102588). **The version to cite** |
| independent confirmation of T317c (T320; claude/t241-neuron-account 2fdb7582, t317c_confirm/, pre-registered 1189f6e7) | D (audit and recount) | own executed count 5,740 vs 5,271 (+8.9%) | own estimates within .0077 pp of estimator_c on 10 DEV2 | at +9% cost: HP2 F ~7.3, upper CI ~9.5 [inferred] | [measured] estimates PASS. The pre-registered 5% cost band FAILED for a benign reason (its root search also tests the crossing direction); executing the frozen rule (post hoc v2) gives +0.1% with identical estimates, so implementation latitude is ~9%. Audit: no undeclared modelling step; three convention gaps (tables recomputed inside est, not listed, transverse points from seeded model-free MC), closed by T317d. Same agent as T312; a second, different agent is pending (T324) |
| red-team follow-ups (T281 1515eb10; review/candidate_redteam/README.md) | D | stacked latitude x1.09 x1.04, plus 807 FPE of per-call tables for T317c: 6.2-7.0k | target M_16,10: +.087 (T317c), +.084 (T317d); p > .95 subsets .37 (HP2), .53 (HP3) | stacked: HP3 T317c 6.51 [4.38, 9.02], T317d 5.75 [3.86, 7.97]; HP2 T317c 8.66 [6.23, 11.25], T317d 7.65 [5.51, 9.95]. HP3 constant .920: 18.9 [14.1, 24.2] | [measured] under the stacked count T317c stays a candidate on HP3 but not on HP2; T317d stays one on both. n* is 1,069 FPE for T317c against 53 for the constant. Subset and target details in the §9 headline |
| kinked-tangent surrogate (T318; claude/t241-neuron-account e6e3fd51; Fable r4 D2) | D | no saving as implemented | integration error vs a closure-c MC (10 DEV2): SURR 13.3 / CAP1 10.4 / CAP2 6.0, against .72 (+.55 mean) for the exact walk | - | [measured] STOP. It equals the exact walk to 3e-17 on single-cell lines, but early flips shift later events by O(1). Side finding: V1's declared approximations cost ~.7 pp |
| Gaussian-smoothed closure verdict (T319; claude/t292-oracle-floor 91b712b6; Fable r4 D3) | D approx. (oracle) | - | R0 hard .527 / smooth 1.326; top-4 hard .399 / smooth .657 (go line: ratio <= .70) | - | [measured] STOP for both closures. The lows switch gates (only 4-12% of draws stay in the anchor's cone), so the first-order band is too wide. Untried: cone-clipped J, an exact 1-D treatment of the largest-|J| low (now T327) |
| capped walks (T315; claude/t218-tie-tail-labels 7936af31) | D | c = 1 / 2 / 3 exact cells: 4.3k / 7.3k / 9.8k | bias -10.4 (c = 1), -5.3 (c = 3) | 1269 / 1678 / 903 | [measured] FAIL, far from the line. Near the tie the first gate cells are tiny (H-E15); there is no cheap middle between first-cell extrapolation and full walks |
| cheap-feature information ceiling, richer learners (T314; claude/t241-neuron-account 5067f9d9) | I (fitted on 40 retrains; diagnostic) | features <= 4k (2,417) | held out: best .855 (per-class basis ridge); with probe v1 (~6.3k) .72 | - | [measured] INFORMATION-INFEASIBLE for cheap strict, scoped to these anchor-feature families and learners: ~2x the ~.43 pp strict needs at 4k |

*Rows added [Updated 2026-10-18 14:00 UTC].* Same columns. Step 13 adds one declared approximation to T317e (the C1
screen) and was scored on the fresh sealed panel HP4 (T328: 100 retrains, seeds 300-399). Step 14 is DEV2 only. The
other rows are the strict-bar lanes (cost, error budget, product, closure), the noise curve and the rare-event pair.
Sources: INDEX rows T317 (for T317f/T335), T324 (for T334 and addendum A) and T326-T353; branches as listed there,
named here by their task suffix; sealed scores by T281. DEV2 unless marked; all unverified unless marked.

| family (task) | class | cost (FPE) | RMS (pp) | F | verdict, status |
|---|---|---|---|---|---|
| **step 13:** T317e plus a declared frozen-gate margin screen, T330 tier C1 (T317f, T335; T218 branch @88453669, r1_cont/estimator_f.py -> cont_f.est) | D (declared approximations; the screen is one more) | DEV2 3,326 as executed | DEV2 .827 (bias +.489); declared error vs T317e RMS .077 (mean +.045, max .34) | DEV2 3.14 (T317e 4.78) | [measured] self-contained, no random draws (RNG-patched probe clean); selected among ~6 DEV2 variants (forking-paths note in its README). HP4's single pre-registered submission |
| **step 13, fresh sealed:** T317f on HP4 (T281 84466121, scored once) | D | 3,302 official; +1.5% with the screen-charge audit; 3,743 stacked (x1.09 x1.04) | .749 (bias +.436); n* 1,255 | **2.63 [1.93, 3.39]**; audit 2.67 [1.96, 3.45]; stacked 2.98 [2.19, 3.85]; spent panels (secondary) HP3 3.22 [2.19, 4.42], HP2 4.41 [3.21, 5.66] | [measured] LENIENT PASS, strict FAIL (the CI excludes 1). HP4 plain-MC control .83-1.25, all CIs cover 1. Sampling at equal compute still has ~2.6x lower MSE (H-L1). HP4 is now spent |
| cost-error frontier of the T317e family (T326; t218-tie-tail-labels 7f1259a9, cost table 50340c1b) | D | 80 pre-registered configs; best at <= 2k: 742 | 3.13 at 742 FPE; caps cost 2.3-14.5, dropping a line 1.35-3.1, tmax .3 1.03 | 13.2 at 742; none beats T317e (DEV2 strict F 4.68) | [measured] negative (unverified), scoped to this family's knobs. Cost table: later-cell tangents 36.5%, B2 prefix tree 31.5%, later-cell readout 10.7%. Structural floor ~2.2k FPE with all three lines and no later cells; an oracle cap keeps T317e's estimate at ~2.2k (strict F ~1.9 at unchanged RMS) |
| rigorous safe-to-tmax screens (T330 tiers A and B, pre-registered 7f3b0fa0; T330b per-gate-pattern operator norms) | D (certified) | best C: Tier B 5,525, T330b 5,561, vs T317e 5,490 | exact (max \|d\| 0.0 pp) | - | [measured] negative (unverified). Tier A fires on 0 rows (beta ~1e8). Tier B stops 85-95% of non-blind rows eventually, but each try costs ~one later cell (~3.4k FLOPs) and succeeds ~25%; the norms skip only 0.3-5% more. The binding looseness is the free gates' full \|dz*\| at Delta .1-.5, not box vs norm. Zonotopes untried |
| declared screens (T330 tier C, pre-registered e8e0c530) | D (declared heuristic) | C1 all lines 3,328; non-blind only 3,522 | declared error: C1 RMS .076 (mean +.045, max .34; one-signed +); C2 .018 | C1 3.14; non-blind only 3.32; C2 4.10 | [measured] C1 fires on 95.5-96.7% of post-last-wrong cells: the family's best DEV2 cost point. Tier-C go (C <= 2.5k) not met. Frozen as step 13 |
| tmax upward (T331; review round 5, D1; pre-registered e8e0c530, e59484d6) | D (declared approximations) | tmax 1 +52% (the review expected +5-12%); tmax 2 +126%; tmax 1 + extension +53%; plus C1 3,854 | tmax 1 .731 (bias +.30); tmax .5 + extension bias -.40; tmax 1 + extension .669 (bias +.10); plus C1 1.034 (bias -.708) | MSE x C vs T317e: 1.28 / 1.11 / 1.08 / 1.19 (go <= .70 for the combination) | [measured] GO = NO (unverified). A frozen early cell is a good proxy inside the window, not beyond it: C1-stopped rows extend an early cell's margin past tmax and mark far-gap mass wrong |
| information ceiling from T317e's own per-row outputs (T332; t241-neuron-account 282d43d4; review round 5 settling test) | I (fitted; diagnostic) | at the 2.2k oracle cost / at 5.4k | held out .577 [.46, .68] (offset only .646; raw T317e .817; GBT .611) | ridge 1.01 at 2.2k, 2.49 at 5.4k | [measured] NEITHER go nor stop. [inferred] What this geometry measures is worth ~.58 pp even when fitted, matching T324's path with tmax, blind and product removed (.583); going lower needs new measurements, not re-weighting |
| per-approximation error budget of T317d (T324; t292-oracle-floor, budget_t317d/README.md, pre-registered e85fff85) and independent re-check (T334; own code, pre-registered 9ee95374) | D (audit; decomposition on exact-law inputs) | - | one at a time (Delta bias, Delta MSE): product -.57 / -40%; tmax -.18 / -18%; blind g12 -.07 / -9%; mu-lin +.02 / +2%; single transverse point +.27 / +25%. Cumulative .720 -> .685 -> .691 -> .583 -> .475 -> .463 (K = 1 v4 node off) | - | [measured] walk approximations 66% of the MSE and +.59 pp bias; closure c 34% and -.15 pp. Product and tmax terms **verified** (DRI; T334: product -.562 [-.623, -.502], -41% [-57%, -20%]; tmax -.183 [-.235, -.136], -20% [-33%, -1%]); the product's MSE share varies across models (20-57%). The K = 1 v4 node is a sixth declared approximation (bias +-.31 pp). Audit: sampling-free at runtime, scheme selected and validated by MC on DEV2 |
| how to combine the two lines, oracle (T324 addendum A; review round 5, D3; pre-registered f37e8bfc; 10 DEV2 x 1e6) | D approx. (oracle) | - | residue vs closure c, model-level: product at one point .385; OR 6.84, L23 2.24, L12 4.65, AVG 3.41; product at half points (T306-B) .495; exact 2-D at one point .377; exact 2-D at half points (J2) .087 | - | [measured] STOP: only the exact 2-D set works. T306-B explained as cancellation: the product overstates by +.30-.34 pp and the single anchor understates by -.34; they cancel at one anchor (per class the residue is 4.9 pp), so fixing either alone makes it worse |
| replacing the product (T333, addenda B1-B3; 10 DEV2) | D | J2 lower bound ~21k extra at the line-cell price (~155k at an honest 2-D price; ~49k with blind in 2-D); corner-only J1 +2.6k / 4.0k / 5.6k | J2 vs truth .583 (.470 with no cut and blind walked); J1 removes .08 / .28 / .53 pp | - | [measured] B1: no zero-cost rule works (ADD -65x, OR -182x, L23 -33x, L12 -81x). B2: J2's extra alone is ~17x the ~1.2k strict allowance, so strict is out of reach for closure c via J2 by arithmetic. B3: net overstatement +.574 pp (corr with 1-p .001; gross over-mass corr .62), in the L-band near both tie walls (66% / 91% / 101% of the net at delta .05 / .1 / .2), not the corner (14% / 49% / 92%). J1 go (>= .25 at <= +600) not met |
| first-principles overstatement (T347; formula frozen 790de13d, t347/tilt.py) | D approx. | ~1.4k extra (4.3k crudely counted) | D = V1 - J1: mean +.568, sd .275 per model (range -.04 to 1.43; corr with 1-p .07); formula mean \|error\| .230 (go .1), corr .54; post hoc V1 - D_pred: bias +.439 -> -.013, RMS .796 -> .667 | - | [measured] NO GO on the pre-registered rule. T333's "model-constant" only means it does not track accuracy; the best constant leaves mean \|error\| .207. The post hoc correction takes ~70% of the oracle product-removal gain (J1 -.129 / .616). A no-compute review note (DRI branch) [inferred]: no model-free derivation of the offset exists; the sign is model-free, the magnitude is not, and the constancy is a corner/outside cancellation |
| closure re-check and untried variants (T327; t286-cheap-frontier ca5c94d3, closure_floor/README.md; pre-registered ccdda646, addendum a61743c7; review round 5, arm B) | D approx. (oracle) | (a) +6.1k; (b) +18k; arm B no added cost | T319 reproduced: R0 hard .532 / smooth 1.329 / product 11.69; top-4 .416 / .662 / 5.87. Deployable R1W: hard .481, (a) cone-clipped J .553, (b) exact 1-D walk on the largest-\|J\| low .440. Arm B (mu + (-1)^j sigma): .482 -> .343 (bias -.15 -> -.04); reverse pattern .255 | projected MSE x C vs T317e: (a) 2.4x, (b) 4.0x; arm B .82x (bar .85), reverse .74x | [measured] T319 dead (**verified**, scoped, DRI); (a) and (b) STOP; arm B GO as an oracle projection by quadrature addition. The gain comes from breaking the lows' equality. R0 gets worse with the spread (.515 -> .62 / .58) |
| spread-lows closure in the estimator (T317g attempt, T339; t286-cheap-frontier 886e31ac, cont_g.py; pre-registered 6e0c7533) | D approx. | s = +1 4,090; s = -1 3,748 (+12-23% over T317f) | s = +1 .911 (bias +.726, sd .550); s = -1 .884 (bias +.758, sd .455); T317f .827 (+.489, sd .667) | 4.68 / 4.04 vs T317f 3.14 | [measured] NO FREEZE by the pre-registered rule; s = 0 reproduces T317f bitwise. The spread cuts the model-to-model error sd .667 -> .455 (corr with truth .978 -> .991; each model moves against its own T317f error, corr -.73), but the estimates shift up +.27 pp: the mean-point closure's negative bias had been cancelling part of the walk approximations' ~+.6 pp. Post hoc, s = -1's variance alone gives F ~1.06 at 3.7k FPE. T327's projection (F ~3.8) did not hold **[T343, 14:15: does NOT replicate on fresh DEV3: sd ratio .912 [.76, 1.11] vs DEV2 .68; the variance cut was DEV2-specific]** |
| lows chosen from W (T338; t241-neuron-account 95691e09, mech_closure/; pre-registered 1d49b429) | D approx. (oracle) | K = 2 at 3,659 nodes (1.3x); no budget 2.4x nodes | closure c .473; W-derived K = 2 .410; latest low .426; adjacency .423 (difference -.017 [-.051, +.017]); no budget .309; T327's spread .255 | - | [measured] NO GO: W-chosen lows equal simple recency. W ranks the earliest low positions most important, which the prefix tree cannot afford. Its running-max reading [speculative] is refuted by T341 |
| order-statistic closure (T341; t241-neuron-account 3c76f604, lows_orderstat/; pre-registered b3342549) | D approx. (oracle) | - | spread s+ .333, s- .248 (reproduces T327); closure c .472; v5 at its true position .314; v5 at a fixed slot .483; true running max .359 (bias +.31); expected running-max profile .602 | - | [measured] NO GO; the running-max reading is REFUTED (best vs s-: +.066 [+.010, +.120]). v5 carries information only at its true position. s- has a parity-dependent bias (+.058 / -.044 by the parity of the latest free position); a sign-alternating, period-~2 recurrent mode is [speculative] |
| combined arms (T317h attempt, T351; t292-oracle-floor, pre-registered 67e624b2) | D approx. | T317f 3,326; spread 3,748; tilt 6,124; spread + tilt 6,543 | .827 (bias +.489) / .884 (+.758) / .670 (+.038) / **.609** (+.413) | 3.14 / 4.04 / 3.79 / 3.35; MSE x C 2,276 / 2,931 / 2,750 / 2,426 | [measured] NO FREEZE: the pre-registered selection keeps T317f. The tilt fixes the bias, but its crossing pass as executed costs +2.8k FPE. Strict still needs RMS <= ~.38 at 5k |
| **step 14 (DEV2 only):** spread + tilt reimplemented cost-only, reusing the walk's crossing-cell state (T353; t292-oracle-floor, pre-registered bfc15b92; frozen as t351/estimator_h2.py) | D approx. | 5,079 as executed (A3 6,542 in this count; reuse charge 5,976 FLOPs per wall row) | .609 (bias +.413, sd .451); estimates bitwise equal to arm A3 on 80/80 | 2.60 (T317f 3.14); MSE x C 1,883 (go line 2,048) | [measured] GO on DEV2 (unverified). Pre-registered no-retune DEV3 check against T317f next (T342 panel); if confirmed, the candidate for HP5's single submission. A lenient-track gain; strict needs RMS <= ~.38 at 5k |
| noise breakdown curve of T317f (T337; pre-registered 4ff2753c; target + zoo0-4, 4 copies per level, 1e7 truths) | D approx. | - | relative sigma 0 / .001 / .0025 / .005: .85 (bias +.58) / 1.04 (+.75) / 2.19 (+1.63) / 3.86 (+2.94) | 0 3.3; .0005 3.1; .001 4.0; .0025 12.0; .005 32.4 | [measured] breakdown between .1% and .25% relative noise, tracking the accuracy drop (per-copy estimate error vs true error r = .81-.84); at .0025 the target is off +.28 pp, zoo1/zoo2 +3.3. Noise creates errors far from the tie faces: interior errors with both gaps >= tmax grow .09 -> .76 pp (about half the bias); rank-4+ errors .97 -> 3.2 pp. At .0025, tmax 6 cuts zoo0's error +1.60 -> +.93; C1 and blind g12 barely matter; the remaining ~+.9 pp is inferred to be closure and product |
| rare-event estimates (T340; t286-cheap-frontier 620eb47b, rare_events/README.md; pre-registered c0564f03, addendum 716ea8b5) | D approx. | variant A 11-19k; variant B 0.21-0.32M | est/truth: E1 (rank-3 pick with g23 > t0) variant B seeds .64-.89, target .20-.29; variant A misses the islands (100x-1e14x low); E2 (wrong with both gaps >= t) variant A 6/6 within 3x | F uninformative for rare p (relative error used) | [measured] PARTIAL POSITIVE (unverified). Truths: GPU float64, target 1e10, seeds 1e9 samples. Equal compute (post hoc, executed C): variant A wins 5 / ties 7 / loses 3 of 15; B 5 / 3 / 7; the better variant per event family 10 / 3 / 2. No single estimator beats sampling in a majority of cases (the DRI's earlier "12 of 15" counted ties as wins; corrected). Independent check T350 and T340c (T349) pending |
| certified rare-event bounds (T344; t218-tie-tail-labels 55368498, r1_cont/README 'T344') | D (certified) | - | - | - | [measured] feasibility negative (unverified), scoped to interval certificates and exact sub-cone enumeration. Interval propagation certifies boxes only <= 1e-4 wide around far-gap correct points, even in 3-D (v2, g12, g23), because \|Wh\| compounds over 10 steps; far-gap sub-cone enumeration projects to ~2e8 regions at step 10 (kill line 1e6). Only the trivial bound P(g23 > t0) (4.4e-2 / 1.7e-2 / 5.9e-3) is rigorous, 8,200-27,500x looser than the target's truths. A bound conditional on the declared closure was not pursued (not rigorous; could undershoot) |

### 9.3 The common failure
Every cheap deductive approximation tried fails for one of four reasons [inferred from the rows above].

1. **Frozen gates erase the model-specific signal.** Frozen at one input per class, every model of this architecture
   gives nearly the same map; models differ through gates switching inside each class. BLIND-11 predicts a spread of .06
   pp across models against a true .14 (T293). A canonical witness is a typical, well-separated input, classified
   correctly even when its class has many errors; finer classes do not help (T287). Exact last steps do not rescue
   frozen early ones (T274b, **verified** by T282).
2. **Gaussian state and readout break by steps 7-9.** Margins pile up near 0 at near-ties and have hard edges where
   gates switch, so their law is a mixture. Fed the exact sampled moments, a class-level Gaussian readout is still off
   by -10.95 pp on DEV2, against -.87 for frozen gates (T286's oracle). Per blind unit, the exact Gaussian orthant
   passes 36% of units, and a mean-field state plus exact last m = 0-3 steps 16-30%: the state is already wrong by steps
   7-9 (T295).
3. **Relaxation bounds widen ~50x per step.** Width over margin grows 10.6 -> 514 (forward) and 5.8 -> 229 (CROWN) from
   T = 3 to 4 at H = 4 (T289). A sound CROWN tree at (16,4) leaves .35 of the mass uncertified at 1e5 cones (T279).
4. **Exact branching grows 5-19x per step.** The regions needed to cover all but 1e-5 of the prefix mass grow x15-19 per
   step (H-E13, **verified** for k <= 4); a branch mixture over the last steps grows ~5-10x per step (T295). T287 reaches
   F <= 10 only with 4 exact first and 4 exact last steps: the region problem again. R0's error set needs its boundary
   resolved in 3-D per triple, >= 20-40k FPE (T297). T270 measures the next step.

*Why the same four* [inferred]. H-E15 (**confirmed**, scoped k <= 4, T <= 4): training puts the finest pieces of the
function at the tie walls, 37-44x finer near ties than far, against 1.3-6.6x for random weights. Each approximation
fails on those pieces: frozen gates miss the switching, Gaussians miss the hard edges, relaxations must shrink toward
every tie face (T279), and exact methods must resolve every piece. ANAT-80 (T292) [measured]: weak retrains differ by a
diffuse, model-specific widening of the soft tail (2.48 +- 1.94 pp), which carries 60% of the cross-model error
variance. In the weaker half, half of a model's excess tail error is spread over a median of 44 triples. Gross-error
triples carry 13% of the variance and the blind region 1%. So the model-specific signal is not in a few places, and R0's
bias grows with this tail (T296).

*[Updated 2026-10-18 12:00 UTC]* The later stops fit the same four reasons [inferred]. Mixtures of up to 128 Gaussian
components still smear the thin near-tie regions where a class's error lives (T309; reason 2). The smoothed closure
fails because the low inputs switch gates (T319; reasons 1-2). Capped walks and kinked tangents fail because the gate
flips near the tie carry the boundary information (T315, T318; reason 4, H-E15). The T317 family passes lenient
because it pays for the exact walks at ~5.4k FPE instead of avoiding them. Its remaining error comes from the closure
(oracle floor .46-.47 pp) and the declared approximations (walk-vs-closure error ~.72 pp, T318), which T324 itemises.

### 9.4 What remains open
- **R0 variants** (T298, T300; running, retrain-only scope). T297 closed the threshold integral, and with T296 the R0
  route in the cheap regime. T298 must beat R0's .51 pp floor by >= 20%. T300 shares R0's boundary across the max's
  position (90 classes, ~4k FPE) and aims only at the lenient bar.
  *[Updated 2026-10-18 09:30] Closed.* T298 passed its line (.389 pp, pre-registered) but its better floors cost an
  extra skeleton dimension. T300's stage-0 design is inductive and, on HP2, a point pass that is not a candidate (upper
  CI > 10); its stage 1 (sharing across the max's position) stopped at 1.40-1.87 pp. The R0 route was then
  carried to R1W V1 (T302-T313), a lenient fail on HP2 (§9.2 added rows) [measured].
- **Coarsened shadow network** (T290, running): can rounding and merging shrink the network at <= .4 pp accuracy change?
  *[Updated 2026-10-18 09:30] Closed:* KILL (unverified). No model both shrinks and stays within +-.4 pp (T290).
- **k = 6 complete masses** (T270, running): coverage growth g_1e-7 >= 12 ends the exact route; <= 7 and falling keeps
  it alive, at its own large cost.
- **Codex lanes** (TASKS). T267 (facet cancellation): bounded H4T3/H4T4 phase done, construction costs more than it
  saves, no general negative. T273 (ARC estimator A calibration): blocked on a DRI target definition. T283: an
  independent audit reproduces the T271/T277 rows; they establish neither equal rates nor prefactor-only causation.
- **Untried variants** (from the READMEs): value-dependent gates or exact comparator splits within a class (T286);
  weight-derived witnesses, finer classes (T274b, T287); sub-case conditioning (T293); Gaussian-mixture states (T295);
  alpha-CROWN, multi-neuron relaxations (T279, T289); lin over several anchors (T292).
  *[Updated 2026-10-18 09:30]* Added: a better boundary representation along v2/g12 (T307); accuracy-for-cost trades
  in R1W (T311); skew readouts (T295, T299); a deductive route to T308's predictor without rank_exact's cost (T308).
- **[Updated 2026-10-18 09:30] Lenient for a deductive estimator on HP2.** R1W V1 is 1.5x above the bar (official
  count; 1.35-1.7x across counts). It needs either ~20% less RMS from the boundary representation along v2/g12
  (T307), or C <= ~8.1k FPE at its RMS of .99 pp (T307, T311). The known cost levers are marginal: dropping the blind g23 line plus B2 gives ~8.7k FPE and a
  projected HP2 F ~10.7 (T311) [inferred]. Any new variant is a new HP2 submission and goes in the forking-paths note
  (~30 DEV2 variants so far across T297-T306).
  *[Updated 2026-10-18 12:00 UTC] Closed by cost alone.* Continuity restarts, the executed B2 prefix tree, G = 20 and
  the telescoping merge bring the same estimates (T317d-e within .0115 pp) to ~5.4k FPE: HP2 6.72-6.76, fresh HP3
  5.07-5.08, both with the upper CI <= 10 (T317-T317e, T321) [measured, unverified]. The candidate holds for retrains
  only, and strict fails.
- **[Updated 2026-10-18 09:30] The B2 recount** (T313). Done: HP2 F 13.51 [9.72, 17.55] with B2 (T281 recount)
  [measured]. It counts only once V1 is re-frozen with the sharing performed (DRI ruling), and it cannot change the
  verdict unless C falls below ~8.1k FPE [inferred].
  *[Updated 2026-10-18 12:00 UTC] Superseded:* T317 executes B2, so it counts (§9.2 step 8).
- **[Updated 2026-10-18 09:30] Future-aware mixture propagation** (T309, running): merge Gaussian components only
  along directions the future readout ignores. It stops if more than 50 components per class are needed.
  *[Updated 2026-10-18 12:00 UTC] Closed:* STOP by the pre-registered rule (unverified). Non-blind classes need a
  median of more than 128 components (T309).
- **[Updated 2026-10-18 09:30] T308's gap.** Anchor-level quantities carry enough model-specific information for .48
  pp, but only with rank_exact's 25.8k-FPE output, and the predictor is fitted. Cheap features alone give .72 pp
  [measured]. Whether a deductive estimator can extract this cheaply is open.
  *[Updated 2026-10-18 12:00 UTC]* T314's richer learners on cheap features (<= 4k FPE) reach .855 pp at best:
  information-infeasible for cheap strict, scoped to those feature families and learners (unverified). Untested: other
  per-class probes, larger training panels.
- **[Updated 2026-10-18 09:30] Strict.** Open in principle, but no built deductive design comes near it (§9 headline)
  [inferred]. The only non-sampling estimator at sampling's level is inductive fixed-node QMC (v5q, HP2 F 1.06 at 4k;
  T281) [measured].
  *[Updated 2026-10-18 12:00 UTC]* Now framed by the strict-bar arithmetic (§9 headline). At RMS .80 pp it needs C ~1k
  FPE, or at ~5.4k FPE it needs RMS <= .37 pp, below the closure floor. Running: T326 (cost), T327 (closure) and T324
  (error budget). These are open lanes, not results.
  *[Updated 2026-10-18 14:00 UTC]* The lanes have reported (strict-bar arithmetic above): T326, T330/T330b and T331
  are negative on cost (T330's declared C1 screen became T317f); T324/T334 itemise the error; T327 -> T339 and T333 ->
  T347 -> T351 each fix part of it but expose or leave a positive bias, and none of these froze a variant [measured,
  unverified]. Open: a deductive removal of the walk approximations' positive bias (~+.6 pp, T339's reading; +.41 pp
  in T353's arm) and a further cut in the model-to-model spread: T353's sd .451 alone gives F ~1.4 at 5,079 FPE
  [inferred, arithmetic]. T329 (a structurally cheaper geometry) is still in the backlog.
- **[Updated 2026-10-18 12:00 UTC] H-L1's open items** (INDEX): the strict bar; noise robustness (HP1 noisy F 39-206;
  bias up to +6.6 pp at absolute sigma .05, red-team); the per-approximation error budget and an audit by a second,
  different agent (T324); est(M_16,10) against the truth, now reported (+.087 pp, one model; T281 1515eb10).
  *[Updated 2026-10-18 14:00 UTC]* H-L1 now cites T317f (HP4 F 2.63 [1.93, 3.39]; sampling ~2.6x lower MSE). The error
  budget and a second-agent audit are done (T324; product and tmax terms **verified** by T334). Noise robustness:
  T317f breaks down between .1% and .25% relative noise, because noise creates errors far from the tie faces (T337)
  [measured, unverified]. Still open: the strict bar.
- **[Updated 2026-10-18 12:00 UTC] Candidate dossier** (T323): a draft is pushed (claude/t241-neuron-account
  4669ad5b) and awaits DRI review. The red-team advises against sharing it with ARC as a candidate solution, since it
  fails "match" by ~5x and fails on weight-noised copies; after the listed fixes it could serve as a progress note
  [inferred, red-team].
  *[Updated 2026-10-18 14:00 UTC]* Done (T323): final v3, after the red-team and T324, framed as a progress note and
  not a candidate solution (HP3 F 5.07 [3.41, 7.03] at 5.4k FPE; six declared approximations; sampling-free at runtime
  but the scheme selected by DEV2 MC). Given to the owner 2026-10-18. It predates T317f.
- **[Updated 2026-10-18 09:30] Proposed, not started:** T291 (Astra r2 C3, shared-innovation polynomial
  coefficients with explicit remainders), awaiting DRI prioritisation.
- **[Updated 2026-10-18 09:30] Round-3 reviews** (T301). Fable round 3: pending publication. Astra round 3: not yet on
  the shared branch.
  *[Updated 2026-10-18 12:00 UTC]* Fable rounds 3 and 4 are on the DRI branch claude/project-handoff-priorities-zpzfm8
  (review/fable_review_r3, review/fable_review_r4). Their ideas became T307-T310 and T317-T319. Astra round 3 is not
  yet on the shared branch.
  *[Updated 2026-10-18 14:00 UTC]* Closed (T301): the second reviewer's round 3 was never delivered. Round 5 of the
  first reviewer is on the DRI branch; its ideas became T326-T332 (T331 = r5 D1, addendum A = r5 D3, T327 arm B, T332
  settling test).
- **[Updated 2026-10-18 12:00 UTC] Untried variants added:** cone-clipped J and an exact 1-D treatment of the
  largest-|J| low (T319, now T327); EM or truncated-Gaussian mixtures, a step-9 representation (T309).
  *[Updated 2026-10-18 14:00 UTC]* Also untried: zonotope certificates for the safe-to-tmax screen (T330); exact 2-D
  correct sets at >= 2 anchors per region at an affordable price (addendum A, T333); a model-free derivation of the
  tilt's magnitude (T347; the review note finds none); an exact 3-D polytope bound conditional on the declared closure
  (T344; not rigorous); rare-event walks at several g12 quantiles or a 2-D walk (T340c, now T349); T340 on noisy
  copies and more models.
- **[Updated 2026-10-18 14:15 UTC] Rare events (T349 / T340c):** with g12 at exact-law Gauss nodes, the rare-event estimates close within 2x (K5 9/9 on E1, K3 6/6 on E2; probabilities 2e-7 to 4e-4) but cost 1.8-4.5M FPE; equal-compute wins only where p x C <~ 90 (4/3/8 of 15). The deductive cost does not scale with 1/p, so the rarer the event the larger the advantage [inferred]. Cost lever T354; independent check T350.
- **[Updated 2026-10-18 14:00 UTC] Running checks** (TASKS): T353's pre-registered DEV3 check against T317f, on DEV3
  (T342: 100 open retrains, seeds 500-599, truths published; being built); T343, an independent check of T339's
  decomposition on DEV3 by a different agent; T350, an independent re-check of T340 on >= 6 DEV3 models; T349 (T340c),
  transverse integration for T340's variant B. HP5 (T336, seeds 400-499) is the reserve for the next single sealed
  submission. Proposed, low priority: T345 (a feed-forward architecture control), T346 (exact-methods brainstorm),
  T352 (quantised-activation variants).

**No impossibility proof exists.** Every negative above is scoped to the methods and settings tested, as both round-2
reviews say (Fable r2 B, Astra r2 B). Their subjective odds of a strict cheap win on HP2 within three months are 15% and
30% [speculative].
**[Updated 2026-10-18 09:30]** This still holds after T296-T313. The strict verdict in the §9 headline rests on floors
measured for the designs built (closure floors, integration-error curves, a fitted information ceiling). It is not a
bound over all designs [inferred].
**[Updated 2026-10-18 12:00 UTC]** This still holds after T309-T325. The T317 family changed the deductive lenient
verdict by cost alone, with unchanged estimates; the strict verdict rests on the same measured floors (§9 strict-bar
arithmetic) [inferred].
**[Updated 2026-10-18 14:00 UTC]** This still holds after T326-T353. T317f moved the lenient score with one more
declared approximation; the strict verdict rests on measured floors and costs (T324, T333, T339, T351), not on a bound
over designs [inferred]. T344's certificate negative is scoped to interval certificates and exact sub-cone
enumeration.

### 9.5 Positive facts worth keeping
- **Deductive errors on retrains are nearly constant offsets** (T286) [measured]. rank_exact is off by -1.39 pp on
  average (76/80 models negative, sd 1.06), probe v1 by +3.28 pp (80/80 positive, sd 1.73), and both correlate .96 with
  the truth. A DEV2-fitted constant would give F ~40 / ~16, but it is advice and not allowed, and the residual spread
  still exceeds the 1k allowance. The methods rank the models; a weight-derived offset is missing [inferred].
- **R0's oracle floor on retrains is .51 pp** (DEV2: mean +.22, slope -.05 pp per pp; LB3 seeds .61; T292, T296)
  [measured]. It was not pre-registered in T292, and T296's retrain-only scope is post hoc. It would fit the strict 1k
  allowance with a cheap integral, but T297 finds none below 20-40k FPE, where the strict allowance is ~.2 pp
  [inferred]. It fails on noisy copies (1.39 pp on all of LB3).
- **The mean-field closure tracks between-model variation** of the blind-region contribution (corr .73 on DEV2, .80 on
  LB3; frozen gates -.30), but its response is compressed about 2x (predicted sd .049 vs true .139 pp) (T294)
  [measured]. A post hoc calibration fitted on LB3 transfers to DEV2 (RMS .138 pp), so the compression is systematic
  (advice, not counted).
- **[Updated 2026-10-18 12:00 UTC] A sampling-free estimator carries real model-specific information on retrains**
  (T317 family; H-L1) [measured, unverified]. Its correlation with the truth is ~.98, and its n* is ~1.1k FPE against
  53-73 for a panel-mean constant. Its error shrinks with accuracy: on the p > .95 subsets RMS is .37-.53 pp and bias
  <= +.18 pp (T281 1515eb10). At ~5.4k FPE it is still ~5x worse than sampling (F 5-7).
