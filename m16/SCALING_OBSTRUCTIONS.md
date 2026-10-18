# Scaling obstructions for deductive accuracy estimation of M_16,10 (2026-10-18)

Status of every claim: **[measured]** or **[inferred]**, and **verified** (independently re-checked, DRI-confirmed) or
**unverified**. Sources are `branch@commit, file`. Branches: shared = `ccr-6a61f25c-wn80ii@93b658db`;
T194 = `claude/t194-width16-probe` (this file's branch); T192 = `claude/t192-gpu-cost-law@e5a40128`;
T218 = `claude/t218-tie-tail-labels@8645f749`. Placeholders `[PENDING ...]` are for the DRI to fill.

## 1. The question and the bar
Estimate the accuracy of the 432-parameter 2nd-argmax RNN M_16,10 (`alg_zoo.example_2nd_argmax()`, accuracy
0.95313575 ± 3.7e-5 from 3.2e7 samples; T192 3e10-sample MC .95311695 ± 1.2e-6) without random model inputs, with MSE
no worse than plain sampling at equal compute. It should also hold on the 5 trained seeds and on noisy weight copies.
Compute is in T196 FLOPs (multiply-add = 2, other scalar op = 1; sampling costs c_s = 5948 FLOPs per input at (16,10))
[shared@bf9f0ead, review/cost_convention/README.md; DRI ruling]. At cost C FLOPs, sampling's standard error is
sqrt(p(1-p) c_s / C) ≈ **16/sqrt(C)**: 1.6e-4 at 1e10 FLOPs, 5e-6 at 1e13. A method beats sampling only if its
error is below that.

## 2. What works
- **M_4,3 (32 params):** a mechanism-derived estimator (E36) and one frozen generic recipe (E37) beat sampling on all 5
  trained seeds under ARC's noise protocol (MSE ratios 0.0002-0.25) [measured, **verified** (scoped): H-U1/H-U2, INDEX].
  Cost is about that of exact enumeration there.
- **T = 3 ladder (H = 4..8)** and **(8,4)**: recipe108 (resolve-by-impact plus a multi-step margin certificate) passes.
  (8,4): MSE ratio <= 0.27 on 9 models, <= 0.36 under the T196 op audit [measured, unverified: shared, estT_fix/README.md
  E46; T192 README §3-4].
- **(16,3) and (16,4)** (target width): the frozen recipe108 beats sampling on all 5 seeds + 4 noisy copies. Ratio <= 4e-5
  at (16,3) (1.7e6 ops); <= 0.017 (0.035 T196) at (16,4) (2.5-3.0e9 ops) [measured, unverified; T194@a594da1,
  width16_probe/README.md "Results"; audit T199 pending].

## 3. Where it breaks
**3.1 The recipe stops winning at T = 5.** (8,5) seed 2 loses (MSE ratio 3.4 estT ops, ~4.6 T196); the other 7 (8,5)
models win [measured, unverified; T192 README §3]. (16,5) zoo0 loses (ratio ~35, error -3.2e-5 vs exact) [measured,
unverified; T192@2930bc92]. From T = 4 the recipe saves only 1.2-1.6x ops over exact enumeration [measured,
unverified; T192 §3; T194@45e1f6c width16_probe/README.md "T206"].

**3.2 Exact enumeration is infeasible.** Gate-pattern regions of the target's first k steps: **34 / 792 / 21,032 /
632,394** for k = 2..5, growth **x23.3, x26.6, x30.1** per step and rising. That extrapolates to ~1.5-3e13 regions at
k = 10, and ~1000 CPU-years of geometry before masses [measured to k = 5, inferred beyond; T192 README §9, results/prefix/
k2-k5.json; INDEX H-E13 "supported, unverified"]. Reproductions with different code: k = 2-4 by T194's numpy port
(identical counts; T194@ee01b49 width16_probe/prefix.py) and k = 2-3 by T218's LP counter (T218 tie_tail_labels/README.md).
These share the estT geometry or the LP method, not an independent arithmetic. Codex T219 independent integer recount:
`[PENDING T219]`. Zoo models grow alike: (16,3) 838 -> (16,4) 28,200 -> (16,5) 786,114 regions (x34, x28) [T192 §8].

**3.3 Cost is cone masses.** Exact-mode ops are 95% (16,4) / 97% (8,4) cone-mass quadrature. A leaf that needs a
mass needs ~6 simplices at ~3e4 ops each (d = 4: 440x the d = 3 closed form; d = 5: 5.2e4-1.7e5) [measured, unverified;
T194@45e1f6c "T206"; T192 §5]. 34-56% of leaves are fully correct and need no mass.

**3.4 Mass does not concentrate fast enough.** The number of regions needed to cover 1 - 1e-5 of the prefix mass grows
**x18.6 then x16.1** per step on the target (x17.5-21.6 on zoo (16,4)/(16,5)). Even N_1e-2 grows x8.6-13, at 65-93% of the
count's log-rate. Projected N_1e-5 at k = 10 is ~2e11 [measured k <= 4, rule pre-registered; inferred beyond;
T194@57bad7b width16_probe/README.md "T222", coverage.json]. T192 k = 5 coverage: `[PENDING T192 k=5]`. Independent
replication (T218 child, LP + adaptive Plackett masses): `[PENDING T222 replication]`.

**3.5 Structural savings found so far are constant factors.**

| idea | saving | status, source |
|---|---|---|
| truth-cone start (2T(T-1) cones instead of 4·T! chambers) | lossless; 0.67x / 0.76x ops at (8,4) / (16,4); chambers ~10% of log-growth | [measured, unverified] T192 §5; T194 "T206"; shared@ab532a27 review/chamber_split |
| oracle skip of small-error leaves (bias <= 1 sampling SE) | <= 1.9x (8,4), 3.8x (16,4); 4.8x with truth start. An upper bound: a real method must bound each cell | [measured, unverified] T194 "T206" |
| merging same-history leaves / same-future cells (T188) | exact merged cells 0.59-0.82x at T = 3-4 (T = 10 extrapolation 2-5x at H = 16); T188 counts with no savings claim | [measured, unverified] review/chamber_split; shared TASKS T188 handoffs b9a6b5df/3ee0667b |
| budgeted pruning by cell mass (T195 phase 2) | `[PENDING T195]` | claude/t195-split-attribution |
| weight symmetry (T216) | none: no permutation symmetry; projecting the near-collinear incoming rows {1,6,7,8; 4 anti} costs 79 pp (8.9 pp for the closest pair) for <= 2.7x fewer regions, growth still >= x16/step | [measured, unverified] T194@ee01b49 "T216" |
| event words record/2nd/low (T217) | modal gate word holds 21% of an event word's mass at k = 4 (go: 99%) | [measured, unverified; DRI: kill] T194@050bfaa "T217" |
| mechanism-labelled tie tails (T218) | top-100 labels cover 2-4% of out-of-sample tail mass (below a random-label null) | [measured, unverified; kill] T218 README |
| local rank-conditioned certificate (T189) | sampling's SD is strictly smaller at every budget >= 7630 / 94116 FLOPs even with the certificate free (H4T4 / H8T4) | [measured, unverified] shared TASKS T189 row |

**3.6 Noise does not change this.** At T = 4 the noisy copies (relative sigma .005-.02, ARC absolute .05) have 0.94-1.21x
the clean regions and 0.94-1.33x the ops [measured, unverified; T192 §7]. Not tested at larger T.

**3.7 Region counts in earlier tables were understated.** The 2e5-sample lower bound missed up to 92% of regions:
(8,5) zoo0 has 42,322 exact regions, against 3,433 sampled. Restated against exact counts, recipe cells/regions are 0.99 / 1.99 /
2.70 along T at H = 8 (published 2.8 / 5.4 / 33) and 2.44 -> 0.46 along H at T = 3 [measured, unverified;
T194@95ed88e "T205"]. The fall along H is real but does not change the per-step growth along T.

## 4. The best deterministic number
sci_tao's tie-slicing with the blind region (<= 1 positive input, P = 11/1024) split off exactly. Over 16 independent rule
scrambles on the target: MSE ratio **0.75 (16k FPE) / 0.89 (64k)**, i.e. **1.1-1.3x better than sampling**, on the
clean target only. It is quadrature class (fixed Sobol node sets), not mechanistic. On leaderboard v3 pooled over 30 models it
does not beat sampling at any budget (3.4-6.5; plain sci_tao 1.4-3.6, CIs reaching below 1 only at 1k and 256k). The
earlier "sci_tao 0.03 @ 256k on orig" is a single lucky deterministic run (2.16 at 16k; 0.99 / 1.56 over scrambles)
[measured, unverified; T194@2809650d m16/approaches/tie_blind/README.md].

## 5. What a successful method must do, and what is untested
**Must** [inferred]: aggregate across regions rather than visit them. ~1e13 linear regions, and covering 1 - 1e-5 of the
mass alone needs ~1e11. The error is not a few families but a continuum of rare tie-band events (T218), spread over many
gate words per mechanism-level event word (T217). The answer must therefore come from a quantity that is smooth across
regions (a bound or an expansion in the order statistics, the clock phase or the margins) and be certified to ~16/sqrt(C).
Any region-by-region method, exact or pruned, inherits growth of >= x16/step.

**Not ruled out** (from the READMEs' "alternatives" sections):
- precision-matched cone masses: quadrature order set by each cell's needed precision (T206; deprioritized as T212);
- richer event words: top-2 gap, age of the max, signs of all inputs (T217);
- intermediate-resolution or continuous-feature tail labels (clock phase) and tail-aware importance rules (T218);
- symmetries beyond neuron permutation and sign: general linear maps mixing neurons, task symmetries such as time
  reversal (T216);
- the coverage curve at k >= 5 (pending), per cell rather than per region, and after same-future merging (T222);
- exact treatment of the all-negative orthant (1/1024, all gates start off) inside the blind region; fixed rather than
  pilot-based allocation; skipping the blind middle of R23 walks (T223);
- certificates with other domains or anchors, and one-sided envelopes (T189);
- whether the noise insensitivity of region counts holds beyond T = 4 (T192);
- a genuinely independent region recount in exact arithmetic (T219, pending).
