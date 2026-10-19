# minimal_pair (T10): which error patches appear at H=16, T=5 that do not exist at T=4?

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [TASKS](../../../TASKS.md) ·
[sci_tao](../sci_tao/README.md) · [ht_seeds](../ht_seeds/README.md)

## Question
The deterministic tie-face calculation (sci_tao / ht_seeds) beats sampling by 56-109x at H=16, T=4 but only by
4.7-15x at T=5 (5 seeds, sci_tao's own cost count; the leaderboard charges that code 2.23x more). Which error mass is
new at T=5, where does it sit in input space, which neurons/gates make it, why does the T=4 calculation mis-weight
it, and can a frozen deductive recipe fix T=5 on all 5 seeds?

## Short answer
1. **New error mass** [measured]. The error rate rises 4.4x, from 0.17-0.27% to 0.75-1.28%. The mix of wrong
   outputs does not change (about 1/3 predict rank 1, 2/3 rank 3, <1% rank >= 4), so this is not confusion with a
   third element. The largest new patch is the **all-negative orthant**:
   - P(err | all x<0) is 5.0-10.2% at T=5 vs 0.6-0.9% at T=4;
   - there the model has a **bulk error floor of 2-6% even far from any tie**, plus wide relative tie bands
     (19-34% error when x(3)/x(2) < 1.1, vs 3-4% at T=4);
   - this orthant holds 48-76% of the wide errors (relevant gap > 0.05).
   The one-positive class and the all/most-positive classes also rise 3-5x. The readout margin shrinks in every sign
   class at T=5 (1.5-2.6x), most in the all-negative class (2.5x). Neuron counts do not explain it: active neurons
   per step are similar.
2. **Why the T=4 calculation loses ground** [measured, inferred]. Its relative accuracy hardly changes. The
   integrand's CV (face-to-face s.d./mean, Neyman-combined) is 1.13-1.32 at T=4 and 1.30-1.93 at T=5, and cost per
   face is ~13 FPE at both. For any estimator with fixed relative error, the gain over sampling is
   (1-err)/(err·CV²·c), so a 4.4x rise in the error rate costs ~4.4x of gain by itself:
   - seed 0's whole T4->T5 drop is the error rate;
   - seed 2's drop is about half error rate, half CV².
   CV stays at 1.4-2.6 through T=10 while err keeps rising (D5). **The "collapse" is mostly the model getting
   worse, not the integrand getting rougher.**
3. **Fix** [measured]. A frozen deductive estimator (`estimator.py`) **beats sampling at H=16, T=5 on all 5 seeds** on
   the leaderboard, with cost counted as implemented:
   - per-seed MSE ratio 0.01-0.23 (16k) and 0.00-0.22 (64k);
   - 32-rule-seed gains of 5.9-10.4x (16k) and 8.7-14.1x (64k), against 1.9-8.7x / 3.3-9.1x for the plain rule
     with the same honest walk.
   It does **not** restore T=4-level gains (43-267x on the same recipe). By point 2, no reallocation can: zeroing
   the whole <=1-positive region's variance would give only 15-38x (iid), and oracle config x sign x u strata give
   only 2-3x over the kind split.

## Class: deductive in the "no random model inputs" sense (AGENTS rule 6), quadrature in the leaderboard taxonomy
- **No random model inputs.** `est` ignores `rng`. The nodes are fixed scrambled-Sobol point sets: five rules,
  seeds `RULE_SEED + 101 j` with RULE_SEED = 31415, chosen in advance and identical for every model. The pilot and
  Neyman stages are deterministic functions of the model's own walk results.
  - Determinism check [measured]: the leaderboard ran each of 116 (model, budget) cells with 24 different rngs, and
    every cell gave exactly 1 distinct estimate (`lb_raw_minimal_pair.jsonl`).
- **What "32 rule seeds" means.** It is a robustness study over 32 alternative fixed rules, as in ht_seeds. The
  spread of their errors is a quadrature-error estimate; the frozen entry uses one rule. **"iid"** appears only in
  diagnostics (D2/D5 use iid faces to measure the integrand's variance); the estimator does not use it.
- **Honest scope.** By the leaderboard's taxonomy (T14) this is **quadrature**, not mechanistic: an average of exact
  1-D integrals over a fixed, model-independent node set. Replacing the nodes by random points would give a
  variance-reduced Monte Carlo, the same class as sci_tao and tube. It therefore meets T10's "deductive"
  criterion as AGENTS rule 6 defines it, but it is **not** a mechanistic solution in ARC's sense.
  - A mechanistic version would replace the 3-4-D face average with an analytic or enumerated treatment of the
    face integrand. Example: exact cone masses per linear region in the all-negative orthant, which carries 33-56% of
    the sqrt-variance.
  - Expected loss: unknown. The variance bounds in D2 cap even a perfect treatment of that region at 15-38x iid
    gain.
- Order of work: the diagnostics (e70e219a) and the estimator were committed before the leaderboard scoring
  (a5063c7d); the pre-registration (1b4143e1) came before any run.

## Pre-registration (written 2026-10-18, before any diagnostic or scored run; commit 1b4143e1)
Unchanged text: see `git show 1b4143e1:m16/approaches/minimal_pair/README.md`. Summary:
- diagnostics D1-D4;
- hypotheses H1 (memory of early inputs), H2 (third-party confusion), H3 (roughness, not location);
- fix candidates (1) configuration/sign-stratified two-stage allocation, (2) region-targeted slicing,
  (3) weight-derived strata;
- pass criterion: `leaderboard.py run minimal_pair --H 16 --T 5`, MSE ratio < 1 on each of seeds 0-4 at 16k and
  64k (point estimate), seed-group CI upper bound < 1 at 64k, and the same recipe re-run at T=4.

**Deviations, declared:**
- The strata actually used are sign-pattern strata (D1c showed sign class, not position, organises the new mass).
  The pre-registered configuration strata were measured only as an oracle bound.
- The radial integration and the honest-cost walk were added after the diagnostics.
- D5 (CV law along T) and the counterfactuals were added.
- Fix candidates 2 and 3 were not run (see exhaustion checklist).

## Method
Key files:
- `core.py`: copy of ht_seeds/core.py, with a `cap` argument in face_points and a `cdf` argument in walk_mass.
- `walk.py`: exact segment walk with the prefix saving **implemented**. Pre-activations of steps before the first
  changed gate are updated linearly (O(H)); only later steps are recomputed with Wh (O(H²)); no tangent before step
  min(a,b). Cost = recomputed value steps + recomputed tangent steps (each 1/T FPE), plus all O(H) updates,
  gate-change tests, root tests and readouts at their FLOP share. Masses match core.walk_mass to 1e-14 (`t_walk.py`).
- `hom.py`: radial integration. The model has no biases, so its error set is a cone; the face law is a Gaussian
  restricted to an order cone, so y = r w with r ~ chi_{T-1} independent of w. Integrating r out:
  F_hom(w) = ∫_{|s|<tau(w)} 1[err(w + s d)] k(s) ds, where k is the Student-t_{T-1} density of s·sqrt(T-1) and
  tau(w) = tau(y)/|y|. E[F_hom] = E[F] exactly (checked on 10 models, `t_hom.py`). The walk is the same; only the
  1-D weight changes.
- `estimator.py` (frozen, `est(model, budget, rng) -> (estimate, cost)`, rng ignored):
  - co-area identity P(wrong) = Σ Z_kind E[F];
  - **model-free sign strata**: R23 {u<0,o1<0} (all negative), {u<0,o1>0} (one positive), {u>0}; R12 {u<0}, {u>0}.
    The conditional face laws are mapped from Sobol points, with the o1 split carried as a weight;
  - radial integrand;
  - pilot of equal power-of-2 Sobol prefixes per stratum until >= 25% of the budget is spent, then Neyman allocation
    n_h ∝ M_h sd_h / sqrt(c_h), all points reused;
  - fixed scrambled-Sobol rules (RULE_SEED 31415).
  Deterministic; class to be assigned by the DRI (it is quadrature in the leaderboard's sense: an average of exact 1-D
  integrals over a fixed node set).
- Diagnostics: `d1_taxonomy.py`, `d1b_wide.py`, `d1c_npos.py`, `d1d_negorth.py` (sampling taxonomy);
  `d2_faces.py`, `d2_analyse.py`, `d2_counterfactual.py`, `d2_config.py` (64k iid faces per kind per model, raw
  in data/, git-ignored); `d4_neurons.py`, `d4_margin.py`; `d5_law.py`.
- Scoring: `mse_study.py` + `mse_analyse.py` (32 rule seeds, 4 variants), `truths.py` (independent 48M-sample
  truths, T=5), `lb_perseed.py` (pre-registered per-seed check on the leaderboard raw runs).

## Status
Done (2026-10-19). Pass criterion met at T=5; the T=4 rung passes with the same recipe. The question "why T=5
loses ground" is answered by an error-rate law plus a located patch. Restoring T=4-level gains is a scoped negative
(unverified) for reallocation-type fixes.

## Results

### D1: error taxonomy (2M or 4M samples per model)
Reproduce: `python d1_taxonomy.py 2000000`, `python d1c_npos.py`, `python d1d_negorth.py`.

| | T=4 (seeds 0-4) | T=5 (seeds 0-4) |
|---|---|---|
| error rate | 0.17-0.27% | 0.75-1.28% |
| predicted rank 1 / 3 / >=4 | 32-41% / 59-68% / <0.3% | 30-35% / 64-70% / ~0.8% |
| relevant gap q99 of errors | 0.016-0.030 | 0.12-0.30 |
| P(err given all x<0) | 0.62-0.86% | 4.95-10.2% |
| P(err given one positive) | 0.15-0.29% | 1.09-1.58% |
| wide errors (gap>0.05) in all-negative orthant | (rare: 0.2-0.6% of errors are wide) | 48-76% of the wide errors |
| all-negative orthant, far from ties (x(1)/x(2) < 0.5) | 0.3-0.6% err | 1.9-5.8% err |
| all-negative orthant, x(3)/x(2) < 1.1 | 2.8-3.9% err | 19-34% err |

Error mass by number of positive inputs k (`d1c_npos.log`): every class rises at T=5. The conditional rates for
k=0 rise 8-12x, k=1 5-7x, and k=T-1, T rise 3-4x; the middle class (k=2) rises least.

### D2: where the face-integrand variance lives (64k iid faces per kind)
Reproduce: `python d2_faces.py 65536; python d2_analyse.py; python d2_counterfactual.py; python d2_config.py`.
- The <=1-positive faces (u<0) carry 60-85% of E[F²] on seeds 1-4 at T=5 (44% on seed 0); at T=4, 41-56%.
- iid Neyman gain (walk cost as in sci_tao):

| | kind split | sign strata | config x sign x u4 (oracle) | <=1-positive variance set to 0 |
|---|---|---|---|---|
| T=4 | 26-31x | 29-34x | 50-67x | 81-96x |
| T=5 | 3.1-6.0x | 5.1-7.4x | 8.1-14.3x | 15-38x |

### D4: mechanism
Reproduce: `python d4_neurons.py; python d4_margin.py`.
- Median relative readout margin (target logit - best other, / max|logit|) by sign class k = 0..T:
  - T=4: 0.14-0.21 (k=0), 0.24-0.41 (others);
  - T=5: 0.057-0.078 (k=0), 0.09-0.28 (others).
- Number of Wi<0 neurons (the only ones the input drives at step 0 on negative inputs): 5-11 at T=4, 6-10 at T=5.
  Active neurons per step on negative inputs: 9-11 at both T. So it is not a dead-neuron effect.

### D5: the error-rate law along T (16k iid faces per kind, honest walk)
Reproduce: `python d5_law.py 16384` (the printed "(1-p)/(err CV^2 c)" column has a typo, (1-p) for p; ignore it).

| T | err | CV | c (FPE/face) | iid gain |
|---|---|---|---|---|
| 3 | 0.05-0.06% | 0.96-1.38 | 13-14 | 67-138x |
| 4 | 0.17-0.27% | 1.13-1.32 | 13-14 | 19-26x |
| 5 | 0.76-1.29% | 1.30-1.93 | 12.5-13.4 | 2.3-4.7x |
| 6 | 1.5-3.0% | 1.96-2.42 | 9-13 | 0.6-1.5x |
| 8 | 2.4-4.2% | 1.61-2.61 | 7.6-9.0 | 0.8-1.4x |
| 10 | 5.2-12% | 1.38-2.01 | 5.3-10 | 0.3-0.9x |

The identity gain = (1-err)/(err·CV²·c) holds by definition of CV. The measured content is that CV stays within
~1-2.6 at every T while err grows ~200x from T=3 to T=10. So the decline in gain is mainly the error rate.

### Frozen estimator vs sampling: 32 fixed rule seeds per cell, gain [95% bootstrap CI]
Reproduce: `python mse_study.py 5 32 16000,64000; python mse_analyse.py 5` (truths: own 48M samples, `truths.json`;
T=4 truths: ht_seeds 16-23M). Variants: plain = sci_tao face law and kind-level Neyman; radial = + hom integrand;
strata = + sign strata; frozen = both. All use the honest walk.

T=5:

| variant | budget | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|---|
| plain | 16k | 8.7 [6.3,14] | 5.3 [3.6,8.6] | 6.0 [3.8,11] | 4.7 [3.0,9.0] | 1.9 [1.3,3.2] |
| plain | 64k | 8.2 [5.7,14] | 9.1 [6.6,14] | 4.6 [3.1,7.8] | 4.8 [3.4,7.8] | 3.3 [2.4,5.1] |
| radial | 64k | 10.8 [7.2,18] | 9.0 [6.1,15] | 4.1 [2.7,7.1] | 5.4 [3.5,9.3] | 3.4 [2.4,5.3] |
| strata | 64k | 13.9 [9.1,24] | 9.0 [6.7,13] | 7.8 [5.1,13] | 9.8 [6.5,17] | 9.0 [6.1,15] |
| **frozen** | 16k | 7.2 [4.7,12] | 10.4 [6.1,22] | 7.9 [5.0,15] | 6.4 [4.5,9.8] | 5.9 [3.6,11] |
| **frozen** | 64k | 14.1 [9.5,24] | 11.5 [8.4,18] | 8.7 [5.6,15] | 13.0 [9.1,21] | 9.6 [7.0,14] |

- The plain rule has a positive bias at low budget (z up to +3.2): the heavy tail is under-sampled.
- The strata remove it (|z| <= 0.7 for frozen at 64k, <= 2.4 at 16k).
- The sign strata are the main gain on the weak seeds (seed 4: 3.3 -> 9.0x at 64k); the radial integrand adds
  ~1.0-1.3x.

T=4 (same recipe; ladder rule 3): frozen 16k: 52 / 138 / 110 / 73 / 49x; frozen 64k: 150 / 267 / 105 / 98 / 43x (seeds 0-4).
The plain rule with the honest walk gives 27-38x (16k) and 42-84x (64k). `python mse_study.py 4 32 16000,64000; python mse_analyse.py 4`.

### Leaderboard (`python m16/leaderboard.py run minimal_pair --H 16 --T 5`, then `--T 4`)
Leaderboard truths 4M samples. Raw rows (final runs only) are in `lb_raw_minimal_pair.jsonl`; the T=5 table is `lb_H16_T5.json`; the logs are
`lb_T5.log` and `lb_T4.log`. "unreviewed" class (as for any new entry), so it was run REPS=24 times; the rng is
ignored, so the runs are identical.

| group | 16k ratio [95% CI] | 64k ratio [95% CI] |
|---|---|---|
| T=5 seeds 0-4 | 0.11 [0.05, 0.17] | **0.08 [0.01, 0.17]** |
| T=5 s0.0025 / s0.005 / s0.01 | 0.04 / 0.12 / 0.05 | 0.08 / 0.25 / 0.02 |
| T=5 s0.02 | **1.46 [0.62, 2.30] (loses)** | 0.73 [0.25, 1.29] |
| T=4 seeds 0-4 | 0.01 [0.00, 0.01] | -0.01 [-0.02, -0.01] |
| T=4 noisy groups | 0.03-0.55 | 0.01-0.26 |

- Mean cost at T=5, 16k: 16.4-17.4k, because the Neyman stage overshoots slightly. Ratios use the charged cost.
- Time ratio: 30-37 at 16k, 10-11 at 64k (see Compute accounting).

Pre-registered per-seed check (`python lb_perseed.py 5`).
MSE ratio per seed, single run, vs leaderboard truth / own 48M truth:
- 16k: s0 0.17/0.04, s1 0.01/0.02, s2 0.22/0.23, s3 0.10/0.13, s4 0.07/0.07;
- 64k: s0 0.21/0.00, s1 0.00/0.04, s2 0.20/0.22, s3 0.05/0.10, s4 0.01/0.02.

**All < 1: the pre-registered pass criterion is met** (seed-group CI upper bound 0.17 < 1 at 64k). At T=4 all
per-seed ratios are <= 0.016.

## Compute accounting
- Unit = one forward pass of one length-T sequence (FPE, leaderboard rule 1).
- Charged per cell (walk.py): recomputed value steps + recomputed tangent steps (each 1/T FPE), plus for all T steps
  the O(H) linear update, gate-change test, root test and the readout, at their FLOP share. The pilot is charged.
- The first version charged the O(H) gate-change test only on non-recomputed steps (~4-5% under-count). It was fixed
  before the final runs; the pre-fix ablation is kept in `mse_T*_precostfix.*` and gave ~5% higher gains.
- Model-free and not charged: face CDF tables, Sobol points (PRECOMPUTE_FLOPS = 0 declared; < 1e8 FLOPs).
- **CPU time ratio vs sampling is high: ~40 at 16k, ~8-11 at 64k.**
  - It falls 4-5x with budget while FLOPs per FPE are fixed. cProfile puts all the time in the walk's Python loop
    over cells with small batches (pilot chunks of 16 faces).
  - [inferred] So it is interpreter overhead, not uncounted arithmetic: every Wh product is restricted to the rows
    being recomputed.
  - Flagged for a DRI audit per leaderboard rule 6.
- Totals for this task:
  - diagnostics: ~2e7 sampled sequences (D1) + 1.3e6 faces x ~13 FPE (D2) + 1e6 faces (D5);
  - truths: 2.4e8 sequences;
  - ablations: 2 x (5 models x 4 variants x 2 budgets x 32 rules) runs, ~2.6e8 FPE;
  - leaderboard: 2 x 2 x 29 models x 24 reps at 16k/64k, ~1.6e8 FPE.
  All single-core, about 3 CPU-hours.

## Claims
- [measured] T=5 errors are 4.4x more frequent than T=4, with an unchanged wrong-output mix. The all-negative orthant
  has a 5-10% conditional error rate (vs 0.6-0.9%), including a 2-6% floor far from any tie, and holds 48-76% of the
  wide errors.
- [measured] H2 (third-party confusion) is rejected: predicted rank >= 4 is < 1% of errors at both T.
- [measured] H1 (early positions) does not organise the new mass. The per-position error tables in the negative
  orthant vary 2-20% without a consistent early/late pattern, and configuration strata give at most 1.2-1.5x over
  the kind split (oracle). H3 (roughness) is partly right: CV rises modestly (up to 1.5x at T=5) but is not the main
  factor.
- [measured] Readout margins shrink 1.5-2.6x in every sign class at T=5, most for all-negative inputs.
- [measured] The radial identity E[F_hom] = E[F] holds to quadrature noise on 10 models. Variance ratio
  Var(F_hom)/Var(F) is 0.75-0.9.
- [measured] The frozen estimator beats sampling at H=16, T=5 on every seed and both budgets, on the leaderboard
  (see the per-seed check above) and over 32 rule seeds (5.9-14.1x), and still passes T=4.
- [inferred] The T4->T5 gain drop is mainly the error rate (gain ∝ 1/err at near-constant CV). Restoring 50-100x at
  T=5 needs an integrand with ~3x lower CV, not a better allocation.
- [speculative] The all-negative orthant degrades because its training weight halves (1/16 -> 1/32) while the
  readout must separate 5 classes with a homogeneous map. Not tested (T12 training-stage question).

## Alternatives not ruled out
1. **The error-rate law is close to definitional.** gain = (1-err)/(err CV² c) is an identity. The substantive part
   is the measured near-constancy of CV, which was measured on these 30 models only, with iid faces. The Sobol/iid
   ratio is a separate, also declining factor (ht_seeds: 6-8 at T=3 to 1.4-4.5 at T=5).
2. **The strata were chosen after looking at the test seeds** (D1c/D2 on seeds 0-4). They are model-free and have no
   fitted constants, but the choice is informed by the test set. A clean check is a held-out (H,T) or the DEV models
   at T=6. Not done. The DRI should rule whether this counts as advice.
3. **Leaderboard per-seed ratios rest on a single deterministic run per model.** The 32-rule-seed study is the
   better gain estimate. The leaderboard's 4M-sample truths (se ~5e-5) dominate the per-seed ratio at 64k; the
   own 48M truths (se ~1.4e-5) give the same pass/fail.
4. **Time ratio**: an implementation that removes the Python overhead was not written, so the inferred "overhead,
   not FLOPs" is untested beyond the profile.
5. **Mechanism of the negative-orthant failure**: margin compression is a description, not a cause. Alternatives:
   (a) training-weight explanation; (b) a specific readout direction degenerating for homogeneous negative states.
   Cheapest test: train with the all-negative orthant up-weighted (T86 training tools) and see whether the T=5 gain
   recovers.
6. The sigma = 0.02 noisy copies at T=5 fail at 16k (ratio 1.46, CI 0.62-2.30; 0.73 at 64k), so the recipe is not
   uniformly better than sampling off the seeds.

## Exhaustion checklist (for the scoped negative "reallocating the tie-face rule cannot restore T=4-level gains at T=5")
Tried:
- [x] sign strata with a deterministic Neyman pilot;
- [x] oracle config / config x sign / config x sign x u-quartile strata (upper bound, iid);
- [x] zero-variance counterfactual for the all-negative and the <=1-positive regions;
- [x] radial Rao-Blackwellization;
- [x] honest walk cost (removes the 2.23 factor);
- [x] budgets 16k and 64k.

Not tried, with the reason:
- [ ] region-targeted second slicing family (pre-reg fix 2): H2 rejected, so there is no third-party patch to target.
- [ ] weight-derived strata (pre-reg fix 3): oracle strata already bound any stratification at 2-3x.
- [ ] control variates from a cheap linearised band (1-cell proxy for F): could lower CV. **Most promising open
  variant.**
- [ ] 2-D exact slicing (polygonal pieces in a plane): removes one more dimension. Not implemented.
- [ ] exact cone enumeration in the all-negative orthant: that region carries 33-56% of the Neyman sqrt-variance
  but only bounds the gain at 15-38x even if exact.
- [ ] larger budgets (rate): ht_seeds measured only x1.5 for 4x budget at T=5.

## Failure modes
- Bulk errors (sigma = 0.02 copies, undertrained models): stratification by sign helps less, and gain ~ 1/err.
- High CPU time per FPE at small budgets (Python loop over cells).
- Sobol prefixes after Neyman allocation are not powers of two, so part of the low-discrepancy benefit is lost.
- Assumes a bias-free ReLU RNN (radial integration and sign strata rely on homogeneity / input sign).
