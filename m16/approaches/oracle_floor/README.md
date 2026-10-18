# T292: oracle bias floor, observer F and error anatomy on the 80 DEV2 retrains (Fable r2 section D)

↑ [Approaches index](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md)

## Pre-registration (copied verbatim from the DRI brief before any data, 2026-10-18)

Panel: DEV2, 80 open retrains, on branch claude/t192-gpu-cost-law under m16/approaches/gpu_cost_law/hp2/dev2/ (weights.npz, manifest.json). Its 1e8-sample truths (truths.json) are being published there now. If they are still missing when you need them, compute 2e7-sample truths yourself (SE ~5e-5) and say so.
1. **Oracle floor.** Run the E31 lin low-input closure (m16/approaches/stoch_margin/diag_cond.py) on 20k sampled skeletons x 256 low draws per model, on all 80 models. This gives a per-model aggregate bias b_m. It is diagnostic sampling only and is not an estimator.
2. **Observer.** Run the E21 observer (m16/approaches/observer/est.py) at configs (1,1,1), (1,1,2) and (2,2,2) on all 80 models. Record e_m, charged cost C_m (instrument FLOPs honestly) and F per config, plus the per-model bias against accuracy.
3. **ANAT-80.** From the same samples, split each model's error into blind / 2~3 band / 1~2 band / soft tail, by top-3 triple and x9 case. Count the "gross-error triples" (error rate > 20% at gap >= .2). Regress accuracy on these parts. The question is what makes a retrain weak.

Go lines (pre-registered):
- **G1:** RMS_m(b_m) <= .40 pp, |mean b| <= .25 pp, and the slope of b_m against accuracy <= .05 pp/pp. Then build C1 (OBS-cheap: the observer at ~5k FPE, as in Fable r2 C1, with a resolution knob).
- **G2:** any E21 config with point F <= 3 on DEV2. Then freeze it with est(model, budget) -> (estimate, cost_flops) and send it to the T281 scorer (session_01UvVWREPpEYpUGudpzkioKd) for the sealed HP2 curve.
- **Dead:** RMS_m(b_m) > .80 pp. Then the lin/observer closures are dead in the cheap regime (scoped).

Report the bias split by blind / band / tail per model, so that any failure names its part.

### Conventions fixed before data (mine)
- Scoring: F = mean_m(e_m^2) * max(C, 1000) / mean_m(p_m(1-p_m)), C in FPE = FLOPs/5948 (Directive 8 amendment 2).
- "Slope of b_m against accuracy" = OLS slope of b_m (pp) on truth accuracy (pp) over the 80 models; the G1 test
  is |slope| <= .05. Between .40 and .80 pp RMS (or G1 failing only on mean/slope) the outcome is "neither".
- Region of an input (ANAT): blind = 2nd largest <= 0; otherwise with g23 = M2-M3, g12 = M1-M2:
  2~3 band = g23 < .05 and g23 <= g12; 1~2 band = g12 < .05 and g12 < g23; soft tail = min(g12, g23) >= .05.
  Tail is also reported at >= .1 and >= .2. x9 case = position 9 is the max / the 2nd / below (as T269).
  Gross-error triple = ordered top-3 position triple whose non-blind error rate at min gap >= .2 exceeds 20%
  (triples with fewer than 30 such samples are not counted and are reported separately).
- The same region split is applied per skeleton for b_m (skeleton = top-3 values, so region is exact per skeleton).
- Truths: if DEV2 truths.json is absent, 2e7-sample numpy truths from my own ANAT pass (seed 292000 + model index).
- **Amendment (2026-10-18, before any scoring):** the DRI published the open 1e8-sample DEV2 truths
  (claude/t192-gpu-cost-law 163cd62e, hp2/dev2/truths.json, sha256 582c21cf...). All e_m, slopes and F use those.
  ANAT was then run at 2e6 samples per model (Fable C6's size) instead of 2e7, since it is no longer the truth source.

## Question
Fable r2 section D: in the cheap regime (1k-4k FPE) on retrains, is the low-input closure good enough to build a
deductive estimator on (oracle floor b_m)? Does any existing E21 observer config already pass own-cost scoring?
And what makes a retrain weak (ANAT-80)?

## Method
- `panel.py`: DEV2 loader (80 retrains, 40 `fused.py` + 40 `trainer.py`; weights hash checked against the
  manifest), target weights, forward pass. `shim/common.py`: torch and alg_zoo are **not installed** here, so
  `m16/common.py` is replaced by a shim that loads the target weights from claude/t241-neuron-account
  `neuron_account/m16_weights.npz`. Only numpy/scipy are installed (fresh `.venv`).
- `oracle.py` (**diagnostic sampling, inductive; not an estimator**): the E31 `lin` closure, code copied from
  `stoch_margin/diag_cond.py` (same skeleton law, same conditional-mean anchor, same eps=1e-6 Jacobian). Per model
  20k skeletons x 256 low draws; b_m = mean over skeletons of q_lin(S) - q(S), paired on the same draws. The `R0`
  closure (low inputs pinned at their conditional mean) comes for free and is also reported (not pre-registered).
  add/gauss of diag_cond were dropped (not needed for b_m).
- `check_lin.py` (diagnostic): an independent re-implementation of lin with its own sampler and seed, and the exact
  gate-pattern map (`observer/core.logit_map`) instead of finite differences. It also measures how many low draws
  stay inside the anchor's gate cone.
- `obs.py` (**deductive**): E21 `observer/est.py` `make(nv, ng, q, nq=12, noise=1, ver=4)` is used unchanged for the
  estimate. The cost is re-instrumented op by op (docstring): 72,800 FLOPs per anchor (12.24 FPE; est.py charges
  10.5), quadrature 41 FLOPs per (kind, triple, LHS row, node), and 10 FPE for the blind probe. The steps are all
  deductive: the anchors are constructed points, and the LHS bin permutation uses a fixed seed and touches no model.
- `anat.py` (**diagnostic sampling**): 2e6 inputs per model (seed 292000 + index, target = index 0); error tallies
  by region x x9 case x ordered top-3 triple, min-gap bin x triple, and the rank of the chosen input.
- `analyze.py` -> `results/summary.json`, `results/per_model.json`; per-model table `results/per_model.md`.

## Status
**Done.** Pre-registered outcome: **DEAD** (lin oracle floor RMS 4.77 pp >> .80). **G2 fails** (best F = 238).
G1 fails on all three criteria. So C1 is not built.

## Results
Reproduce (from this folder, venv active, `OMP_NUM_THREADS=1`): `python oracle.py 20000 256`, `python anat.py 2e6`,
`python obs.py 1 1 1` (and `1 1 2`, `2 2 2`), then `python analyze.py ../../../data/oracle_floor/dev2_truths.json`.
DEV2 truths: mean .9201, sd .0316, range .8136-.9602, mean p(1-p) = .0725.

**Sanity checks against the target** [measured]: lin on the target gives -0.454 +- .062 pp (E31's
diag_cond_orig: -0.453). Observer (1,1,2) on the target gives err23 3.69%, err12 1.00%, blind .60 and bias -.66 pp
(E21 README: identical). ANAT on the target gives a soft tail of .66 pp at gap >= .05 (T269: .66).

### 1. Oracle floor (lin closure), 80 retrains
| | value | pre-registered bar |
|---|---|---|
| RMS_m(b_m) | **4.77 pp** (noise-corrected 4.76; mean per-model SE .08) | G1 <= .40, Dead > .80 |
| mean b | **-3.38 pp** (range -11.15 to -0.05; every model negative) | G1 \|mean\| <= .25 |
| slope of b on accuracy | **+0.27 pp/pp** | G1 <= .05 |
| models with \|b\| <= .40 / > .80 | 8 / 59 | |

Bias by region (pp, mean / RMS over models): blind +.002 / .01; 2~3 band -.52 / .73; 1~2 band -.31 / .41;
**soft tail -2.54 / 3.66**. The failure names its part: **the soft tail** (the tail part alone has RMS^2 = 59% of the total;
parts covary), with both bands also over .4 RMS on their own. The bias is continuous, not a few outliers (sorted
values in the per-model table). It is **not** mainly a weak-model effect: fused_s33 and torch_s19 (acc .959/.960,
tail error .3-.4 pp) have b = -8.8 / -10.7 pp. The target (-.45) sits at the good end, so E31's small aggregate on
3 models was a lucky draw, as Fable r2 B "against (2)" suspected.

Independent check (`check_lin.py`, 2000 skeletons, different code and seed) [measured]: target -.88 +- .22,
fused_s0 -8.13 +- .33 (oracle.py -7.80 +- .10), torch_s39 -10.15 +- .34 (-10.13 +- .10). **Only 4-12% of low draws lie
in the anchor's gate cone**, so the linear map is extrapolated across ReLU kinks for ~90% of draws. lin invents
errors: in the tail its bias averages -1.7x the model's own tail error.

Not pre-registered: **R0** (the low inputs pinned at the conditional mean, one deterministic point per skeleton)
has RMS_m .51 pp (noise-corrected .49), mean +.22 pp, slope -.046 pp/pp, max |b| 1.93 pp. That is between the G1
and Dead lines. See Claims 4.

### 2. Observer (E21 v4), 80 retrains, 1e8 truths
| config (nv,ng,q) | RMS e | mean e | sd e | slope e on acc | C (FPE, my count / est.py) | sampling RMS at C | **F** (my C / est.py C) |
|---|---|---|---|---|---|---|---|
| (1,1,1) | 1.18 pp | -.88 | .80 | +.056 pp/pp | 122,911 / 106,489 | .077 pp | **238** / 206 |
| (1,1,2) | 1.01 pp | -.70 | .73 | +.070 | 245,811 / 214,628 | .054 | **346** / 302 |
| (2,2,2) | .63 pp | -.31 | .55 | +.093 | 880,302 / 758,948 | .029 | **476** / 411 |

G2 (F <= 3) fails by about 80x at the best config. RMS on the stronger / weaker half of the panel: (1,1,1)
.89/1.42, (1,1,2) .72/1.24, (2,2,2) .37/.80 pp. The bias grows as accuracy falls (slope .06-.09, but below
Fable's 10% stop slope). corr(e, error) = -.22/-.30/-.54. Even at zero bias, F could not reach 3: e.g. (1,1,1) at
123k FPE needs RMS <= .042 pp.

### 3. ANAT-80: what makes a retrain weak
Mean error by part (pp, mean / sd over the 80 models): blind .73 / .14; 2~3 band 3.24 / .74; 1~2 band 1.54 / .65;
**soft tail (min gap >= .05) 2.48 / 1.94**. tail >= .1: 1.28 / 1.24; tail >= .2: .55 / .64.
The cross-model variance of the error decomposes exactly into the parts (OLS of accuracy on the four parts has
coefficients -0.98..-1.01, as it must). Shares of cross-model variance: **tail 60%**, 2~3 band 22%, 1~2 band 17%,
blind 1%. (tail >= .1 alone: 36%; >= .2: 17%.)
- By x9 case: x9 below the top 2 carries 5.54 pp mean and 60% of the variance; x9 max 1.76 pp and 34%; x9 2nd .69 pp and 6%.
- By rank of the wrong answer: rank 3 5.05 pp, rank 1 2.06 pp, **rank 4+ .88 pp** (9% of the variance; the
  observer does not model these).
- **Gross-error triples** (error > 20% at min gap >= .2, every triple had >= 30 samples): mean 10.9, median 5,
  max 64; 66/80 models have at least one. corr with the error is .80, but they hold only .33 pp of error on average
  and 13% of the variance.
- Concentration: in the weaker half, half of each model's tail error above the panel-median triple profile sits
  in a median of **44 triples (IQR 33-58)**; in the stronger half, 15.
- **Headline:** a retrain is weak through a diffuse, model-specific widening of the soft tail over tens of
  triples (mostly with x9 below the top 2), plus a smaller widening of both bands. It is not weak through a few
  gross-error triples, and not through the blind region (nearly constant). Per Fable C6 this rules out
  "C1-plus-census" (a 720-point census cannot hold the weak part) and leaves only a band-and-tail-resolving method,
  which is exactly what the dead lin closure was supposed to supply.

## Compute accounting
- Oracle: 81 models x 20k x 256 = 5.1M forward passes per model, plus 8 anchor passes per skeleton. Diagnostic
  sampling, about 22 CPU-min. `check_lin.py`: 3 x 512k passes.
- ANAT: 81 x 2e6 forward passes, about 11 CPU-min.
- Observer: charged C in the table (FLOPs counted op by op; model-free precomputation of tie levels, grid and bin
  moments is uncharged, per LB3 convention). Wall time 1-6 s per model.
- Total about 50 CPU-min, single process, < 1 GB.

## Claims
1. [measured] The E31 lin low-input closure has an aggregate oracle bias with RMS 4.77 pp over the 80 DEV2 retrains
   (mean -3.38, slope +.27 pp/pp). By the pre-registered line it is **dead in the cheap regime (scoped to lin closures
   at the conditional-mean anchor)**. Reproduced by independent code on 3 models.
2. [measured] The lin failure is mostly in the soft tail (-2.54 pp mean), and ~90% of low draws leave the anchor's
   gate cone.
3. [measured] The E21 observer has F = 238 / 346 / 476 at (1,1,1) / (1,1,2) / (2,2,2) on DEV2. G2 fails. Its RMS
   (.63-1.18 pp) is about twice as large on the weaker half.
4. [measured, not pre-registered] The R0 point closure (low inputs at their conditional mean) has an aggregate
   oracle bias with RMS .51 pp on DEV2. [inferred] That is far better than lin, partly because R0 makes errors
   only where the anchor itself is wrong, and the over- and under-shoots cancel across skeletons. [speculative] A
   deterministic skeleton integral of R0 would have a ~.5 pp floor, which by itself fails the strict 1k bar
   (.86 pp is the total RMS budget, so it may fit at 1k only if the integration error is < ~.7 pp). Its
   per-skeleton output is 0/1, so a deterministic 3-D integral per triple needs the decision boundary, i.e. many
   evaluations. This is a new backlog item, not a result.
5. [measured] ANAT-80: weak retrains differ mainly in the soft tail (60% of the cross-model error variance),
   spread over tens of triples. Gross-error triples correlate with weakness (r = .80) but hold only 13% of the
   variance. The blind region is nearly constant (sd .14 pp).

## Alternatives not ruled out
- **Anchor choice.** lin is linearised at the conditional mean of the low inputs. The anchor's gate pattern
  covers only 4-12% of draws. A lin at the most probable gate pattern, a piecewise-lin closure over several low-input
  anchors, or the observer's cone clipping applied to E31's closure were not tried. The "dead" verdict is scoped to
  the pre-registered closure. The cone clipping is the E21 fix, and E21 at (2,2,2) reaches .63 pp, but at 880k FPE.
- **R0's .51 pp may be partly a lucky cancellation**, like E31's lin was on 3 models. It is not tested on a second
  panel. Its per-region split was not computed.
- **Finite skeleton sample.** The SE of b_m is .08 pp, which is negligible against 4.77.
- **ANAT categories are mine.** Ties between the two gaps go to the 2~3 band, and the .05 / .2 thresholds are T269's.
  Another split (e.g. by g23 only) would move mass between band and tail, but not the 60% vs 1% blind contrast.
- **The observer cost count is mine** (12.24 FPE per anchor vs est.py's 10.5). F is reported both ways, and the
  verdict does not depend on it.

## Failure modes
- Linearising in the low inputs at one anchor fails whenever the low inputs routinely switch gates. On retrains
  this is the norm (only 4-12% of draws keep the anchor's pattern), not the exception.
- Observer-class methods cost >= 1e5 FPE because they resolve 720 triples x cells x LHS rows. At that cost the
  sampling bar is <= .08 pp, which no observer config approaches.

## Suggestions (for the DRI; not edited elsewhere)
- Backlog: **R0-floor check** on a second panel (HP1/DEV seeds) + region split; if <= .5 pp holds, design a
  deterministic boundary-finding skeleton integral for R0 (cost per triple?) and see if it fits 4k FPE.
- Fable r2 odds: G1 failed -> per Fable E, cheap-regime deductive odds < 3%; C1 not funded; spend on T203 accounting.

---

# T296: the R0 point closure (follow-up of T292 Claims 4)

## Stage A pre-registration (written 2026-10-18, before any LB3 R0 data)
DRI brief (verbatim): "Reproduce R0's oracle floor on a second panel: the LB3 models (orig, seeds 0-4, the noisy
groups). HP1/HP2 stay sealed. Split the bias by region (blind / 2~3 / 1~2 / tail), and check it is not lucky
cancellation as E31's was. Go if RMS <= .55 pp, no group mean beyond +-.3 pp, and every region's |bias| <= half the
total RMS. Also report the per-region RMS on DEV2, so we know which part the cancellation would have to come from."

My operationalisation (fixed now):
- Panel: all 30 LB3 test models (m16/leaderboard.py `models()`): orig, seed0-4, s{.0025,.005,.01,.02}_{0..5}
  (noise seed 0), truths from results/leaderboard_truths.json (4e6 samples). Seeds come from the T241 branch's
  zoo16_10_weights.npz (alg_zoo is not installed). Each one is checked against the LB3 truth with 2e6 samples (must be
  within 4 SE), and so is every noisy copy.
- R0 bias b_m = mean over 20k skeletons of [R0 correct] - q(S), with q(S) from 256 low draws (oracle.py, unchanged).
- **Go** iff all three hold over the 30 models: (i) RMS_m(b_m) <= .55 pp; (ii) for each of the 6 groups (orig as a
  group of 1, seed, 4 noise levels), |mean b| <= .30 pp; (iii) for each region r, |mean_m b_{m,r}| <= RMS_m(b_m)/2.
  Secondary (no gate): per-region RMS_m(b_{m,r}) on LB3 and DEV2; the same three tests restricted to the
  retrain-like subset (orig, seeds, s.0025, s.005; 18 models) are reported separately, because the s.01/.02
  copies (45-89% accurate) are outside the target regime. The verdict uses all 30.
- Label: R0 is a **declared structural approximation** (deductive under Hilton's "approximations are fine"). The
  oracle here is diagnostic sampling. Stage B's skeleton integral (if go) is exact reasoning.

## Stage A results (2026-10-18)
Reproduce: `python check_lb3.py` (all 30 reconstructed LB3 models match the LB3 truths, |z| <= 2.5, 0 mismatches);
`PANEL=lb3 python oracle.py 20000 256`; `python r0_split.py lb3` and `python r0_split.py dev2`
(-> results/r0_lb3.json, results/r0_dev2.json). Diagnostic sampling only; about 8 CPU-min.

**Outcome: NO-GO** (pre-registered rule, all 30 LB3 models): (i) RMS 1.39 pp > .55 FAIL; (ii) group means orig +.04,
seed +.17, s.0025 +.27, **s.005 +.55, s.01 +1.39, s.02 +2.51** pp FAIL; (iii) region means blind +.01, 2~3 +.28,
1~2 +.04, tail +.64 against a limit of .69: pass. The retrain-like subset (orig + seeds + s.0025 + s.005, 18 models)
has RMS .53 (pass) but fails (ii) on s.005 (+.55). The 5 zoo seeds alone have RMS .61, mean +.17 (seed1 +1.33).

| region | DEV2 mean / RMS (pp) | LB3-30 mean / RMS | LB3 retrain-like mean / RMS |
|---|---|---|---|
| blind | +.04 / .05 | +.01 / .02 | +.01 / .02 |
| 2~3 band | +.11 / .20 | +.28 / .33 | +.19 / .23 |
| 1~2 band | +.01 / .10 | +.04 / .12 | -.02 / .09 |
| soft tail | +.05 / .32 | +.64 / 1.01 | +.15 / .31 |
| total | +.22 / .51 | +.98 / 1.39 | +.32 / .53 |

Reading [measured unless marked]: R0's bias is **positive** (it overestimates accuracy) and grows with the soft-tail
error. On LB3 it rises monotonically with the noise level. On DEV2 corr(R0 b, tail error) = .37, slope +.09 pp per
pp of tail error, and the RMS on the weak half is .64 vs .33 on the strong half. [inferred] R0 misses part of the
tail errors (where the low inputs, not the skeleton, decide), and that part is exactly what grows on weak or noisy
models (ANAT-80). On DEV2 the regions do not cancel each other (every region mean <= .11 pp). The .51 pp there was
small because DEV2's tail is small enough, not because of cross-region cancellation, but the 2~3 band (+.11 to +.28)
is a second, systematic positive part on every panel.

Claims: [measured] the R0 oracle floor does not reproduce on the LB3 panel (RMS 1.39 pp; noisy groups +.55 to
+2.5 pp); on retrains only (DEV2 80 + LB3 seeds 5) it is .51 / .61 pp with a positive bias that tracks the tail.
[inferred] As a floor for a 1k-4k FPE estimator it leaves no room: the strict 1k bar is .86 pp *total* on retrains,
and R0 alone uses .5-.6 of it before any integration error. Stage B not run (pre-registered).

Alternatives not ruled out: an R0 variant with the low inputs at another point (e.g. conditional median, or two
points per skeleton averaged) could shift the positive bias; not tried (it would be a new tuned closure, not this
pre-registered check). The noisy groups are outside the HP2 retrain regime. A retrain-only rule would have been
borderline (DEV2 .51, seeds .61, group means <= .28 on DEV2 trainers), but that rule was not pre-registered.
