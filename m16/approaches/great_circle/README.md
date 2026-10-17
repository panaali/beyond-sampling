# Great-circle exact integration (directional simulation lifted to 2-D)

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [STATUS](../../../STATUS.md)

## Question
The model is positively homogeneous, so accuracy is the correct fraction of the sphere S^9. If we
integrate exactly around random great circles (exact piecewise-sinusoidal walk), is the estimate
better than plain Monte Carlo per unit compute? The literature note `lit/domain_engineering_finance_cs.md`
(item 2) predicted 5-20x. Does stacking variance reduction on top (orthogonal frames, control variate)
make it competitive with `m16/stratified.py`?

## Method
- `gcirc.py: walk()`: for a batch of planes (u, v), x(th) = cos th u + sin th v. In a fixed gate pattern
  every pre-activation, logit and input is a cos th + b sin th, with zeros at atan2(b, a) + pi/2 + k pi.
  The walk goes piece by piece over [0, 2pi). Each piece takes one pass that carries the (cos, sin)
  coefficient vectors. Inside a piece it solves all 45 logit-pair and 45 input-pair crossings, then tests
  argmax == 2nd-largest input at each sub-interval midpoint and adds up the arc length. The result is exact.
- Unbiasedness: for a uniformly random plane and any fixed th, x(th)/|x| is uniform on S^9. So
  E[arc fraction] = accuracy. This holds for any arc length, not only the full circle.
- `frame_planes()`: one Haar orthogonal 10x10 frame gives 5 mutually orthogonal planes.
- Control variate (CV): the fraction of each circle in each `stratified.py` stratum (blind, and bins
  of the relative top gap). These are model-free and computed on a 512-point grid. Each has a known
  mean (the stratum probability P_h, by rotation invariance). The coefficients were fitted once on an
  independent pilot of the original model (`pilot_0.npz`) and then frozen, so the estimator is unbiased
  for every model.
- `covfun.py`: autocovariance C(D) of the correct-indicator along circles. It gives the efficiency of
  exact integration over an arc of any length alpha.
- `estimator.py`: leaderboard interface (frames + CV, `DEDUCTIVE = False`).

## Status
**Done. Negative result.** Great-circle integration is unbiased but **3-10x worse in MSE than plain
sampling per unit compute**, even with frames and the CV, and worse still against stratified sampling.
Not worth pursuing further in this form.

## Results
All commands run from `/home/user/clean-neurons` with the venv active.

1. **Exactness** [measured]: on 20 circles, the walker matches a 200k-point theta grid to within 1.5e-5 (the grid's resolution).
   `python m16/approaches/great_circle/validate.py`
2. **Cost and variance per circle (original model, 2000 circles)** [measured]:
   `python m16/approaches/great_circle/pilot.py 0`
   - pieces per circle **250** (about 40 per radian, one per 1.4 degrees). With partial recompute, a
     piece costs on average 0.53 of a full pass (the flip happens on average around step 4.7).
   - variance per circle 1.3-1.5e-3, against Bernoulli 4.46e-2. That is a **34x variance reduction for
     250x the cost**, so the MSE gain over sampling is **0.12-0.14x** (1 pass per piece).
   - Orthogonal frames gain x1.3 over independent planes.
   - Heavy tail: circles with fraction < 0.85 contribute 49% of the variance.
3. **Why it fails: correlation length vs piece length** [measured]: `python m16/approaches/great_circle/covfun.py`.
   The indicator's correlation along a circle is 0.56 at 1 degree, 0.27 at 5, 0.08 at 20 and about 0 at 45 or more.
   It is also about 0 at 90 and 180 degrees, so antithetic x/-x and orthogonal points give no gain.
   The efficiency of exact arc integration over sampling, by arc length:
   0.2 degrees 0.89, 1.4 degrees 0.67, 11 degrees 0.29, 90 degrees 0.15, 360 degrees 0.13.
   It **falls monotonically**, so no arc length beats plain sampling. The gates switch about 5-10 times
   faster than the correct/incorrect indicator decorrelates, so most of the exactness is paid for and not used.
4. **ARC protocol** [measured]: original model plus `noisy_models(6, .0025)` and `(6, .005)`. Truth from
   `true_acc(n=4M)`. Pools of 2000 circles (models 0, 1, 7, 8) or 800 circles (the rest).
   `python m16/approaches/great_circle/pool.py <idx> [n_frames]` then `python m16/approaches/great_circle/analyze.py`
   (output in `results.json`). The table gives the MSE gain over sampling at equal compute, counting 1 pass per piece:

   | models | stratified | GC independent | GC frames | GC frames+CV | GC frames+CV, partial-recompute cost |
   |---|---|---|---|---|---|
   | original | 2.12 | 0.117 | 0.163 | 0.339 | 0.64 |
   | sigma .0025 (6) | 1.41-1.98 | 0.13-0.15 | 0.11-0.19 | 0.24-0.30 | 0.45-0.57 |
   | sigma .005 (6) | 0.98-1.85 | 0.09-0.14 | 0.11-0.18 | 0.12-0.29 | 0.23-0.55 |

   RMSE in pp at budgets 16k / 64k / 256k:

   | models | sampling | stratified | GC frames+CV |
   |---|---|---|---|
   | original | .167 / .084 / .042 | .115 / .057 / .029 | .287 / .143 / .072 |
   | sigma .0025 | .193 / .097 / .048 | .146 / .073 / .037 | .371 / .185 / .093 |
   | sigma .005 | .238 / .119 / .059 | .208 / .104 / .052 | .560 / .280 / .140 |

   The RMSEs are predicted from the per-model pool variances as var/n. A disjoint-block empirical check
   (226 / 56 / 4 blocks) gives 0.50 / 0.20 / 0.03 pp against predicted 0.46 / 0.23 / 0.12 pp, which is
   consistent given how few blocks there are at 256k. The sampling and stratified figures are analytic
   (p(1-p)/b, and the sum of P_h^2 s_h^2 / n_h with s_h from a 20k-per-stratum pilot).
5. **Unbiasedness** [measured]: the per-model z-scores of the pool mean against truth range from -0.93 to
   +0.89 without the CV and from -1.39 to +0.73 with it. With 13 models there is no sign of bias.

## Compute accounting
- In this README, 1 piece = 1 forward pass, as with lineint. That flatters the method. A piece really
  runs the recurrence on 2 coefficient vectors (2 matvecs per step) and then does root-solving and
  midpoint argmax checks on up to 181 sub-intervals (about 7e3 FLOPs, about 1 pass). `estimator.py`
  therefore charges **3 passes per piece** for the leaderboard. Under that charge every GC gain above
  is divided by 3: frames+CV is **0.04-0.11x** sampling.
- The "partial" column assumes an incremental implementation that recomputes only steps t_flip..9.
  That implementation is not written; `walk()` recomputes all 10 steps.
- Not counted: model-free work (the 512-point stratum grid per circle, the P_h from `stratified_design.json`).
  Counted once, not per model: CV coefficients fitted on 2000 original-model circles (about 500k pieces).
- Wall time: about 60 ms per circle with N=2000 batched, on a heavily shared CPU.

## Claims
- [measured] The walker is exact (grid check) and the estimator is unbiased (13 models).
- [measured] A random great circle has about 250 linear pieces, and its exact correct-fraction has 30-35x lower variance than one Bernoulli sample.
- [measured] Exact arc integration is less efficient than sampling at every arc length (covariance analysis, original model).
- [measured] Best variant (frames + stratum CV) reaches 0.12-0.34x sampling MSE at equal piece count, and 0.04-0.11x under the FLOP-honest 3x charge. Stratified sampling gets 1-2.1x.
- [inferred] The cause is that gate pieces (about 1.4 degrees) are much shorter than the indicator's correlation length (about 5-20 degrees), so exactness inside correlated spans is wasted.
- [speculative] No 2-D or 1-D exact-integration scheme driven by full gate walking can beat sampling on this model unless the cost per piece drops by about 10x or pieces can be skipped (certified-constant arcs).

## Alternatives not ruled out
For the main claim (GC is inherently worse than sampling here):
1. *The implementation is just too expensive per piece.* Partially tested: even at 0.53 passes per piece
   (ideal incremental recompute) frames+CV reaches only 0.23-0.64x. A rank-1 update scheme could cut the
   cost further, but it would need about 2-4x more just to break even and 4-8x to match stratified.
2. *The variance comes from a fixable heavy tail* (circles through the blind region). Partially tested:
   the stratum CV removes 1.5-3x of the circle variance and frames another 1.3x, which is not enough.
   Not tried: a richer CV (a surrogate error probability per point, using position triples), and
   stratifying planes by their blind-region exposure.
3. *A different curve or plane distribution would cross fewer gates per unit of decorrelation.* Not
   tested. Coordinate lines (lineint) were similar: 33 pieces for a 2x variance cut.
4. Not tried: RQMC over the plane choice. FINDINGS reports RQMC gave nothing on lines, and with
   correlation near zero beyond 45 degrees there is little smooth structure for it to use.

## Failure modes
- Cost grows with gate activity: noisier models keep about 250 pieces but have more variance, so the gain shrinks with sigma.
- `max_pieces=5000` truncation (never hit). Eps-based gate evaluation at breakpoints could misclassify a piece shorter than 1e-10 rad (negligible).
- The CV coefficients were tuned on the original model. Applied to noisy copies they stay unbiased but give less gain (sigma .005 model 8: x1.1).

## Files
`gcirc.py` (walker, plane samplers), `validate.py`, `pilot.py`, `covfun.py`, `cvtest.py`, `pool.py`,
`analyze.py`, `estimator.py`, `results.json`, `pool_*.npz`, `pilot_0.npz`, `cov.npy`, `pool_*.log`.
