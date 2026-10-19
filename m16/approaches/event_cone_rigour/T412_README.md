# T412: fast rigorous M5t (`m5t_fast.py`)

↑ [event_cone_rigour README](README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md)

Owner: Claude child session_01333jBMKr2Pn3Hfse5vLAUW. Branch `claude/t412-m5t-fast` (off `claude/t400-bound-rigour`;
T400 merges). New files only; `rigour.py` is not edited.

## Question
Can the per-simplex M5t bound `rigour._bounds_tangent` (arb, ~36 ms per simplex) be replaced by a batched float
computation in which every rounding error is bounded and rounded outward, at >= 100x the speed and with
essentially the same tightness?

## Pre-registration (pushed before any timing or tightness number)
Reference: `rigour._bounds_tangent` in arb (python-flint), as called by `rigour.mass_cell(..., tangent=True,
zbins=24, zratio=23/20)` (the `a_m5.py ... t` settings).

Test set: every simplex that `mass_cell(..., tangent=True)` evaluates on the 27 k = 4 cells (t0 = 2.25) and the
28 k = 4 cells (t0 = 2.5) of `data/bench_182_k4.npz` (cells with >= 30 MC hits, as in `a_m5.py`).

PASS requires all three:
1. **Speed:** >= 100x per simplex against arb `_bounds_tangent` (wall time per simplex, same machine, single core,
   fast version batched over the simplices of a cell).
2. **Tightness:** fast hi <= 1.01 x arb hi on >= 99% of the test simplices.
3. **Soundness:** 0 violations of (a) fast hi >= arb lo and fast lo <= arb hi on every test simplex, and (b)
   end-to-end `mass_cell` with the fast bounds gives ub >= MC - 4 s.e. on every cell (MC and hits from
   `results_a_m5t_*.json`; s.e. = MC / sqrt(hits)). Also reported, not part of the line: the fraction of
   simplices with fast hi >= arb hi (1 - 1e-12); any exception is explained.

Then (reported, no line): time the 10-D region 3 (`data/regions_1-8-2.npz` index 3, event=True, t0 = 1.5) with
the fast bounds against the arb reference (374 s, ub 6.19e-11).

Soundness comes from the error analysis below; the tests only check it.

## Method
Files (all new; `rigour.py` untouched):
- `m5t_fast.py`: `bounds_batch(Mlo, Mhi, vlo, vhi, P)` gives the rigorous per-simplex [lo, hi] (the quantity bounded
  by `rigour._bounds_tangent`), batched over simplices. `mass_cell_fast(tri, gates, k, t0, ...)` is
  `rigour.mass_cell(..., tangent=True)` with these bounds: the same exact geometry (code copied as
  `cell_geometry`), the same longest-edge bisection, but done in rounds (each round splits the top
  max(8, count/16) simplices by hi - lo).
- `t412_record.py`: re-runs arb M5t on the bench cells and records every simplex it evaluates (float enclosures of
  the exact vertices and volume, arb lo/hi, arb time). It wraps `_bounds_tangent`; nothing in `rigour.py` changes.
- `t412_test.py`: the tests and timings (commands under Results).

Pipeline per simplex (same construction as `_bounds_tangent`, each step bounded outward):
1. Tangent centre m_c: the min-norm point of the simplex (Wolfe's algorithm in float; rigour uses NNLS for the
   same proposal). m_c is then an exact float vector, so s = s_c + tau + |m - m_c|^2 holds exactly. Enclosures of
   s_c = |m_c|^2, tau_i = 2 m_c.(M_i - m_c), R^2, s_min (separating hyperplane through m_c, and >= 1/2), s_max,
   y_p3 at the vertices.
2. z bins: the float edges e_i = fl(t0 (23/20)^i), with e_0 = t0 exactly. Any partition of [t0, inf) is valid. These
   edges differ from arb's rational ones by <= 1 ulp.
3. Per (simplex, bin): the exponent constants (split point T0, lambda bounds, tangent / chord of log Phi, the
   constant sup / inf of Phi^m) and the z-integral G in closed form. These are numba scalar code; only log_ndtr and
   erfcx are scipy (vectorised).
4. E_exp (mean of exp(-u) over the simplex, u affine) is evaluated exactly only where it matters. First every bin
   gets the trivial bound; then the bins with the largest upper terms get the exact bound, until the largest
   remaining trivial term is <= 1e-8 x the current total. In each exact bin both arb variants are evaluated
   (tangent/chord and constant Phi^m) and the better is kept; a variant is skipped when a cheap bound shows it
   cannot win. These choices only decide tightness: every bin term is a valid bound either way.
5. Sum over bins (directed), plus the last bin [e_23, inf), times c0 vol.

## Error analysis (each bound used, with its derivation)
**E0 Number model.** Every float is an exact rational. +, -, *, /, sqrt are IEEE double round-to-nearest, so
|fl(x op y) - (x op y)| <= ulp/2 (subnormals included).
- *Outward step.* up(x) = (x + |x| 2^-52) + 2^-1074, and dn symmetrically. For normal x, |x| 2^-52 lies in
  [ulp(x), 2 ulp(x)), so x + |x| 2^-52 rounds to at least the next float above x. This also holds at a binade
  boundary and for x < 0 (the step is then >= the spacing towards 0). For x = 0 or subnormal x, the + 2^-1074 is
  exact and gives the next float. So up(fl(a op b)) >= a op b. Checked against np.nextafter on 1e6 values over
  [1e-300, 1e300] and the special cases: never below nextafter, at most 2 ulp.
- *gamma_n.* A sum or product of n nonnegative floats evaluated in any order equals the exact value times a
  product of <= n factors (1 + delta), |delta| <= u = 2^-53. So it lies within (1 +- gamma_n), gamma_n = n u /
  (1 - n u). A mixed-sign dot product has |error| <= gamma_n sum |terms|, used with SAFE = 2 as in verify.py.
- *Transcendental functions (assumption A1).* exp and log (numpy and numba/libm) are assumed within REL_T = 1e-13
  relative; exp also within 4 x 2^-1074 absolute, for subnormal results. erfcx is assumed within 1e-13 relative.
  scipy log_ndtr is assumed within 1e-13 (1 + x^2) absolute: its argument scaling x/sqrt 2 makes the error grow
  like x^2 u for x << 0. This is the only non-proved ingredient. The worst cases measured against arb (200-bit,
  2e4 to 4e4 points each, `python t412_test.py fns`) are below. Every bound that uses these functions
  multiplies by (1 +- 2 REL_T) or adds the absolute term, rounded outward.

| function | assumed | measured worst | margin |
|---|---|---|---|
| exp (numpy) | 1e-13 rel + 4·2^-1074 | 1.2e-16 rel (normal range) | ~800x |
| exp (numba, libm) | 1e-13 rel + 4·2^-1074 | see results_t412_fns.json | |
| log (numpy, numba) | 1e-13 rel | 1.0e-16 | ~1000x |
| erfcx (scipy) | 1e-13 rel | 7.9e-16 (T417: 8.3e-16) | ~120x |
| log_ndtr (scipy) | 1e-13 (1+x²) abs | 4.0e-16 (1+x²) | ~250x |

**E1 Inputs.** `q2iv` rounds an exact rational to floats lo <= x <= hi, checked by exact comparison.
- Vertices are carried as enclosures [Mlo, Mhi] of exact rational vertices. A bisection midpoint is exact in the
  real geometry (children partition the parent exactly); its enclosure is [dn(dn(lo_i + lo_j) 0.5),
  up(up(hi_i + hi_j) 0.5)].
- The root volumes are the exact fmpq determinants / n!, rounded outward. A child volume is the parent's / 2,
  rounded outward.

**E2 Centre and quadratic pieces.** m_c is any float vector (exact).
- m_c.M_i: midpoint-radius dot product, radius SAFE (sum |c| rad + gamma_{n+2}(sum |c mid| + sum |c| rad)),
  then dn / up.
- tau_i = 2 (m_c.M_i - s_c): interval subtraction.
- R^2 and s_max: sum of squares of max(|lo|, |hi|) times (1 + SAFE gamma_{d+1}), then up.
- s_min: L = min_i lower(m_c.M_i). If L > 0, every point m of the simplex has |m| >= m_c.m / |m_c| >= L / |m_c|,
  so s_min = dn(dn(L^2) / upper(|m_c|^2)). Also s >= 1/2 on the slice (m_p2 = 1 + m_p3).

**E3 lambda and the split.** T0 = min_i lower(tau_i) is an exact float, so tsh_i = tau_i - T0 >= 0, with enclosure
[max(0, dn(tau_lo - T0)), up(tau_hi - T0)]. For lambda = z^2/2 in [llo, lhi] (llo = dn(A^2)/2, lhi = up(B^2)/2):
-lambda T0 <= lhi |T0| if T0 < 0 and <= -llo T0 otherwise (lower bound symmetric). -lambda tsh <= -llo tsh
(upper) and >= -lhi tsh (lower), exactly as in `_bounds_tangent`.

**E4 log Phi bounds with float slopes.**
- *Tangent (upper).* For concave f = log Phi and any slope r~: f(x) <= f(x0) + r~(x - x0) + |f'(x0) - r~| |x - x0|.
  x0 = mid(z) * y_p3(m_c) is any float. f(x0) comes from log_ndtr with its A1 error. f'(x0) = exp(-x0^2/2 -
  log sqrt(2 pi) - f(x0)) is enclosed, r~ is a float inside that enclosure, and |x - x0| <= Wx = max(xh - x0,
  x0 - xl) over the domain [xl, xh] of z y_p3 (computed outward). So cu includes mlow |f'(x0) - r~| Wx.
- *Chord (lower).* L1 <= f(xl) and L2 <= f(xh); the chord through (xl, L1), (xh, L2) lies below f on [xl, xh] by
  concavity. Its slope beta' has an enclosure [bl, bh]. The slope used is bt = max(0, bl) (flushed to 0 below
  1e-200), and the line moves down by max(bh - bt, bt - bl) (xh - xl).
- *z y_p3 with z in [A, B].* For beta >= 0: z y <= B y + (B - A) max(0, -y_lo) and z y >= A y - (B - A)
  max(0, -y_lo), as in `_bounds_tangent`. A and B are exact floats here.
- *Constant variant.* sup Phi(z y)^m = Phi(xh)^m and inf Phi(z y)^m = Phi(xl)^m (Phi increasing), used through
  log_ndtr. Products kappa y_p3 with kappa >= 0 take the end of kappa matching the sign of y.

**E5 E_exp = E_unif exp(-u), u affine with vertex values t_i.** E decreases in every t_i, so an upper bound uses
the lower ends of the knot enclosures and a lower bound the upper ends. Write E = e^-T F(s), T = max t, s_i =
T - t_i >= 0 (rounded up for an upper bound and down, clamped at 0, for a lower; an exact 0 stays 0), w = max s.
Then F(s) = n! e[s_0..s_n] = n! sum_k h_k(s) / (n+k)! (Hermite-Genocchi; Dirichlet moments): every term is >= 0.
- (a) *Series, w <= 24.* H_k^(j) = c_k h_k(s_0..s_j), c_k = n!/(n+k)!, from H_k^(j) = H_k^(j-1) + s_j H_{k-1}^(j) *
  fl(1/(n+k)). The rounding chain is delta(k,j) <= max(delta(k,j-1), delta(k-1,j)+3) + 1 <= 4k + j + 1 (the
  reciprocal's own rounding included). The sum over k adds K, so D = 6K + n + 2 covers the whole computation,
  so S_fl lies in S_exact (1 +- gamma_D); the code's factors (1 +- 2 gamma_D) also give S_exact from S_fl.
  - Tail: h_k(s) <= C(n+k, k) w^k, so c_k h_k <= w^k / k!, and sum_{k>K} <= t_{K+1} / (1 - w/(K+2)) <= 2 t_{K+1}
    when K + 2 >= 2w. t_k is propagated upward (each step times 1 + 2^-50, which covers 3 roundings). The loop
    stops when also 2 t_{K+1} <= 2^-60 S; the upper bound adds 4 t_{K+1}.
  - Underflow: each op adds at most 2^-1075 absolute. A perturbation of H_k'^(j') reaches F with coefficient
    <= sum_m w^m / m! <= e^w (because (n+k')! (n+k-k')! <= (n+k)! n!). With <= 4 (n+1) K ops, w <= 24 and
    K <= 200, the total is < 1e-300; the slack 1e-50 is added (upper) or subtracted (lower).
- (b) *Scaling and squaring, 24 < w <= 5000 (Opitz).*
  - Setup: e[s_i..s_l] = exp(Z)_il for the upper bidiagonal Z (diag s, superdiagonal 1), and exp(Z) =
    exp(Z/2^m)^(2^m) with 2^m >= w. exp(Z/2^m)_il = 2^(-m(l-i)) e[x_i..x_l], x = s/2^m in [0, 1].
  - Taylor step: the all-positive subinterval series with K = 18. The tail per entry is <= 1.1 / ((K+1)! (l-i)!)
    (h_k <= C(L+k, k) for x <= 1), added to the upper bound; the rounding chain is <= 6K + 3V + 6.
  - Normalisation: multiplying by the scalar e^-xm (xm = max x; outward exp) gives exp(Z/2^m - xm I). All its
    entries are <= 1, which is what removes the overflow limit; e^{2^m xm} = e^{s_max} is restored exactly in
    the log.
  - Squaring: m squarings of nonnegative matrices. Squaring k turns a chain of D_{k-1} roundings into
    D_k = 2 D_{k-1} + n + 1, so D = 2^m (n+1) + 2 in all.
  - Underflow: entries <= 1 give eps_{k+1} <= (2V+1) eps_k + V 2^-1074, so the slack
    n! (2V+2)^(m+1) (K+V) 2^-1074 is added to the upper bound.
  - Clamping: s_i < 2^-500 is raised to 2^-500 (upper; raising s raises F) or set to 0 (lower).
  - Check against 8000-bit arb divided differences: relative error <= 2e-9 up to w = 4900, n = 4 and 9; 0
    violations in 5297 random cases including clustered knots (`fns`).
- (c) *Cheap bounds.* Trivial: E <= e^-min t and E >= e^-max t. Convexity: E <= mean_i e^-t_i (e^-u <= sum
  lambda_i e^-t_i). Jensen: E >= e^-mean t.
- (d) *w > 5000.* Upper bound: lower the knots above min t + 5000 to that value (valid, E decreases in every knot;
  E depends on far knots only polynomially). Lower bound: Jensen.
- (e) log F is taken with the A1 margin, and LF <= w always (the trivial bound).

**E6 z-integral.** G(d, s, A, B) = int_A^B z^(d-1) e^(-z^2 s/2) dz = (2/s)^h / 2 [Gamma(h, xa) - Gamma(h, xb)], with
h = d/2 and x = A^2 s/2. G decreases in s, so the upper bound uses the exact float s = lower(s_c) and the lower
bound uses s = upper(s_c) + upper(R^2).
- Closed forms, with P(h, x) = e^x Gamma(h, x): for integer h, P = sum_j (h-1)!/j! x^j (exact integer
  coefficients). For h = m + 1/2, P = Gamma(h) erfcx(sqrt x) + sum_{k=1..m} Gamma(h)/Gamma(k+1/2) x^(k-1/2) (exact
  dyadic coefficients; Gamma(h) is an arb enclosure).
- Each term is bounded on the x interval by its monotonicity (erfcx decreasing, powers increasing).
- G = e^-xa C [P(xa) - e^-(xb-xa) P(xb)]: e^-xa goes into the exponent, the bracket is bounded outward
  (upper: P_up(xa) - e^-Delta_hi P_lo(xb); lower: max(0, P_lo(xa) - e^-Delta_lo P_up(xb))), and C = (2/s)^h / 2 is
  rounded outward.
- Last bin [A, inf): C P(xa), with s_min / s_max.

**E7 Assembly.** Each bin term is exp(sum of exponent bounds - T + log F) G_m. The exponent pieces are added with
directed rounding and exponentiated once (no intermediate overflow), then multiplied by G_m outward. The bin sums
are directed; c0 = (2 pi)^(-d/2) is an arb enclosure; the volume is outward.
- `mass_cell_fast` sums the leaves with math.fsum (correctly rounded), then one outward step.
- The numba port is checked against the numpy reference `_bounds_batch_np` (max relative difference of hi
  3.6e-15; the residual is numpy's SIMD exp against libm exp).
- The choice of which bins get the exact E bound (E8) and which variant is skipped are tightness decisions only.

## Status
Done (2026-10-19). The pre-registered line passes on its pooled per-simplex reading; the per-cell speed caveat is
under Results. Merge and confirmation are for T400 and the DRI.

## Results
Setup: `.venv` as in the project README, plus `pip install python-flint scipy numba`; `python harvest.py 1,8,2 200000`;
`data/bench_182_k4.npz` from branch claude/t373-event-cone (`m16/approaches/event_cone/nodes/`). Arb reference
runs: `python t412_record.py 2.25` and `python t412_record.py 2.5` (they reproduce the `results_a_m5t_*` ub exactly).

**Pass line (pre-registered): PASS on all three counts.**

| check | t0 = 2.25 (27 cells) | t0 = 2.5 (28 cells) |
|---|---|---|
| simplices compared (every simplex arb evaluates) | 107,113 | 111,101 |
| soundness: fast hi < arb lo / fast lo > arb hi | **0 / 0** | **0 / 0** |
| tightness: fast hi <= 1.01 x arb hi | **100%** | **100%** |
| max fast hi / arb hi | 1 + 5.3e-9 | 1 + 5.5e-9 |
| median fast hi / arb hi (fast tighter: arb's knot separation) | 1 - 8.6e-9 | 1 - 8.6e-9 |
| fast hi >= arb hi (1 - 1e-12) (reported, not in the line) | 2.6% | 3.4% |
| arb time per simplex | 8.74 ms | 8.54 ms |
| fast time per simplex (batched per cell, single process) | 77 us | 68 us |
| **speed-up, pooled per simplex** | **113x** | **125x** |
| speed-up per cell: median (min, max) | 116x (72x, ~155x) | 126x (100x, 178x) |
| end-to-end ub >= MC - 4 s.e. | **27/27** | **28/28** |
| end-to-end ub/MC median (arb M5t: 1.55 / 1.49) | 1.545 | 1.493 |
| end-to-end ub / arb ub: min, median, max | 0.990, 0.9996, 1.012 | 0.993, 0.9993, 1.006 |
| end-to-end time per cell, median (arb) | 0.66 s (23.2 s) | 0.57 s (22.2 s) |

Commands: `python t412_test.py simplex 2.25`, `... simplex 2.5`, `... cell 2.25`, `... cell 2.5`
(-> `results_t412_simplex_*.json`, `results_t412_cell_*.json`).
- *Timing.* Per-cell timing noise on this machine is about +-30%: cell 75 measured 72x in the test run and 102x
  on a rerun; the pooled numbers are stable. Arb was recorded with two arb processes running at once; run alone it
  is the same within 6% (`python t412_test.py calib`: 8.90 vs 9.47 ms and 8.60 vs 8.53 ms).
- *Why fast hi is mostly below arb hi.* The exceptions to fast hi >= arb hi (1 - 1e-12) come from arb's own
  looseness. Arb shifts clustered knots by 1e-9 relative in `_E_exp` (worth ~1e-8), its 53-bit `_G_arb`
  (gamma_upper) ball has relative radius up to ~2e-8, and it falls back to the trivial bound when its ball is not
  tight (fast hi / arb hi down to 1e-4). To check this, arb was re-run with these three loosenesses removed:
  knot separation 1e-30, 300 to 5000 bits until every ball is tight, `_G_arb` at 300 bits, 0 fallbacks.
  Compared with that, fast hi >= tight arb hi on **all** 950 sampled simplices; the excess is between 4e-12 and
  1e-8. Samples: 400 of cell 3 (t0 2.25), 400 of cell 71 (t0 2.5), 150 of region 3. The tight arb is up to
  2.7x below the 53-bit arb on the bench and up to 11x below it in 10-D. Commands: `python t412_test.py explain 2.25 3 400`,
  `... explain 2.5 71 400`, `... explainregion 3 1.5 150`.
- *Function assumptions (A1)* and E_exp against 8000-bit arb: `python t412_test.py fns` -> `results_t412_fns.json`.
  0 violations in 5297 random E_exp cases (n = 4, 9; spreads 1e-12 to 5000; clustered knots); worst upper excess
  1.9e-9.

**10-D region 3** (`data/regions_1-8-2.npz` index 3, event = True, t0 = 1.5, k = 10, d = 10, mlow = 0; 142 vertices,
10,301 root simplices; budget 4000 < 10,301, so no refinement in either method):

| | arb M5t | fast |
|---|---|---|
| ub, zbins 24 / ratio 23/20 (the reference setting) | 6.1886e-11 (reference: 6.19e-11) | **4.630e-11** |
| ub, zbins 48 / ratio 27/25 | 6.1885e-11 | 4.574e-11 |
| per-simplex bounds time, zbins 24 | 412 s (this run; reference 374 s) | **0.75 s warm (554x); 0.96 s incl. numba cache load** |
| per-simplex bounds time, zbins 48 | 474 s | 0.97 s (490x) |
| soundness (10,301 simplices, x 2 settings) | | 0 violations each way |
| exact geometry (enumeration, triangulation, fmpq volumes; same code) | ~40-55 s | 32.8 s |
| fast hi / arb hi per simplex: median (min) | | 0.75 (0.006) |
| budget 100,000 (fast only) | not feasible (~1 h) | ub **3.59e-11** in 22 s bounds + 29 s geometry |

Commands: `python t412_test.py regionarb 3 1.5 24`, `... regionarb 3 1.5 48`, `... region 3 1.5 4000`,
`... region 3 1.5 100000`. The 25% lower ub is not a looseness of fast. On the 150 region-3 simplices
checked, arb's 53-bit M5t is 1.21x to 11.2x (median 1.30x) above the tight arb value of the same formula, and fast
is within 1.7e-9 above it. The region MC (T400 README) is 9.1e-12, so ub/MC is 5.1 (budget 4,000) and 3.9
(budget 100,000), against 6.8 for arb.

## Compute accounting
Deductive method: no random model inputs. MC numbers are T400's (`results_a_m5t_*.json`, README).
CPU (one container, at most 2 heavy processes):
- arb reference recordings: 2 x 11 min;
- arb on region 3: 2 x 8 min;
- explain checks: ~25 min in total;
- fast tests: < 10 min in total.
Per simplex: arb 8.6 ms (d = 5) and ~40 ms (d = 10); fast 70 us (d = 5) and 73 us (d = 10).

## Claims
- [measured] 0 soundness violations of fast against arb: 218,214 bench simplices (fast hi >= arb lo, fast lo <=
  arb hi) and 2 x 10,301 region-3 simplices. End-to-end ub >= MC - 4 s.e. on all 55 bench cells.
- [measured] fast hi <= 1.01 x arb hi on 100% of the bench simplices; in 10-D fast hi is below arb hi (median 0.75).
- [measured] Pooled speed-up per simplex: 113x (t0 2.25) and 125x (t0 2.5) on d = 5 cells; ~500x on the d = 10
  region (arb is slower per simplex in 10-D, fast is not).
- [measured] Against arb with its looseness removed (tight E and G, 0 fallbacks), fast hi >= that value on 950 of 950
  sampled simplices, at most 1e-8 above it.
- [inferred] fast [lo, hi] is rigorous for every input, by E0-E7, given assumption A1 (libm / scipy accuracy, tested
  with >= 120x margin, not proved).
- [measured] Arb's default 53-bit M5t is looser than its own formula: up to 11x per simplex in 10-D, and region-3 ub
  6.19e-11 against 4.63e-11 for the same formula evaluated tightly.
- [speculative] The exact geometry (unchanged code) is now the bottleneck: ~3 ms per root simplex in region 3, so
  ~1.7 h per 2e6-simplex region if it scales linearly.

## Alternatives not ruled out
- A1 is an assumption about this platform's libm, numpy SIMD and scipy (cephes, Faddeeva). A different build could
  be less accurate; the margins are >= 120x and `python t412_test.py fns` re-checks them.
- Compiler: numba runs without fastmath. If LLVM fused a*b + c into an FMA, the per-operation error bounds would
  still hold (one rounding instead of two), but the outward step x + |x| 2^-52 relies on not being rewritten.
  This was checked only empirically (the port agrees with numpy to 3.6e-15, and fast is never below the tight arb).
- Timing is noisy (+-30% per cell). One reading of the pre-registered line, "per simplex" as a per-cell minimum,
  would not pass at t0 2.25 (72x). The pooled reading passes at both t0.
- Refinement in rounds (top max(8, n/16) at once) changes the order relative to arb (one at a time): ub differs
  from arb's by -1.0% to +1.2% at the same budget.

## Failure modes
- Knot spread w > 5000 in a bin that matters: the upper bound caps the knots (loose only polynomially) and the
  lower bound uses Jensen. Not seen in a bin that matters.
- Results below ~1e-290: the absolute underflow slack dominates (still valid, but loose).
- A poor tangent centre (the Wolfe iteration cap is hit) makes T0 << 0, and the upper terms grow like
  exp(lambda |T0|): valid but loose, and inf in the worst case. FISTA showed this; Wolfe has not.
- Large regions: the exact geometry (`cell_geometry`, a copy of rigour's) dominates the run time.

## Proposed follow-ups (for the DRI's backlog; not added here, as T412's instructions limit the board edit to its row)
1. Fast rigorous geometry: float vertex enumeration and triangulation with exact a-posteriori checks, and float
   simplex volumes with a determinant error bound. This is now >= 97% of the region time.
2. Re-run T400's tiered charging and k = 10 regions with `mass_cell_fast` at larger budgets (region 3: ub 6.19e-11
   -> 3.59e-11 at budget 1e5 in under a minute).


## T417 audit gaps and fixes (claude/t412-audit-fixes)
T417's independent audit (claude/t417-m5t-audit @ 456eefe5) found no soundness counterexample: 1e6 adversarial
simplices against an independent 256-bit arb evaluation of the formula, and 1e6 _e1 knot vectors against
arb_mat.exp. Its 12 gaps, by number:

| # | gap | fix |
|---|---|---|
| 1 | s_min floor 1/2 assumes m_p2 = 1 + m_p3, unchecked | `Params(jp2=...)`: the floor is used only when jp2 is given (mass_cell_fast passes it), after a check that every vertex enclosure is consistent with m_p2 = 1 + m_p3 |
| 2 | up/dn and TINY need gradual underflow | import-time check that numpy and numba do not flush subnormals |
| 3 | s_c lower end <= 0: ZeroDivisionError / NaN | `_Cn` returns inf (upper) or 0 (lower) for s <= 0 |
| 4 | P(x) overflow for huge edges gives NaN | final guard: hi NaN -> inf; lo non-finite -> 0 |
| 5 | NaN inputs: infinite series loop, NaN outputs | non-finite inputs rejected in bounds_batch; _e1 returns the trivial bound for non-finite knots; series loop capped (k <= 10000) |
| 6 | _emat lower bound lacked the underflow slack | slack now subtracted |
| 7 | convexity branch omitted TINY | added |
| 8 | idot/sumsq ignore product underflow | covered by SAFE = 2 (no change); noted here |
| 9 | vol = 0 gives lo = -2^-1074 | lo clamped at 0 |
| 10 | A1 margin overstated (erfcx 120x, not 130x); docstring cited a nonexistent test file | corrected |
| 11 | E5(a) wording | corrected |
| 12 | edges ** 2 relies on exact squaring | written as e * e (one correctly rounded product) |

Regression after the fixes: `python t412_test.py simplex 2.25` and `... simplex 2.5`, and `python t415_test.py case
region 3` (results below).

Regression (after the fixes):
- simplex t0 2.25: 0 / 0 violations, 0 non-finite, 100% within 1.01, hi quantiles bit-identical to before.
  Speed-up 101x (113x before; the extra finiteness checks plus +-30% timing noise).
- simplex t0 2.5: 0 / 0 violations, 100% within 1.01, 108x (125x before).
- geom_fast region 3: identical geometry and mass.
- Guards checked directly:
  - a NaN or inf knot gives the trivial bound (inf / 0);
  - `_Cn(s <= 0)` gives inf / 0;
  - a non-finite vertex raises;
  - vol = 0 gives lo = 0;
  - m_c = 0 gives hi = inf (no exception).
