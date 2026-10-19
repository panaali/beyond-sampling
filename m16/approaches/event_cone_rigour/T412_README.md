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
| erfcx (scipy) | 1e-13 rel | 7.9e-16 | ~130x |
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
  and S_exact lies in S_fl (1 -+ gamma_D).
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
