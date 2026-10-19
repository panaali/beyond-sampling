# T417: independent soundness audit of T412's fast rigorous M5t (`m5t_fast.py`)

↑ [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · audited code: [event_cone_rigour/T412_README.md](../event_cone_rigour/T412_README.md)

Owner: Claude child session_01V9adSLcKt8pZdpCQgkW4du. Branch `claude/t417-m5t-audit`, off `claude/t412-m5t-fast` @ f61b1f22.
Only this folder is written; T412's files are not edited (gaps are reported to T412's author, not fixed here).

## Question
Is `m5t_fast.bounds_batch` (T412) a sound enclosure of the per-simplex M5t quantity for *every* input, not only the
bench simplices it was tested on? Concretely: (1) do the error derivations in T412_README.md match the code, with
no unbounded operation, wrong gamma_n count, missing outward rounding or unstated assumption; (2) can adversarial
inputs make fast hi fall below (or fast lo rise above) a high-precision reference; (3) does assumption A1
(libm / numpy / scipy accuracy) hold with the claimed margin on the inputs actually used?

## Pre-registration (pushed before the large runs)

### Reference (independent code, `ref.py`)
Written from the formula, not from `m5t_fast.py`. python-flint arb at 256 bits on the exact rational vertices.
For each simplex it evaluates *exactly* the M5t upper formula HI and lower formula LO of `rigour._bounds_tangent`
with the same tangent centre m_c and the same float z-bin edges that fast uses:
- E_unif exp(-u) by Hermite-Genocchi with flint's rigorous `arb_mat.exp` of the bidiagonal divided-difference
  matrix (handles coincident knots; a different algorithm from T412's series / Opitz float code);
- the z-integral by arb `gamma_upper`; log Phi by arb `erfc`; tangent slope phi/Phi at the same float x0;
- s_c, tau, T0 = min tau, R^2, s_max and s_min (the same separating-hyperplane bound, floored at 1/2) exactly.
- Bins whose trivial bound is < 1e-30 of the running total enter as the interval [0, trivial] (rigorous).
By T400's derivation LO <= Q <= HI, with Q the true per-simplex integral.

### Test sets
**E2E (end-to-end, `audit.py`):** >= 1,000,000 cases, 13 generator families (`gen.py`), each case a simplex on the slice
m_p2 = 1 + m_p3 (checked exactly), seeds drawn from the real simplices of the 27 + 28 bench k = 4 cells and of
harvested 10-D event regions (`harvest.py 1,8,2 200000`, region 3 plus random others):
`real` (as is), `child` (0-14 bisections), `shrink` (scaled by 1e-1 .. 1e-15.5: clustered knots), `coinc`
(shared coordinates and duplicated vertices: tied knots, zero volume), `flat` (vertex 1e-1 .. 1e-15 from a facet),
`needle`, `huge` (x 2 .. 1000), `far` (translated by up to ~300: large s), `zbins` (t0 1e-3 .. 40, ratio 1 + 2^-20 .. 20,
2 .. 48 bins), `under` (far + t0 up to 100: results far below 1e-308), `encl` (vertex enclosures of relative width
1e-16 .. 1 around the exact vertex), `synth` (random d = 2 .. 12, mlow in {0,1,2,5,8}, all y_p3 sign patterns),
`advmc` (tangent centre 0, a vertex, far away, tiny, noisy, or fast's own Wolfe). Tangent centre otherwise:
fast's Wolfe (50%) or scipy NNLS (50%).

**UNIT-E (`unit_e.py`):** >= 1,000,000 knot vectors for `_e1` (n = 1 .. 10), spreads 1e-320 .. 3e5 with extra mass at
the switches w = 24, 5000 and 10000, patterns: random, clustered with gaps 1e-17 .. 1 or exactly 0, one outlier,
all-equal-but-one, grids, geometric; offsets up to 1e9. All four modes: upper with cap, lower, upper without cap,
trivial. Reference: `ref.logE` at 256 bits.

**Q-level (`qcheck.py`, secondary, not rigorous):** >= 10,000 cases (all families but `under`): Monte Carlo over the
simplex (2048 points) of the exact z-integral (composite Gauss-Legendre in log space). Checks the formula itself.

**A1 (`a1.py`):** every argument passed to scipy `log_ndtr` and `erfcx` while running ~20,000 E2E-style cases is
recorded and checked against mpmath (60 digits); numba (libm) and numpy exp / log on sweeps of their ranges;
environment: subnormals not flushed, no FMA / fast-math / SVML in the compiled functions.

### Definitions and pass line
- **Violation (hard):** fast hi < LO (lower end of the reference ball) or fast lo > HI (upper end), i.e. fast
  contradicts a rigorous enclosure of Q. Also a violation: NaN output, or an exception on a valid input.
- **Shortfall (rounding-analysis check):** fast hi < HI (1 - 1e-10) or fast lo > LO (1 + 1e-10). Fast evaluates the
  same formula with every step rounded outward, so its hi should never be below the exact formula value beyond
  ~1e-12 (float slope substitutions); a shortfall points at a gap in the error analysis.
- **UNIT-E violation:** exponent -T + LF below log E (upper modes) or above it (lower mode); non-finite output.
- **Q-level flag:** fast hi < Q_mc (1 - 6 rse) or fast lo > Q_mc (1 + 6 rse).

**PASS (no counterexample) requires all of:** E2E >= 1e6 cases with 0 violations and 0 shortfalls; UNIT-E >= 1e6 with
0 violations; Q-level >= 1e4 with 0 flags; A1: every measured worst error >= 10x inside the assumed bound on the
recorded inputs. Independently of PASS, every gap found in the derivations is listed and classified as
**(S)** can produce a wrong bound, **(B)** benign (covered by slack elsewhere; margin stated), or **(D)** documentation.
Exceptions on inputs outside the documented domain (e.g. overflowing z edges) are listed, not counted.

## Method
(to be completed with results)

## Status
Running.
