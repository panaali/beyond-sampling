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
Files (all in this folder; T412's code is imported read-only from `../event_cone_rigour/`):
- `ref.py`: the independent 256-bit arb reference (see Pre-registration).
- `gen.py`: the 13 adversarial families; `geom.py` caches their seeds (28 bench k = 4 cells and 10-D regions 3, 401, 424
  of `harvest.py 1,8,2 200000`; 54,716 seed simplices; other sampled regions exceeded a 2-min geometry limit).
- `audit.py`: E2E driver (fast `bounds_batch` with an explicit `mc`, then `ref.ref_bounds` on the exact vertices).
- `unit_e.py`, `qcheck.py`, `a1.py`: UNIT-E, Q-level and A1 (see Pre-registration).
- Line-by-line reading of T412_README.md sections E0-E7 against `m5t_fast.py` (production path: `_bounds_batch` ->
  `_prep1`/`_prep2`/`_bins`/`_bin_terms`/`_e1`/`_emat`; the numpy `_bounds_batch_np`/`G_fac` are reference-only).

## Status
Done (2026-10-19). No soundness counterexample. The pre-registered line fails on its robustness clause (NaN lo /
exceptions on extreme inputs, all with hi = inf or an exception, never a wrong finite bound); see Results.

## Results
**Verdict: no counterexample to soundness; 0 violations and 0 shortfalls in 1,000,000 E2E cases and 1,000,000 E_exp
cases. Pre-registered PASS line: not met, because of 240 non-finite / exception outcomes on inputs far outside the
production settings (details below), which the pre-registration counted as violations.**

| check | cases | result |
|---|---|---|
| E2E hard violations (fast hi < ref LO, fast lo > ref HI) | 1,000,000 | **0** |
| E2E shortfalls (fast hi < HI (1 - 1e-10), fast lo > LO (1 + 1e-10)) | 1,000,000 | **0** |
| min fast hi / exact HI; max fast lo / exact LO (all cases with finite values) | | 1 + 2.4e-13; 1 - 2.2e-13 |
| median fast hi / exact HI (real / child / encl families) | | 1 + 8e-11 / 1 + 6e-11 / 1 + 1e-6 |
| NaN output | 1,000,000 | **229** (all `zbins`, ratio 20, 48 bins: lo = NaN, hi = inf) |
| exceptions | 1,000,000 | **11** (all `advmc`: ZeroDivisionError, tangent centre with s_c rounding to 0) |
| hi = inf (valid, useless) | | 50,835 (48,732 `advmc`, 2,103 `encl` with relative width ~1) |
| hi = -inf or negative (separate fast-only re-run, advmc / encl / zbins) | 60,000 | 0 |
| UNIT-E `_e1` violations / non-finite | 1,000,000 | **0 / 0**; max upper excess (log) 0.029, lower gap 1.95e-9 |
| Q-level (MC, non-rigorous): hi flags / lo flags | 10,000 | **0 / 16**; the 16 are all `advmc` with rse 0.4-1.0 (Q = 1e-20 .. 1e-178, integrand peaked over many orders): MC failures, see below |

Commands: `python audit.py mix 500000 1`, `python audit.py mix 500000 2` (2 x 4,800 s) -> `results/audit_mix_{1,2}.json`;
`python unit_e.py 1000000 1` (340 s) -> `results/unit_e_1.json`; `python qcheck.py 5000 2`, `... 5000 3` ->
`results/qcheck_{2,3}.json`; `python a1.py 20000 1` -> `results/a1_1.json`. Seeds: `python geom.py` after
`python ../event_cone_rigour/harvest.py 1,8,2 200000` and `data/bench_182_k4.npz` from claude/t373-event-cone.
(A first `qcheck.py 10000 1` finished but crashed writing its JSON, a numpy int64; its numbers are lost, re-run as above.)

*Q-level flags.* The 16 lo flags have plain-MC relative s.e. 0.4-1.0, so the estimate is not usable (a heavy-tailed
sample mean is usually far below its expectation). Stratified MC (1024 equal-volume pieces x 64 points, `python qstrat.py
flagged 10 64` -> `results/qstrat_flagged_10_64.json`) moves 15 of 16 below 1 (lo/Q 2.6e-7 .. 0.59), one at 4.8 with rse 0.65.
Q does not depend on the tangent centre, so the decisive check is consistency: fast lo with the adversarial centre
<= sum of fast hi over 2^0, 2^6, 2^10 bisection children with the production (Wolfe) centre. It holds in 16 of 16 cases,
by factors 1.1 to 1e8 (`python consistency.py` -> `results/consistency.json`; children volumes vol/2^K with 1e-12 slack, so this is a
consistency check, not a certificate). So the pre-registered "0 Q-level flags" is not met literally: 16 flags, all
attributable to the MC estimator. No hi flag in 10,000 cases (min hi / Q_mc 0.016, at rse > 1/6).

**A1** (`results/a1_1.json`; this container: numpy 2.5.4, scipy 1.18.1, numba 0.68.0; not T412's builds). Inputs
recorded while running 20,000 E2E-style cases: 857,168 distinct log_ndtr arguments in [-4.6e62, 1.1e63], 1,211,516
erfcx arguments in [4.9e-160, 4.0e63]; 40,000 of each checked (both extremes of the sorted inputs included).

| function | assumed | measured worst (mpmath 60 digits) | margin |
|---|---|---|---|
| log_ndtr (scipy), abs / (1 + x^2) | 1e-13 | 4.3e-16 at x = -0.82 | 232x |
| erfcx (scipy), rel | 1e-13 | 8.3e-16 at x = 1.3e-3 | **120x** (T412 claims >= 130x) |
| exp (numba / libm), rel beyond 4·2^-1074 | 1e-13 | 1.1e-16 | 908x |
| exp (numpy), same | 1e-13 | 1.2e-16 | 804x |
| log (numba / numpy), rel | 1e-13 | 1.1e-16 / 1.2e-16 | 916x / 842x |

Environment: subnormals are not flushed (numba and numpy); the compiled numba functions contain no FMA, `contract`
or `fast` flags and no SVML calls (LLVM IR and assembly, fresh cache; the IR counts inside `a1.py` itself are invalid
because numba disables inspection of cached code).

### Gaps in the derivations (sent to T412's author)
Classes: **(S)** can produce a wrong bound, **(B)** benign (covered elsewhere), **(D)** robustness / documentation.
1. **(S, precondition)** `simplex_consts` floors s_min at 1/2 using m_p2 = 1 + m_p3. `bounds_batch` cannot check it (it
   is not given p2). Valid for `cell_geometry` and its bisections; any other caller with simplices off the slice gets
   an invalid last-bin bound. Document it in the interface or pass jp2 and check it.
2. **(S, environment)** `up`/`dn`/`_up`/`_dn` and the absolute TINY terms rely on gradual underflow (2^-1074 > 0). If
   FTZ/DAZ were set in the process (a library linked with -ffast-math), up(0) = 0 and subnormal results flush. Not
   checked at run time; holds here. Suggest an import-time assert in numpy and numba.
3. **(D)** `_Cn` divides by s with numba's Python error model: s_c lower bound rounding to exactly 0 (tangent centre 0 or
   tiny) raises ZeroDivisionError (11 E2E cases); a slightly negative lower bound (dn of 0) is not rejected either.
   It gives hi = inf only because NaN then drops out of a later `min`. Not reachable with the Wolfe centre on
   the slice (s_c >= ~1/2). Suggest clamping s to >= tiny positive or returning inf explicitly.
4. **(D)** P(x) overflow for huge z edges (A^2 s/2 > ~1e61-1e77 for d = 10-12): bracket inf - 0·inf = NaN, so lo = NaN with
   hi = inf (229 E2E cases, edges up to 7e60). Not reachable with zbins 24 / ratio 23/20. `mass_cell_fast` then gives
   lb = max(0, NaN) = 0: safe, but silently. Suggest assert finite edges^2·s range or map NaN lo to 0.
5. **(D)** No NaN guard anywhere: a NaN knot makes `_e1`'s series loop run forever (reproduced: hang); a NaN vertex
   coordinate gives lo = NaN, hi = inf; `mass_cell_fast` sums with fsum and would return ub = NaN for a NaN leaf
   (the refinement test `th > tol*tl` is then False). Suggest asserting finite inputs and treating non-finite hi as inf.
6. **(B)** `_emat` lower branch subtracts no underflow slack (README E5(b) states it for the upper bound only).
   Round-to-nearest in the subnormal range can round up; the effect is <= ~1e-296 absolute against M[0,n] >= ~1e-40,
   i.e. inside the unused (1 - 2g) vs (1 - g) slack by > 250 orders. One-line fix: subtract the same slack.
7. **(B)** `_e1` convexity branch uses exp without the absolute TINY term; covered because acc >= 1 and `_up` leaves a
   spare half ulp (~1e-16) >> V·TINY.
8. **(B)** `idot` / `sumsq_*` use relative gamma bounds that ignore underflow of individual products; covered by the
   SAFE = 2 factor because the sums are O(1) on the slice.
9. **(B)** vol = 0 gives lo = dn(0) = -2^-1074 < 0. Harmless (`mass_cell_fast` clamps), but a negative lower bound.
10. **(D)** A1 margins: T412_README says ">= 130x" while its own table gives erfcx 1e-13/7.9e-16 = 127x; measured here
    120x. The module docstring cites other numbers (6.7e-16, ">= 150x") and a file `test_m5t_fast.py` that does not
    exist (it is `t412_test.py`). A1 itself holds with >= 120x on every recorded input.
11. **(D)** E5(a) wording: "S_exact lies in S_fl(1 -+ gamma_D)" should be S_fl in S_exact (1 +- gamma_D); the code's
    factor (1 +- 2 gamma_D) covers the correct statement and the final roundings.
12. **(D)** `llo`/`lhi` in `_bounds_batch` use numpy `edges ** 2`: correct because numpy squares exactly-rounded (x*x)
    and dn/up add a full ulp; worth a comment.
Checked and found correct: the gamma_n chains of E5(a) (delta(k,j) <= 4k + j + 1, D = 6K + n + 2) and E5(b) (Taylor
chain <= 4K + 2V + 2 <= DT; squaring D_k = 2D_{k-1} + n + 1), the series tail and its directed propagation, the Opitz
normalisation, the clamping at 2^-500, the cap and Jensen branches, E1-E4 (every interval end, including the y sign
cases for kappa, the z y relaxations staying inside [xl, xh], the chord's slope substitution with eb, the tangent's
|f' - r~| W_x term), E6 (half-integer and integer P, erfcx arguments rounded the right way, the e^-xa factor and the
bracket), E7, midpoint enclosures and child volumes in `mass_cell_fast`.

## Compute accounting
Deductive check (no model inputs except the seed geometry: `harvest.py` 2e5 line-walk model runs, 14 s).
CPU, one container, <= 2 heavy processes: E2E 2 x 80 min (~15 ms / case: fast ~0.1 ms, reference ~15 ms); UNIT-E
6 min; A1 ~2 h (mpmath); Q-level 2 h lost + re-run 2 x 20 min + follow-up ~20 min; seed geometry ~25 min.

## Claims
- [measured] 0 violations and 0 shortfalls of `bounds_batch` against an independent 256-bit arb evaluation of the same
  M5t formula on 1,000,000 adversarial simplices (13 families, d = 2-12, mlow 0-8, t0 1e-3-100).
- [measured] fast hi >= exact formula hi (1 + 2.4e-13) and fast lo <= exact formula lo (1 - 2.2e-13) in every case:
  fast is never tighter than exact arithmetic would be, which is what its error analysis claims.
- [measured] `_e1` (E_exp) never crosses the arb_mat.exp reference on 1,000,000 adversarial knot vectors.
- [measured] A1 holds on this platform with >= 120x margin on the arguments actually used (T412's ">= 130x" is not met
  for erfcx: 120x here, 127x in T412's own table).
- [measured] Non-finite outcomes exist (NaN lo, ZeroDivisionError, a hang on NaN knots) but only on extreme or invalid inputs,
  and never as a finite wrong bound.
- [inferred] With the production settings (zbins 24, ratio 23/20, Wolfe centre, slice geometry) `bounds_batch` is sound,
  given A1 and gap 2 (no FTZ). Gaps 1-5 are interface preconditions to document or enforce.

## Alternatives not ruled out
- The reference shares the formula with fast (by design: it checks the evaluation, not T400's derivation). The Q-level
  Monte Carlo check covers the formula itself, but it is statistical and float-based.
- Both reference and fast use the tangent centre chosen by the driver; with `mc=None`, fast uses Wolfe, and 50% of the
  cases used exactly that centre (`gen.pick_mc("wolfe")` replicates `bounds_batch`).
- A1 was measured on this container's builds; another numpy / scipy / libm could differ (T412's numbers are similar).
- Finite sampling: 1e6 cases cannot exclude a rare input region not covered by the 13 families.
- `mass_cell_fast` end to end (refinement, fsum) was checked by reading only, not by an independent run.

## Failure modes
- Extreme inputs: z edges >~ 1e30 (NaN lo), tangent centre with |m_c|^2 rounding to 0 (exception), NaN inputs (hang).
- The reference's arb_mat.exp is slow for d >= 11 (~0.4 ms per E); 1e6 cases need ~2 container-hours.
