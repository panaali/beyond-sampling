# T291: shared-innovation polynomial coefficients with explicit remainders (Astra r2, C3)

Owner: Claude child session_018Np7T8Xt9g9bHPmXWfyU1i, branch `claude/t291-poly-coeff`. Pointers:
[review/codex_astro_review_r2](../../../review/codex_astro_review_r2/README.md) (C3), INDEX H-S1, T324, T361.

## Question

H-S1 (dead, verified, scoped) says the exact-walk family cannot reach the strict bar (F <= 1) at C <= ~6k FPE; a new
representation is needed. C3 proposes one: represent each hidden-state function h_t^i(x) by its low-degree Gaussian
(Hermite) coefficients, propagate them jointly from the weights (the common input x_t enters every neuron at once),
replace ReLU by analytic polynomial approximants on a declared domain, integrate endpoint correctness inside the exact
rank cones, and carry every truncation remainder in a ledger. **Does this representation, at a 4k-FPE all-in screen
with degree and rank fixed in advance, give a sound interval narrow enough for the strict bar on four DEV2 models?**

## PRE-REGISTRATION (written and pushed before any scored run; nothing below this section changes it)

Written 2026-10-18 before computing anything with the four screen models. Read so far: model-free cost counts of the
Hermite algebra (`herm.py`, table below) and the panel-wide mean p(1-p) = .0725 over all 80 DEV2 models (used only
to state the allowance). No screen-model truth, estimate or diagnostic has been computed.

### Screen models and harness

- DEV2 (open panel, T284; copied with hashes into `panel/`: weights sha256 f96e0d99..., truths 582c21cf...), the
  four models C3/C1 name: initialization IDs 0 and 1 from both kernels = **fused_s0, fused_s1, torch_s0, torch_s1**
  (indices 0, 1, 40, 41). No HP panel is touched (HP5 = seeds 400-499 is never read).
- Estimator interface `est(model, budget, rng) -> (estimate, evals)` of [m16/arc_eval.py](../../arc_eval.py);
  the harness's own `forward` is used for every diagnostic Monte Carlo. (`arc_eval` imports `m16/common.py`, which
  needs torch + alg_zoo for the ORIGINAL weights; they are not installed in this container, so `run.py` injects a stub
  `common` module before the import. The stub only supplies placeholder arrays used by `noisy_models`, which is not
  called. No shared file is edited.)
- Class: **deductive** (no random model inputs). Every Monte Carlo below is a labelled diagnostic, never part of the
  estimate.

### Representation and the fixed configuration

- Basis: probabilists' multivariate Hermite polynomials He_alpha(x), x in R^10, total degree |alpha| <= d. Full
  coefficient tensor (no compression), so the **"rank" is the full count M(d) = C(10+d, d)** (66 / 286 / 1001 at
  d = 2 / 3 / 4). Shared innovation: at step t only indices on x_1..x_t are active, and the truncated product only
  touches those.
- Step t: u_t = Wh p_{t-1} + Wi x_t (exact). Neuron i: mean mu and sd s of u_t^i from its coefficients (exact). If
  [mu - 4s, mu + 4s] lies in u >= 0, p_t^i = u_t^i; if it lies in u <= 0, p_t^i = 0; otherwise
  p_t^i = q_k(u_t^i), where q_k is the degree-k Chebyshev interpolant of ReLU on [mu - 4s, mu + 4s], evaluated by
  Horner in the **truncated algebra** (each product projected back to degree <= d).
- Endpoint: logits l = Wo p_10; margins m_jk = l_j - l_k; correctness = sum_j P(x_j is 2nd largest and m_jk > 0 for
  all k != j), the rank cone handled exactly.
- **Fixed configurations (chosen from the model-free cost table only):**
  - **S (the screen): d = 3, k = 8**, full tensor (M = 286 per neuron).
  - S2 (cheap rung): d = 2, k = 8.
  - S4 (over the cap, diagnostic only): d = 4, k = 4 (propagation alone ~20-30k FPE, see cost table).
  No other degree, k, domain width (4 s) or basis is tried on the screen models. A failure of S is a failure of S.

Model-free cost table (`python herm.py`): structure constants of the truncated product, summed over the active
variables at t = 1..10 (3 FLOPs each): d = 2: 4.0k, d = 3: 59.9k, d = 4: 822k. Projected propagation cost of S
(all 16 neurons straddling at every step, 7 Horner products each): 16 x 7 x 3 x 59.9k ~ 2.0e7 FLOPs ~ 3.4k FPE,
plus linear mixing ~ .25k FPE. That leaves ~ .35k FPE for the whole endpoint step. 1 FPE = 5948 FLOPs (T281
convention). C is counted as executed (FLOPs of every operation that depends on the model); model-free tables
(product structure constants, Chebyshev nodes, cone-moment tables) are built once and not charged, as in earlier
rows.

### Stages and pre-registered stops (cheapest first)

Allowance. With C_eff = max(C, 1000), the strict bar is mean_m MSE_m <= mean_m p_m(1-p_m) / C_eff. For a sound
interval [l_m, u_m] with half-width h_m, the screen **passes** only if
`mean_m h_m^2 x C_eff / mean_m min(l_m(1-l_m), u_m(1-u_m)) <= 0.8` (C1's headroom form) and C_m <= 4000; the interval
is **too wide** (STOP) if `mean_m h_m^2 x C_eff / mean_m p_m(1-p_m) > 1` (p_m the published truth). At C = 4000 and
p(1-p) ~ .0725 the "too wide" line is RMS h > .43 pp, the pass line RMS h <= .38 pp.

- **G1, cost / rank cap (no truths).** Execute S's propagation on the four models and count FLOPs. STOP ("rank cap")
  if the propagation alone exceeds 4000 FPE on any model, or if no endpoint step with a non-vacuous bound fits in
  4000 - C_prop (the endpoint cost is itemised: the cheapest sound endpoint I can write down with polynomial
  indicator bounds needs cone moments of degree k_s x d; its FLOP count is reported).
- **G2, necessary width (uses the truths of the four models).** Let s_m be the exact accuracy of the propagated
  surrogate (its argmax rule integrated against the exact input law). Any sound remainder ledger built around the
  surrogate must contain both s_m and p_m, so h_m >= |s_m - p_m| / 2. s_m is measured by a **diagnostic** Monte Carlo
  of the surrogate (2e6 inputs per model, SE ~ .02 pp; a stand-in for exact integration, i.e. generous to the method).
  STOP ("interval too wide") if `mean_m (|s_m - p_m|/2)^2 x 4000 / mean_m p_m(1-p_m) > 1`.
- **G3, the ledger itself** (only if G1 and G2 pass): build the remainder ledger (L2 propagation of every truncation
  and Chebyshev remainder, Gaussian tail outside the domain, union/indicator bounds at the endpoint) and apply the pass
  rule above. If the ledger cannot be made sound within 4000 FPE, that is a STOP, not a fallback: no Gaussian
  closure, fitted closure or independence assumption replaces a failed bound.

Verdict: PASS only if G1-G3 all pass. Otherwise a scoped negative with the exhaustion checklist (AGENTS rule 5).

### Diagnostics (all Monte Carlo, never part of an estimate; reported whatever the verdict)

- D1 oracle representation floor: the L2-optimal degree-d Hermite projection of the TRUE logit functions (coefficients
  by MC averages, 4e6 inputs per model at d <= 4; d = 5 at 4e6 if memory allows) and the accuracy of its argmax rule,
  d = 1..4(5). This bounds what ANY degree-d coefficient propagation could reach, independent of the propagation rule.
- D2 per-step relative L2 error of the propagated h_t versus the true h_t.
- D3 s_m for S2 and S4 as well as S.

### Alternatives written before the run, and the cheapest separating test

- A1. The class itself fails: low-degree polynomials of the 10 inputs cannot place the argmax walls (piecewise linear
  logits with ~10 kink layers; the decision lives in thin tie bands). Then D1 is already above the allowance.
- A2. The class is adequate but the within-budget propagation (truncated algebra, Chebyshev ReLU) loses it. Then D1
  is under the allowance but s_m - p_m is not.
- A3. Both are fine and only the ledger is loose (C3's own warning). Then G2 passes and G3 fails.
- Cheapest separating test: D1 versus G2 at the same d on the same models (one MC pass each). If G2 fails, A3 is moot.

## Method

(Filled in after the run.)

## Status

Pre-registered; not run.

## Results / Compute accounting / Claims / Alternatives not ruled out / Failure modes

(After the run.)
