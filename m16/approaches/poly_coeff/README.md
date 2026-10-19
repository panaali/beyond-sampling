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

Everything above the pre-registration line was run as written; no configuration was changed after the pre-registration
commit (67e44f70).

- `herm.py`: the truncated multivariate Hermite algebra (model-free structure constants; shared-innovation restriction
  of each product to the active inputs x_1..x_t).
- `run.py`: `propagate` (deductive, FLOPs counted as executed), the endpoint cost itemisation, the diagnostic MC
  (`diag`: surrogate accuracy s_m, per-step relative L2 error, Cantelli width) and D1 (`oracle_floor`). It uses
  `arc_eval.forward` for every true-model evaluation, through a stub `common` module (see the pre-registration).
- `floor.py`: D1 driver; `summarize.py`: the gate statistics and tables below from the saved JSON.

## Status

**Done. Verdict: STOP (screen fails) at G1 and at G2, by wide margins; G3 not reached.** Negative (unverified),
scoped below.

## Results

Four DEV2 models (fused_s0, fused_s1, torch_s0, torch_s1; truths .8720 / .9515 / .9591 / .9438). Reproduce:
`python run.py 3 8 2 8; python run.py 4 4; for m in fused_s0 fused_s1 torch_s0 torch_s1; do python floor.py $m 1 2 3 4 5; done;
python summarize.py` (about 25 CPU-min in all; each process under 2 GB).

**G1, cost / rank cap: STOP** [measured]. Config S's propagation executes at **C_prop = 3,496 FPE** (all 160
neuron-steps straddle zero, so every one needs the 7 truncated products). The cheapest sound endpoint I could write
(conditional first and second moments of all 90 margins on the 10 regions {x_j is 2nd largest}, via model-free Gram
tables) costs another **2,856 FPE**, so C >= **6,352 FPE > 4,000**. And that endpoint is vacuous anyway: the Cantelli
union bound from the surrogate's own conditional moments, ignoring every propagation remainder, gives a lower bound
on accuracy of **0.0** on all four models (the interval is [0, 1]).

**G2, necessary width: STOP** [measured, diagnostic MC 2e6 inputs per model, SE ~.03 pp]. The surrogate's exact
accuracy s_m (its argmax rule against the true rank, integrated by MC as a stand-in for exact integration):

| config | C_prop / C_end FPE | s_m (fused_s0, fused_s1, torch_s0, torch_s1) | RMS(s - p) | G2 statistic (STOP if > 1) | point F if s were the estimate (C_prop only) |
|---|---|---|---|---|---|
| S2: d=2, k=8 | 321 / 171 | .170 .184 .144 .179 | 76.3 pp | 9,321 | 9,321 (C_eff 1k) |
| **S: d=3, k=8** | **3,496 / 2,856** | **.274 .256 .251 .269** | **67.1 pp** | **7,193** | **25,149** |
| S4: d=4, k=4 (over cap) | 20,216 / 34,062 | .245 .100 .228 .220 | 73.8 pp | 8,700 | 175,883 |

Per-step relative L2 error of the propagated hidden state (D2, S): .17-.18 at step 1 (the univariate floor for
ReLU(w x_1) at degree 3 is .15), then .16-.31 to step 10. S4 diverges on fused_s1 (relative L2 1e1 at step 5, 7e10 at
step 10: the k = 4 interpolant composed in the truncated algebra is unstable there).

**D1, the oracle floor of the representation class** [measured, diagnostic; Hermite coefficients by MC averages over
4e6 inputs, accuracy on 1e6 fresh inputs; cross-checked on torch_s1 at d = 3 with different code (monomial basis,
least squares, fresh seed): .2135 vs .2131]. Accuracy of the argmax of the L2-OPTIMAL degree-d polynomial of the
TRUE logits (relative L2 error of the logits in brackets):

| model (truth) | d=1 | d=2 | d=3 | d=4 | d=5 |
|---|---|---|---|---|---|
| fused_s0 (.872) | .154 (.66) | .136 (.52) | .259 (.39) | .301 (.34) | .397 (.30) |
| fused_s1 (.951) | .212 (.66) | .141 (.54) | .223 (.44) | .268 (.37) | .336 (.33) |
| torch_s0 (.959) | .211 (.69) | .137 (.58) | .246 (.48) | .310 (.41) | .389 (.37) |
| torch_s1 (.944) | .155 (.58) | .126 (.47) | .213 (.37) | .279 (.31) | .362 (.28) |

So the best possible degree-3 coefficient representation is 61-74 pp off, and the propagated surrogate at d = 3 is
about as good as that optimum (s .25-.27 vs floor .21-.26): the loss is in the representation class, not in the
propagation (alternative A1, not A2).

## Compute accounting

- Charged (as executed, per model): the propagation, i.e. linear mixing (2 H^2 M_active per step), means and sds of
  the pre-activations, Chebyshev coefficients O(k^2), and 3 FLOPs per structure constant of each truncated product.
  S: 3,496 FPE. The endpoint step was not executed (G1 stop); its itemised cost (2,856 FPE for S) is a count.
- Not charged (model-free, built once): the product structure tables (nnz 1,496 / 21,956 / 369,501 at d = 2 / 3 / 4),
  Chebyshev nodes, the cone Gram tables (counted only).
- Diagnostics (not compute of any estimator): 2e6 MC inputs per model per config for s_m and D2, 5e6 per model per
  degree for D1, 3e5 + 3e5 for the cross-check. Wall clock ~25 CPU-min.

## Claims

1. [measured] Config S (d = 3, k = 8, full tensor) costs 3,496 FPE to propagate, and with the cheapest sound endpoint
   it is >= 6,352 FPE: over the 4k cap (G1 STOP).
2. [measured] S's surrogate is 67 pp RMS off on the four models; the necessary-width statistic is 7,193 against a
   STOP line of 1 (G2 STOP). Even ignoring all propagation remainders, the Cantelli endpoint bound is vacuous.
3. [measured] The L2-optimal degree-d Hermite representation of the true logits has argmax accuracy .13-.40 for
   d = 1..5 on all four models (truths .87-.96); its logit relative L2 error falls only from ~.65 (d = 1) to ~.32
   (d = 5).
4. [inferred] Therefore no estimator of the form "degree-d (d <= 5) polynomial surrogate of the hidden states, any
   propagation rule, exact indicator integration, sound ledger" can reach the strict bar on these models: a sound
   interval around such a surrogate has half-width >= (floor gap)/2 >= 20 pp, and the allowance is .43 pp. This
   assumes the surrogate is no better than the L2-optimal projection at classifying, which is not proved (see below).
5. [speculative] Extrapolating the logit relative L2 error (~d^-0.6) to the few-percent level that the argmax would
   need puts d in the hundreds, where M(d) = C(10+d, d) is astronomically large; tensor compression would have to do
   all the work.

## Alternatives not ruled out (exhaustion checklist, AGENTS rule 5)

Tried: total-degree Hermite, full tensor, d = 2, 3, 4 propagated (k = 8, 8, 4; Chebyshev ReLU on mu +- 4 sd, truncated
algebra), the oracle projection at d = 1..5, Cantelli endpoint, on four DEV2 models. Not tried, with reasons:

- **Non-L2-optimal or weighted polynomials.** The L2-optimal projection minimises logit error, not classification
  error; a polynomial fitted to the tie bands could classify better. Not tried: no deductive way to build it from the
  weights was proposed, and a 60-70 pp gap leaves little room.
- **Tensor-compressed representations of higher degree** (TT/CP in the 10 inputs with per-input degree >> 5). The
  floor above covers total degree <= 5 only; a low-rank high-degree function is outside it. This is the main open
  variant of C3, but it would need per-input degree large enough to resolve the ReLU kinks (claim 5) and a sound
  remainder for the compression.
- **Other bases**: piecewise polynomials / splines on gate cells (that is the cell-enumeration family, not C3),
  polynomials in sorted or gap coordinates (a different representation; the decision walls are linear there).
- **Larger ReLU degree k or wider domain at d = 3**: cannot change the outcome, since the propagated surrogate is
  already at the d = 3 oracle floor.
- **Other models/panels**: only four DEV2 models (pre-registered); the floor pattern is the same on all four, both
  kernels.

Other explanations of the result: (i) a bug making polynomial classification look worse than it is: the cross-check
with independent code reproduces D1, the harness's forward reproduces the published truths to <= .03 pp, and the
algebra matches exact products to 1e-14; (ii) MC noise in the D1 coefficients: noise energy M/N <= 3003/4e6 = .08% of
the logit variance, far below the measured 9-50% residual.

## Failure modes

- The total-degree polynomial representation smooths the ~10 layers of ReLU kinks; the argmax margins live in thin
  tie bands that a global L2 approximation does not resolve (A1). This is the harmonic-sphere warning (E34) in a
  different basis.
- The truncated algebra is unstable for low k at d = 4 (fused_s1 diverges).
- Moment-based indicator bounds (Cantelli) are vacuous at accuracy-level precision even with exact moments.
- Costs: the full tensor at d = 4 already costs 20k FPE to propagate and 34k for second moments, so a higher degree
  is impossible in the cheap regime without compression.

## Proposed next step (for the DRI)

Do not fund more total-degree C3 work. If anything, the only open C3 variant is a compressed high-degree
representation (above); its own cheap gate is the same D1 test in that class: the argmax accuracy of the best
rank-r, per-input-degree-D TT fit of the true logits (diagnostic MC) must be within ~1 pp of the truth before any
propagation or ledger is built.

---

# T375: tensor-train (compressed high-degree) oracle gate (follow-up of T291)

## PRE-REGISTRATION (pushed before any run; written 2026-10-19 after T291's verdict, no TT fit computed yet)

**Question.** T291's D1 covers total degree <= 5 only. Is there a compressed high-degree representation, a tensor
train in the 10 inputs with a rich per-input basis, that represents the TRUE logits well enough to classify within
~1 pp of the model, at a rank whose propagation is plausibly <= ~4k FPE? Gate only: no propagation is built unless
it passes.

**Representation (fixed).** Inputs in time order x_1..x_10 (the RNN's own order). Per input a basis of n = 8
univariate functions, two bases:
- **H**: normalised Hermite He_0..He_7 (per-input degree 7, total degree up to 70);
- **P**: piecewise linear {1, x, relu(x - c_1), ..., relu(x - c_6)}, c = N(0,1) quantiles at 1/7..6/7 (axis-aligned
  kinks).

TT with ranks r between cores: logits(x) = s_0 G_1(x_1) G_2(x_2) ... G_10(x_10) W, where G_t(x_t) = sum_i
phi_i(x_t) G_t[:, i, :] (r x r; the first is 1 x r), and W is an r x 10 readout. This is a rank-r multiplicative linear
state-space model, the natural compressed analogue of the 16-unit RNN, and it shares all 10 outputs (the generous
choice for classification). **Ranks r in {1, 2, 4, 8}.** No other basis, rank or ordering is tried.

**Fit (oracle, diagnostic MC; never an estimator).** L2 regression of the true logits (`arc_eval.forward`) by ALS:
1e5 fit inputs per model, random init (seed fixed), 4 full sweeps (left to right and back; W fitted with the last
core), ridge 1e-8 x mean diagonal. Accuracy of the argmax rule and relative L2 logit error on 1e6 fresh inputs
(SE ~ .02 pp); train relative L2 also reported (overfit check). Models: the T291 four (fused_s0, fused_s1,
torch_s0, torch_s1).

**Cost plausibility (fixed in advance, model-free).** Propagating a rank-r TT through one ReLU needs a TT-cross
re-approximation of each of 16 neurons per step: >= 2 sweeps x T n r^3 x 2 FLOPs per neuron-step, i.e.
C_TT(r) ~ 160 x 2 x 10 x 8 x r^3 x 2 / 5948 FPE = 1.1k FPE at r = 8 (8.8k at r = 16), before any endpoint. So the
plausible-cost budget is **r <= 8**.

**Gate.** **GO** iff for some (basis, r <= 8): max over the 4 models of |acc_TT - truth| <= 1.0 pp. Otherwise
**STOP** (scoped negative: TT with these bases at r <= 8, L2 fit). A near miss (max gap <= 3 pp) is reported as
such, but it is a STOP.

**Alternatives written before the run.** (i) Separable bases cannot place kinks that are general hyperplanes in x
(the ReLU walls), so high per-input degree does not help: the rank needed grows fast (STOP, accuracy far off at
r <= 8). (ii) The function is close to a low-rank state-space model (16 units, mostly linear regimes): a moderate
rank already classifies well (GO). (iii) An ALS artefact (local minimum, underfit). Separating test: train vs test
L2 and the trend in r; if r = 8 does not improve on r = 4 while train L2 stays high, suspect (iii) and report it.

## T375 Results (run as pre-registered; nothing changed after c5c5a46e)

**Verdict: STOP.** Negative (unverified), scoped to TT with bases H/P, n = 8 per input, r <= 8, L2 ALS fit. Reproduce:
`for b in H P; do for m in fused_s0 fused_s1 torch_s0 torch_s1; do python tt.py $m $b 1 2 4 8; done; done`
(about 2 CPU-h; under 2 GB per process). torch_s1 at r = 8 was re-run after a container restart. Seeds are fixed, so
the run is identical to the one that was lost. Raw rows: `tt_log_{H,P}.txt`, `tt_<model>_<basis>.json`.

[measured, diagnostic MC: 1e5 fit inputs, 1e6 fresh evaluation inputs]. Argmax accuracy of the best TT fit of the
TRUE logits, gap to the truth in pp, and test / train relative L2 error of the logits:

| model (truth) | basis | r=1 | r=2 | r=4 | r=8 |
|---|---|---|---|---|---|
| fused_s0 (.872) | H | .159 (-71.3) | .198 (-67.4) | .287 (-58.5) | .382 (-49.0); L2 .46 / .36 |
| | P | .159 (-71.3) | .200 (-67.2) | .288 (-58.4) | **.404 (-46.8)**; L2 .37 / .36 |
| fused_s1 (.951) | H | .159 (-79.3) | .187 (-76.5) | .233 (-71.9) | .328 (-62.3); L2 .40 / .35 |
| | P | .158 (-79.3) | .190 (-76.1) | .246 (-70.5) | .349 (-60.3); L2 .34 / .34 |
| torch_s0 (.959) | H | .157 (-80.2) | .186 (-77.3) | .202 (-75.7) | .327 (-63.2); L2 .48 / .42 |
| | P | .157 (-80.2) | .192 (-76.7) | .252 (-70.7) | .357 (-60.2); L2 .43 / .42 |
| torch_s1 (.944) | H | .157 (-78.7) | .224 (-72.0) | .255 (-68.8) | .352 (-59.2); L2 .46 / .35 |
| | P | .156 (-78.8) | .223 (-72.1) | .261 (-68.3) | .384 (-56.0); L2 .35 / .35 |

- The gate needs max gap <= 1 pp at some r <= 8. The best configuration (P, r = 8) has a max gap of **60.3 pp**
  (best single model 46.8 pp). STOP by a factor of about 50.
- At r = 8 the compressed fit is no better than T291's total-degree-5 polynomial: the logit relative L2 error is
  .34-.43 here and .28-.37 there.
- Alternative (iii), an ALS artefact: H overfits at r = 8 (test L2 is .05-.11 above train), so H's r = 8 row
  understates that class somewhat. P shows no overfit, and every model and basis improves steadily with r. So the
  fits are not stuck, they are simply far from the target. Alternative (i), separable bases cannot place the oblique
  ReLU walls, fits the data; (ii) is refuted at r <= 8.
- [speculative] The gap shrinks by about 8-12 pp per doubling of r. Extrapolated, 1 pp needs r in the hundreds or
  more, and the propagation cost ~ r^3 would be far above the cheap regime.

**Claims.** [measured] No TT of rank <= 8 in either basis classifies within 46 pp of the truth on any of the four
models. [inferred] The cost-plausible TT class (C_TT <= ~1.1k FPE propagation at r = 8) is dead for the strict bar
on these models, whatever the propagation or ledger. Not covered: other input orderings (for example by gap or rank
coordinates), bases with oblique kinks, losses tuned for classification, ranks > 8, other compressed formats
(hierarchical Tucker, CP). Some of these change the representation (sorted coordinates) rather than compress it.

**Compute accounting.** Diagnostics only, no estimator was run: 1.1e6 true-model evaluations per (model, basis)
plus the ALS work, about 2 CPU-h in all.
