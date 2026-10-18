# meanfield — T294: mean-field ReLU closure (switching probabilities, not frozen gates), stage 1 on the blind region

**Status:** done (T294 stage 1 + T294b: all inconclusive, no go; stage 2 not run). Bottleneck: Gaussian/Clark readout (see T294b diagnostic).

↑ [TASKS](../../../TASKS.md) · [blind11 (T293)](../blind11/README.md) · E35 (arc_cumulants)

**All claims unverified.**

## Pre-registration (written 2026-10-18, before any T294 number was computed)
**Method (deductive).** For each of the 180 blind units (case, label) of T293 (same model-free role moments for x):
- State: a joint Gaussian over v = (x_0..x_9, h_1..h_16): mean (26,) and covariance (26, 26). It starts with x's conditional
  moments and h = 0.
- Step t: z = W_hh h + W_ih x_t is linear in v, so mean(z), cov(z) and cov(z, v) are exact under the Gaussian state. ReLU moment
  matching:
  - E[relu z] = mu Phi(mu/s) + s phi(mu/s);
  - Var exact;
  - cov(relu z_i, w) = Phi(mu_i/s_i) cov(z_i, w) for every state variable w (Stein's lemma, exact for jointly Gaussian);
  - cov(relu z_i, relu z_j) = Phi_i Phi_j cov(z_i, z_j) for i != j (first-order closure).
  The new state is (x, relu z). Switching probabilities Phi(mu/s) replace frozen gates.
- Readout: L = W_out h_10, margins m_i = L_j - L_i (j = label). P(correct | unit) = P(all m_i >= 0) by T293's Clark closure.
  Contribution c = (11/1024)(1 - weighted mean). Cost: instrumented FLOPs, reported in FPE (6000 FLOPs).
**Why it might work where E35 failed:** the case fixes every input's sign (one positive or none), so the step-1 pre-activations
W_ih x_0 have a known sign. The near-degenerate perfectly correlated step-1 units of an unconditioned closure become
deterministic-sign variables whose moments the closure carries. (The Gaussian treatment of half-normal x still leaks some sign
mass; noted.)
**Test (stage 1).** DEV2 80 retrains; reference = T293's sampled contributions (blind11/blind.json, 2.2e5 per model, diagnostic).
**Go:** corr(pred, sampled) >= 0.7 across the 80 models AND RMS(pred - sampled) <= 0.15 pp. **Stop:** corr < 0.3. Otherwise
inconclusive. Stage 2 (D0/D1 classes with F under Directive 8 amendment 2) runs only on a go. Cap ~3 CPU-h.

## Results (unverified; `mf.py` -> mf.json, mf.log)
Cost: **11,883 FPE** per model (180 units x 10 steps of 26 x 26 covariance algebra; unoptimised, the x block is static and could
be cut several-fold).

| panel | RMS (pred - sampled), pp | mean | corr(pred, sampled) | pred sd | sampled sd |
|---|---|---|---|---|---|
| DEV2 (80) | **0.232** | +0.206 | **0.73** | 0.049 | 0.139 |
| LB3 (30) | 0.106 | +0.090 | 0.80 | 0.030 | 0.078 |
| (T293 frozen gates, DEV2) | 0.234 | +0.162 | -0.30 | 0.064 | 0.139 |

## Verdict (pre-registered rule): **inconclusive** (corr 0.73 >= 0.7 passes, RMS 0.232 > 0.15 fails; not stop)
- **Switching probabilities restore the model signal**: corr -0.30 (frozen gates) -> **+0.73** (mean field) on DEV2, 0.80 on LB3.
  This is the first deductive blind-region closure that tracks the model-to-model variation.
- **But the response is compressed ~2x and biased upward** (+0.21 pp): pred sd 0.049 vs 0.139.
- *Post hoc, not part of the rule (labelled):* an affine calibration fitted on LB3 (sampled ~ -1.10 + 2.07 pred) transfers to DEV2
  (in-sample slope 2.06), giving DEV2 RMS **0.138 pp** (`calibration_posthoc.json`). The compression is systematic, not noise.
  Using it would be advice fitted on scored models, so it is not claimed as a pass. It points to a fixable closure defect: the
  first-order cross-covariance Phi_i Phi_j Sigma_ij underestimates how differently models behave, and the Gaussian treatment of the
  half-normal inputs leaks sign mass.
- **Stage 2 not run** (no go).

## Next (proposed, needs its own pre-registration)
- Exact bivariate ReLU second moments (E[relu z_i relu z_j] in closed form via the bivariate normal CDF) instead of first order.
- Exact half-normal handling of the current input at each step (its sign is fixed by the case): relu(w x_t + rest) with x_t
  half-normal, not Gaussian.
- If either removes the ~2x compression without calibration, stage 1 would pass, and stage 2 (D0/D1 classes) becomes worth running.

## T294b pre-registration (written 2026-10-18, before any T294b number was computed)
Three closure variants of stage 1 (`mf2.py`), all else identical (same 180 units, role moments, Clark readout, no calibration):
- **(a) exact bivariate ReLU second moments.** For i != j, cov(relu z_i, relu z_j) = E[relu z_i relu z_j] - E relu z_i E relu z_j
  with the closed form for jointly Gaussian (z_i, z_j) (alpha = mu/s, rho = corr, r = sqrt(1 - rho^2)):
  s_i s_j [(alpha_i alpha_j + rho) Phi2 + alpha_i phi(alpha_j) Phi((alpha_i - rho alpha_j)/r) + alpha_j phi(alpha_i)
  Phi((alpha_j - rho alpha_i)/r) + r^2 phi2], Phi2 by 20-node Gauss-Legendre in theta = asin(rho) (|rho| clipped at 0.9999).
  Checked against Monte Carlo before use. Stein cross-covariances and the diagonal stay as in stage 1 (already exact).
- **(b) sign-fixed law of the current input.** At step t, x_t's own marginal law given the case (exact half-normal at the positive
  position; the model-free role law, which is one-sided, elsewhere) is split into K = 16 equal-probability bins (model-free MC,
  4e6 draws). In each bin x_t is Gaussian with the bin's mean and variance, the rest of the state is conditioned on it by
  Gaussian regression, the ReLU step is applied in closed form, and the 16 results are collapsed back to one Gaussian state by
  the law of total covariance. So no mass of x_t sits on the wrong sign (up to within-bin Gaussian tails). Cross terms first order.
- **(c) both** (b's bins with (a)'s exact cross terms).
**Reference:** T293 sampled contributions (blind11/blind.json, diagnostic). **Go line (each variant, DEV2 80 models, no
calibration):** corr(pred, sampled) >= 0.7 AND RMS(pred - sampled) <= 0.15 pp. Stop: corr < 0.3. Else inconclusive.
**Mechanism test (reported, not part of the rule):** OLS slope of pred on sampled (stage 1: corr x sd ratio = 0.26; ideal 1),
the slope of sampled on pred (stage 1: ~2.07; ideal 1) and the sd ratio pred/sampled (stage 1: 0.35). Compression counts as
removed if the sd ratio is in [0.75, 1.33] and the sampled-on-pred slope is in [0.75, 1.33].
**Also reported:** LB3 (30) the same numbers; per-variant cost breakdown (covariance propagation, ReLU moments, Phi2, bins,
Clark readout) in FPE; ideas to cut cost ~10x. The LB3-fitted calibration is not used in any scored number.
**Stage 2** (D0 classes, top-2 positions x positive count, F under amendment 2) runs only if (c) passes, after its own
pre-registration committed before running. Cap ~3 CPU-h for T294b.

## T294b results (unverified; `python m16/approaches/meanfield/mf2.py /tmp/claude-0/dev2/weights.npz a,b,c` -> mf2.json, mf2_summary.json, mf2.log)
No calibration in any number. Reference: T293 sampled contributions.

| variant | DEV2 RMS pp | mean | corr | slope pred~sampled | slope sampled~pred | sd ratio | LB3 RMS / corr | cost FPE |
|---|---|---|---|---|---|---|---|---|
| stage 1 (first order, Gaussian x_t) | 0.232 | +0.206 | 0.73 | 0.26 | 2.07 | 0.35 | 0.106 / 0.80 | 11,883 |
| (a) exact bivariate cross moments | 0.230 | +0.203 | 0.74 | 0.28 | 1.99 | 0.37 | 0.103 / 0.77 | 24,281 |
| (b) sign-fixed x_t, K=16 bins | 0.200 | +0.170 | 0.73 | 0.29 | 1.83 | 0.40 | 0.102 / 0.81 | 192,612 |
| (c) both | **0.191** | +0.155 | **0.64** | 0.27 | 1.48 | 0.43 | 0.095 / 0.77 | 391,755 |

**Verdict (pre-registered rule): all three inconclusive. No variant goes** (RMS > 0.15 throughout; (c) also falls below corr 0.7);
none is a stop (corr >= 0.3). **The compression is not removed**: sd ratio 0.35 -> 0.43, and the sampled-on-pred slope 2.07 -> 1.48,
both outside [0.75, 1.33]. **Stage 2 not run.** [measured]

**Cost breakdown (FPE per model, mean):** (a) cov propagation 10.7k, Phi2 pairs 12.6k, ReLU moments 0.3k, readout 0.6k, Clark
0.07k. (b) bin conditioning 6.7k, cov propagation 171.7k (16 bins x 10.7k), ReLU moments 7.5k, collapse 6.0k. (c) adds Phi2
201.6k (16 x 12.6k). [measured, instrumented counts]

**Diagnostic: where the remaining error sits** (`diag/diag.py` -> `diag/diag.out`, 2 DEV2 models x 5 units, 4e5 samples per
unit; diagnostic, not scored). For each unit we compare:
1. the sampled P(correct | unit);
2. the exact Gaussian orthant probability using the sampled logit-margin mean and covariance, i.e. the h closure is perfect and
   only the Gaussian-logit assumption remains;
3. Clark applied to those sampled moments.

The readout alone is badly wrong and pulls every P toward ~0.1–0.4:

| model, unit | sampled P | Gaussian, sampled moments | Clark, sampled moments | (c) closure |
|---|---|---|---|---|
| DEV2 #0, q=9, label j=0 | 0.789 | 0.573 | 0.496 | 0.37 |
| DEV2 #0, q=3, label j=5 | 0.013 | 0.068 | 0.108 | 0.13 |
| DEV2 #5, none, top a=0, label j=1 | 0.005 | 0.042 | 0.112 | 0.14 |
| DEV2 #5, q=3, label j=5 | 0.689 | 0.329 | 0.276 | 0.29 |

So [measured on these 10 units, inferred in general]: (i) the logits given a unit are far from Gaussian (the ReLU gates make them
mixtures), and (ii) Clark's max closure adds an error of the same size. Both compress the per-unit P toward the middle. That
compression is the same direction as the ~2x model-level compression. Better h moments (a, b, c) only move each unit's
prediction from roughly the "0" column toward the "Gaussian, sampled moments" column, which is itself compressed.
**Inferred:** no Gaussian-state closure with a Gaussian/Clark readout can pass the go line on these units. The bottleneck is the
readout, not the h-h closure.

**What would cut cost ~10x (for a stage that fits the cheap regime) [inferred, untested]:**
- **Shared prefixes across units.** All units of a case see the same input law at every position before the first special
  position (a, j or q). A prefix tree over positions shares those steps: about 2–3x on stage 1.
- **Static x block.** Propagate only the 16x16 h block plus the 16x10 h–x cross block; the x block never changes. About 2x on
  cov propagation.
- **Lower-rank covariance.** If h's covariance is dominated by r ≲ 4 directions, the step cost drops from O(H n^2) to O(H n r).
- **Cheaper cross moments.** Use Phi2 with 4–6 nodes, or only on pairs with |rho| > 0.3, or a second-order expansion in rho.
- **Fewer bins.** K = 3–4 Gauss-type nodes per role law instead of 16.

Combined, these plausibly bring (a) to ~2–3k FPE and (b) or (c) to ~10–20k FPE. Not worth doing while the readout bias stands.

**Proposed next (needs its own pre-registration; not run):** replace the readout, not the closure. Carry a small Gaussian mixture
through the last 1–2 steps: split on the sign of the 2–4 least certain units at the final step, so each component's logits are
closer to Gaussian. Or compute the orthant probability exactly (Genz) instead of by Clark. Check first, with the per-unit
diagnostic above, whether "mixture + exact orthant, sampled moments" reaches the sampled P on these units. That check costs no
model-level run and could have failed.

**Alternatives not ruled out:**
- T293's reference has sampling noise (2.2e5 per model, ~0.02 pp). It is too small to explain RMS 0.19.
- The role moments are MC estimates.
- The diagnostic covers only 10 units on 2 models.
- The bins use Gaussian regression for the rest of the state given x_t, which is not exact for the order-statistic x's.

## Compute accounting
110 models x ~11.9k FPE (~1.2 CPU-s each); ~5 CPU-min total. T294b: 110 models x (24k + 193k + 392k) FPE; ~12 CPU-min. Diagnostic ~2 CPU-min.
