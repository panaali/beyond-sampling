# meanfield — T294: mean-field ReLU closure (switching probabilities, not frozen gates), stage 1 on the blind region

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

## Compute accounting
110 models x ~11.9k FPE (~1.2 CPU-s each); ~5 CPU-min total.
