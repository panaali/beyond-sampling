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
