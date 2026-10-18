# T334: independent check of T324's two largest error-budget terms (line-set product and tmax cut)

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T312](../r1w_repro/README.md) · [T320](../t317c_confirm/README.md)

## Question
T324 (claude/t292-oracle-feature, `budget_t317d/`) measured each approximation of T317d/T317e by switching it off.
One at a time, from V1 (= T317e), it found:
- **product of the two 1-D line sets** off: Δbias −.57 pp, ΔMSE −40%;
- **tmax cut** off: Δbias −.18 pp, ΔMSE −18%.

T333 builds on the product's 40%. Under AGENTS rule 4, this re-measures both terms with my own code. **I have not read
`budget_t317d/`'s code**, only its README numbers.

## Pre-registration (2026-10-18, before any run)
**The pointwise form, my own derivation, from T312's re-implementation.** T317e's estimate is exactly
E_x[s23_r(g23(x)) · s12_r(g12(x))] under the exact input law. Here x is 10 i.i.d. N(0,1) inputs:
- p1, p2, p3 are the positions of the top 3, and g12 = v1 − v2, g23 = v2 − v3;
- r = nb if v2 > 0, else b;
- s23_r(t) = 1[argmax of the model = p2] at the region's line-23 input at gap t. That input has v1 = v2* + g12*, v2 = v2*,
  v3 = v2* − t, v4 = v3 − δ at the latest free position, and lows = μ(v4*) + μ'(v4*)(v4 − v4*). s23 = 1 for t ≥ tmax;
- s12_r(t) = 1[correct] at v1 = v2* + t, v2*, v3*, v4*, with lows μ(v4*). s12 = 1 for t ≥ tmax, and s12 ≡ 1 in region b.

Why this is exact: given v2, the GL integrand of `est` is the product of the line-12 mass ∫φ(v1)1[s12] and the
line-23 mass ∫φ(v3)Φ(v3)^7 1[s23]. Each walk returns the exact correct set along its line.

The transverse constants (v2*, g12*, g23*, v4* = μ(v3*), δ, μ(v4*), μ'(v4*)) come from my own quadrature
(scipy.integrate), checked against `cont_e.TABLES` (T317e, not T324) to 1e-6.

**Variants**, all evaluated by forward passes on the same inputs (paired):
- **V1:** as above. **Check, which must pass before any Δ is used:** on every model, the MC of V1 agrees with
  `cont_e.est` within 4 SE, and the RMS of the differences is ≤ 2 × the mean MC SE.
- **(a) product off.** In region nb, if g12 < tmax and g23 < tmax, use the joint verdict at the line-23 closure with
  v1 = v2* + g12(x), v2*, v3 = v2* − g23(x), v4 = v3 − δ and linearised lows. If only one gap is < tmax, use that line's
  verdict. If neither, 1. Region b is unchanged: the blind rule is kept, so V1 never uses g12 there.
- **(b) tmax off.** s23 and s12 are evaluated at any t, with no cut. The blind rule is kept.

**Data.**
- All 80 DEV2 models, with open 1e8 truths. If the runtime would exceed ~45 CPU-min, I fall back to fused_s0..11 +
  torch_s0..11 (24 models) and say so.
- N = 2e5 inputs per model, seed 334000 + index.

**Reported.**
- Bias and MSE for V1, (a) and (b), against truth.
- Δbias = bias(variant) − bias(V1).
- ΔMSE as a fraction of MSE(V1).
- 95% CIs from 10,000 bootstrap resamples over models. MC noise is included by drawing per-model Gaussian noise
  with each model's paired SE.

**Verdict.** **CONFIRMED** for a term if T324's Δbias and ΔMSE fraction both lie inside my 95% CIs. Otherwise
**DISCREPANCY**: I report both values, and give the most likely definitional difference before any claim.

**Alternatives first.**
1. My product-off definition may differ from T324's, for example in the handling of region b or of one-sided
   tmax. In that case the numbers measure different things. The region-b and one-sided rules are fixed above for
   that reason.
2. If my pointwise form fails its check, the Δs mean nothing, and I stop.

## Status
Pre-registered; not run.
