# budget_t317d: T324, per-approximation error budget and deductive-status audit of the T317d candidate

↑ [Approaches index](../README.md) · [TASKS](../../../TASKS.md) · candidate: claude/t218-tie-tail-labels @5b4d4806
m16/approaches/r1_cont/ (cont_d.py, estimator_d.py) · red-team: review/candidate_redteam/README.md (ccr)

I am a different agent from the T241/T218 child that built the candidate and did the T312/T320 checks. I read the
candidate's code (needed for both parts) but wrote all the budget code myself.

## Pre-registration (written 2026-10-18, before any T324 number)
**Pointwise form of the estimator** (derived from reading r1w.est / cont_d.est). Per ordered top-3 triple and region
r (nb: v2 > 0, b: v2 <= 0), with r's transverse point (v2*, g12*, g23*) (cont_d.TABLES, exact law):
- the line-23 verdict s23_r(t) = 1[model correct at the line-23 input]. The input has v1 = v2* + g12*, v2*, v3 = v2* - t,
  and v4 at the latest free position = v3 - delta (delta = v3* - v4*, v4* = gauss_v4(v3*, 1)), with the 6 lows at
  mu(v4*) + mu'(v4*)(v4 - v4*). s23 = 1 for t >= tmax = 0.5.
- the line-12 verdict s12_r(t) at v1 = v2* + t, (v2*, v3*, v4*), with the lows at mu(v4*). s12 = 1 for t >= tmax, and
  s12 = 1 in region b ("blind g12 assumed correct").
- est = E_x[s23_r(g23(x)) s12_r(g12(x))] over the exact input law, up to G = 20 Gauss-Legendre in v2.
  **Check (must pass before any budget number is used):** the MC of this pointwise form must agree with cont_d.est
  within 4 combined SE on every model, and its RMS difference must be <= 2x the MC SE.

**Switch-offs** (each replaces one approximation by its exact counterpart, evaluated pointwise on the same x,
paired). For each model: 2e5 inputs, seed 324000 + index. Truths: the DEV2 1e8 open truths.
1. **tmax off:** no cut (lines evaluated at any t).
2. **blind g12 off:** region b uses its own walked line-12 verdict s12_b.
3. **mu linearisation off:** line 23 uses the exact closure-c law along t: v4 = gauss_v4(v3(t), 1), lows = mu(v4).
   This removes both the constant-delta v4 and the linear lows.
4. **product off:** joint 2-D verdict at (v2*, g12, g23) with line 23's closure, instead of s12(g12) s23(g23). The tmax
   cut and the blind rule are kept: beyond tmax in one gap, the other line alone; in region b, g12 is free.
5. **one transverse point off** (the v2 direction): both lines re-anchored at the input's own v2, at (v2, g12*, g23*),
   with the closure parameters recomputed there. The cross-gap part of "one transverse point" is the product
   approximation (4): with both coordinates at the input's own point, the lines meet at x and the product is the joint
   verdict.
6. **closure c off:** the true model on x (truth), measured as the last step of the path below.
- **Cumulative path** V1 -> 1 -> 1+2 -> 1+2+3 -> 1+2+3+4 -> 1+...+5 = E1 (closure c, exact law, MC) -> truth. The last
  path point must equal a direct MC of closure c pointwise (identity check).
- **Reported:** bias, RMS and MSE over the 80 models for every variant. Each approximation's share of the bias
  (+.44 pp) and of the MSE, both one-at-a-time (from V1) and along the path. The red-team's claim (about 80% of the
  error and all of the bias come from approximations 1-5) is tested as: MSE(V1) - MSE(E1) >= 0.6 MSE(V1), and
  |bias(E1)| < |bias(V1)| / 2. "Supported" iff both hold.

**Audit (part 2):** read every runtime file of estimator_d's import closure (cont_d, cont_c, cont, r1w, r1w2, walk,
v1b2, r0). List any random draw, any data-dependent extrapolation from observed model behaviour, and any constant
chosen on model data, with file:line.
Labels: the budget is diagnostic sampling. Cap 2.5 h.
