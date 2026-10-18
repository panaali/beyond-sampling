# r0_walk (T302): R0 integral by exact segment walks along the gap lines (no Newton, no omitted classes)

↑ [r0_shared (T300 stage 0)](../r0_shared/README.md) · [r0_integral (T297)](../r0_integral/README.md) · [TASKS](../../../TASKS.md)

**Scope (declared post hoc after T296): lenient bar, retrains (DEV2).**

## Pre-registration (written 2026-10-18, before any run)
**Estimator "WALK".** Per position triple (p1, p2, p3), two value regions: nonblind (v2 > 0) and blind (v2 <= 0).
Each region has a declared transverse point (the conditional means of v2, g12, g23 given the region, from the exact
model-free law).

| step | label |
|---|---|
| R0 closure (7 low inputs at mu(v3)) | structural approximation (deductive) |
| mu(v3) linearised at the transverse point along the g23 line, so the inputs are affine in the gap | structural approximation |
| **g23 line**: R0's margin is piecewise linear along v3 = v2* - g, g in [0, 6]. The exact segment walk (value + tangent, every gate flip located, correct set solved in closed form per piece) gives S23, the exact set of g where R0 is correct on the line | deductive |
| **g12 line**: the same along v1 = v2* + g at (v2*, g23*); exact (mu does not depend on v1) | deductive |
| correct region of the triple taken as {g23 in S23} x {g12 in S12}, independent of v2 within the region | structural approximation (the "point closure" of the boundary) |
| masses: per region, a 1-D Gauss-Legendre integral over v2 of closed forms. Per interval: [Phi(v2+b) - Phi(v2+a)] for g12 and [Phi(v2-a)^8 - Phi(v2-b)^8]/8 for g23 (top-3 law) | deductive, model-free, charged |

Both tie classes and the blind class are included, so no error class is omitted.

**Cost:** T196 FLOPs as implemented. A walk charges 2 forwards (value + tangent) for its first cell, and
2 (T - s)/T per later cell (restart from the flipping step s). GL and Phi evaluations are charged.

**Go line (DRI):** lenient point F <= 10 on DEV2 (all 80 retrains, 1e8 truths) at <= 4k FPE, every step labelled
deductive or as a declared structural approximation (none inductive). Report:
- the per-class error split (nonblind 23 / 12 / blind; estimator vs 1e7 MC of R0 on the 10 T297 models), to show
  whether cancellation still drives the result;
- the error against MC of R0 (integration) and against the truths (total).

On a pass: freeze, and send to the T281 scorer for HP2. Cap: 2 hours.
