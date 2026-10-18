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

*Amendment before scoring on all 80 (after a 3-model smoke test showed 44-53k FPE):* the walked range is a parameter
tmax. Beyond tmax the gap is **assumed correct**, a declared structural approximation (R0's errors sit at small
gaps). Scanned tmax = 0.05-6.

## Status
Done (2026-10-18). **FAIL by the pre-registered line.** No design fits <= 4k FPE: the first cell alone of the four
walked lines costs 4 x 2 FPE x 720 = 5,760 FPE. At any cost the best F on DEV2 is 20.6.

## Results
**Validation.** The model-free mass integral with full intervals gives 1 - 1.6e-15. walk_line is the T218/T272
exact walk structure, with the correct set per piece in closed form.

**Walk-length scan, 10 T297 DEV2 models** (`python scan.py`, `results_scan.json`). Integration error vs 1e7 MC of R0;
per-class errors = estimator error mass minus MC error mass:

| tmax | charged FPE | integration RMS (pp) | mean (pp) | nonblind class: mean / RMS (pp) | blind class: mean / RMS (pp) | total vs truth RMS (pp) |
|---|---|---|---|---|---|---|
| 0.25 | 22,604 | 0.62 | +0.48 | -0.46 / 0.61 | -0.02 / 0.08 | 0.96 |
| 0.5 | 25,910 | **0.29** | +0.12 | -0.21 / 0.35 | +0.09 / 0.12 | 0.66 |
| 1.0 | 32,564 | **0.28** | +0.01 | -0.11 / 0.32 | +0.10 / 0.13 | 0.60 |
| 2.0 | 40,394 | 0.29 | -0.04 | -0.08 / 0.32 | +0.12 / 0.14 | 0.58 |
| 6.0 | 46,691 | 0.29 | -0.05 | -0.07 / 0.31 | +0.12 / 0.14 | 0.57 |

**All 80 retrains, lenient point F** (`python f80.py`, `results_f80.json`):

| tmax | C (FPE) | bias (pp) | RMS (pp) | F [95% CI] |
|---|---|---|---|---|
| 0.05 | 10,033 | +3.23 | 3.85 | 205 [150, 267] |
| 0.5 | 26,318 | +0.42 | 0.75 | **20.6** [15.0, 27.3] |
| 1.0 | 33,333 | +0.28 | 0.67 | 20.7 [15.2, 27.2] |

**Reading.**
- **The integration is now accurate, with no cancellation.**
  - At tmax >= 0.5, the integration RMS is 0.28-0.29 pp against the MC of R0. This is 4-10x better than T297's
    Newton thresholds (1.16-2.75 pp).
  - Per class: the nonblind class is off by -0.1 to -0.2 pp mean, the blind class by +0.1 pp. Both are small
    individually, so the result no longer depends on omitted classes offsetting each other.
  - The integration error is now at about R0's own closure floor (total 0.57-0.66 pp).
- **The cost is the obstacle.**
  - The exact walk from the tie point crosses about 7 gate cells within the first 0.25 of gap length (H-E15's
    tie-wall fragmentation, now as cost). 4 lines x 720 triples cost 22-47k FPE.
  - At those costs R0's ~0.5-0.6 pp floor alone gives F ≈ 15-20, so **no walk length passes the lenient bar**, even
    without the 4k cap.
  - Cheaper walks (tmax = 0.05, 10k FPE) put the tie bands outside the walked range and are much worse
    (F 205).
- **Comparison with T300 stage 0** (F 5.97 at 1.75k FPE): that design's low F came from its very low cost together
  with error cancellation. A deductive, non-cancelling design needs ~15x the cost and lands at F ≈ 21.

## Compute accounting
Scans: 10 models x 7 settings plus 80 models x 3 settings, ~25 CPU-min.

## Claims (unverified)
- [measured] The exact segment-walk integral of R0 (all classes, declared structural approximations only) has
  integration error 0.28-0.29 pp RMS vs MC of R0 at 26-33k FPE, with small per-class errors. Lenient F on all 80
  DEV2 retrains is 20.6 [15.0, 27.3] at best. FAIL; no <= 4k design exists (5,760 FPE for the first cells alone).
- [inferred] For R0-based estimators the tie-wall fragmentation sets the price of a deductive integral at
  >= ~20k FPE. With R0's ~.5 pp floor this caps F at >= ~15 on retrains. A cheap pass needs either a closure better
  than R0 or a way to cross the tie walls without walking their cells (e.g. sharing walks across triples).
