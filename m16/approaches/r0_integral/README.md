# r0_integral (T297): can the R0 point closure be integrated deterministically within 4k FPE?

↑ [Approaches index](../README.md) · [TASKS](../../../TASKS.md) · R0 is defined in T292
(branch claude/t292-oracle-floor, `m16/approaches/oracle_floor/oracle.py`) · runs in parallel with T296.

## Question
R0 keeps the top-3 skeleton exact (positions p1, p2, p3 and values v1 > v2 > v3). It pins the 7 low inputs at
their conditional mean mu(v3) = -phi(v3)/Phi(v3) and calls the skeleton correct iff argmax logits = p2. Its oracle
bias on DEV2 is .51 pp RMS (T292).

acc_R0 = (1/720) sum over triples of the integral of f(v) 1[R0 correct], with f(v) = 720 phi(v1) phi(v2) phi(v3)
Phi(v3)^7 on v1 > v2 > v3. Can this integral be computed deterministically to <= .25 pp within 4k FPE (about 5.6
FPE per triple)?

## Pre-registration (written 2026-10-18, before any run)
**Integrator "TB" (threshold boundaries), per position triple; no random or Sobol nodes.**
- Coordinates: v2, g23 = v2 - v3, g12 = v1 - v2.
- **2nd-vs-3rd boundary:** tau23(v2), linear in v2 through 2 anchors (v2 = v2_a, v2_b, at g12 = typical). At each
  anchor, R0's margin m(g23) = L_p2 - max_{j != p2} L_j is evaluated with its exact forward-mode tangent along
  g23 at g23 = 0+ (value + tangent = 2 FPE). Newton gives the root (exact if no gate flips between 0 and the root;
  one more Newton step if the budget allows).
- **1st-vs-2nd boundary:** tau12, one anchor, 2 FPE, the same way along g12.
- **Blind part** (v2 <= 0): one R0 verdict at a canonical blind skeleton (1 FPE). The blind mass is model-free.
- Region assumed correct: {v2 > 0, g23 >= tau23(v2), g12 >= tau12}, plus the blind part if its verdict is correct.
  Its mass is a model-free 1-D integral over v2: 720 phi(v2) (1 - Phi(v2 + tau12)) Phi(v2 - tau23(v2))^8 / 8. This
  is computed with fixed Gauss-Legendre nodes (deterministic numerical integration of a known density, no model
  calls) and **charged** in FLOPs.
- Margin sign convention: tau = 0 if the margin is already positive at the tie side... (see code). If the margin is
  negative beyond the anchor range the root is capped, and the triple is wrong there.
- Charged cost: T196 FLOPs, multiply-add = 2, as implemented; value + tangent counted as 2 forwards; GL nodes and
  Phi evaluations counted (20 FLOPs per Phi).

**Reference (diagnostic, sampling):** acc_R0 from 1e7 skeletons drawn from the exact law (R0 evaluated at each).
s.e. <= 0.01 pp.

**Models:** 10 DEV2 retrains (claude/t192-gpu-cost-law hp2/dev2/weights.npz, array sha256 checked as T292):
indices 0, 8, 16, ..., 72, a mix of fused and trainer. Also the target, for information.

**Go line (DRI):** integration error |acc_TB - acc_R0^MC| with RMS over the 10 models <= .25 pp, at charged cost
<= 4k FPE.
- If TB misses, report the error-vs-cost curve (more anchors / Newton steps) and the decomposition into
  2nd-vs-3rd, 1st-vs-2nd, blind and "other" errors, to say what a passing design would need.
- **Dead (scoped to threshold designs)** if no variant reaches .25 pp below 20k FPE.
- Also reported, not decisive: error vs the DEV2 truths (closure + integration).

**Alternatives.**
- R0's error set may not be "near the ties". If R0 is wrong on bulk regions of some triples, threshold designs
  fail; the "other" error class measures this.
- Thresholds may depend on v3 and g12 as well as v2; the anchor variants test this.
