# T312: independent reproduction of R1W V1 (the best deductive estimator so far)

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · frozen original: claude/t218-tie-tail-labels `m16/approaches/r1_walk/estimator.py` @684a22b9 (sha256 81e30b6a…, r1w2.py 30780215…)

## Question (DRI)
Re-implement R1W V1 from its README alone (r1_walk sections T304/T306, plus the r0_walk T302 walk structure) and
check three things:
- (a) every step is deductive or a declared approximation; list any undeclared assumption;
- (b) per-model DEV2 estimates within .1 pp of the frozen code;
- (c) FLOP count within 10%.

## Method
- `r1w_indep.py` was written from the README text only.
  - Transverse points come from exact numerical integration of the top-3 law.
  - The K = 1 node of v4 is the exact conditional mean.
  - The segment walk is my own vectorised code (value + forward tangent, the next gate flip, and the correct set per
    piece in closed form).
  - Masses: 48-node Gauss–Legendre per region over v2 ∈ (0, 6.5] or [−6.5, 0].
  - Validation: full intervals give total mass 1 − 2e-15.
- `compare.py` imports the frozen `estimator.py` only for the final side-by-side comparison on all 80 DEV2 models.
  After the comparison I read the frozen cost code (r1w2.py) to diagnose (c).
- One bug of mine, fixed before the final comparison: I first charged each later cell by the flip that ends it,
  instead of the flip that starts it ("restart from the flipping step s"). The README was clear; the bug was mine.
  The estimates were unaffected.

## Results (all 80 DEV2 retrains, 1e8 truths; `python3 compare.py`, `results/compare.json`, `logs/compare.log`)
| | mine | frozen |
|---|---|---|
| bias / RMS vs truth (pp) | +0.444 / 0.794 | +0.445 / 0.795 |
| mean charged cost (FPE = FLOPs/5948) | 11,065 | 13,280 |
| point F (mean e² × mean C / .0725) | 9.63 | 11.58 |

**(b) Estimates: PASS.** All 80 models agree within .1 pp: max |diff| .012 pp, RMS .004 pp, mean −.001.

**(c) FLOPs: FAIL as literally posed, but explained.**
- My total is 0.80–0.87x the frozen total (mean .834), outside the 10% band.
- Decomposed (fused_s0 / torch_s0), the **model-dependent walk work agrees to 0.3%**: 9,917 / 8,721 walk FPE
  (mine) vs 9,893 / 8,714 (frozen).
- The whole gap is in the **charge for the model-free Gauss–Legendre / Φ work**: mine 7.5e6 FLOPs, frozen 2.1e7.
  The frozen code counts:
  - one density evaluation per triple per node (the density is model-free and shared across triples);
  - a flat 4·NT·G·10 per region;
  - FLOP_PHI = 20 per Φ.
- It also converts walk FPE to FLOPs at its own 5,760 FLOPs per forward, then reports C at 5,948 per FPE.
- **None of this charging convention is written in the README** ("GL and Phi evaluations are charged"). Both counts
  are defensible "as implemented" counts.
- **The convention decides the DEV2 verdict:** F 9.63 (mine) vs 11.58 (frozen). On HP2 the frozen F is 16.35; at my
  cost convention it would be about 16.35 × .834 ≈ 13.6, still a fail.

**(a) Deductive / declared audit.** Every step is deductive or explicitly declared in the READMEs:
- closure c (v4 at the latest free position, the 6 lows at μ(v4));
- K = 1 node for v4; v4 moving with v3; μ linearised along g23;
- the g12 walk dropped in the blind region; the gap beyond tmax assumed correct;
- the correct region as the product of the two 1-D line sets at the transverse point (T302 "point closure of the
  boundary");
- exact walks and model-free mass integrals.

**Undeclared items found** (numerical or accounting choices, not hidden modelling assumptions):
1. **The transverse points are computed by seeded Monte Carlo** (model-free sample means; seen in r1w2 after the
   comparison), while T302's README says "from the exact model-free law". The effect is small: v3* .665 (theirs) vs
   .6661 (exact integration), v4 node .4025 vs .4034. The estimates still agree to .012 pp.
2. **Gauss–Legendre rule and v2 ranges** (number of nodes, cut-offs): not stated in the README. My choice (48 nodes,
   ±6.5) reproduces the estimates.
3. **Walk numerics** (ε past a flip, cell cap): not stated; immaterial here.
4. **FLOP accounting** of the model-free Φ/GL work, and the walk FPE→FLOP unit (5,760 vs 5,948): not stated, and it
   changes C by ~17% (above).

## Claims
- P1 [measured] R1W V1 reproduces from its written description: all 80 DEV2 estimates within .012 pp of the frozen
  code. Walk work agrees within 0.3%.
- P2 [measured] The charged FLOP total differs by 13–20%. The cause is an unwritten cost convention for the model-free
  Φ/GL work (and the FPE unit), not different work. It moves DEV2 point F from 11.58 to 9.63. A confirmation of any
  descendant (T311) should fix the convention in writing first.
- P3 [measured/inspected] No undeclared modelling assumption was found. The undeclared items are numerical (MC
  transverse points, GL rule, walk ε) or accounting choices.

## Compute accounting
Both implementations on 80 models: about 3 CPU-min (mine ~0.1 s per model, the frozen one similar).
