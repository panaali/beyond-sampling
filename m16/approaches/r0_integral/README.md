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
- Margin convention: tau = 0 if R0's margin is already positive at the tie (gap 0+). Otherwise the Newton root is
  clipped to [0, 3]; a non-increasing margin gives tau = 3, i.e. wrong across the band.
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

*Amendment before any TB score (cost check):* the pre-registered design (2 anchors for tau23, 1 for tau12, 1 blind
verdict) costs 7 FPE per triple, about 5.4k FPE, which is over budget. A cost ladder is therefore run: (n23 anchors,
Newton steps, blind verdicts) = (1,1,1) at ~3.9k FPE, the only variant within 4k, up to (4,3,2). The go line is
unchanged.

## Method (as run)
- `r0.py`: DEV2 loader (array sha256 as T292), R0, exact skeleton law, logits with exact forward-mode tangent.
- `mc.py` (`run_mc.sh`): reference acc_R0 from 1e7 skeletons per model (s.e. 0.01 pp), with R0's errors by region.
  `results/mc.json`.
- `tb.py`: the integrator. Model-free check: the nonblind mass integral reproduces 1 - 11/1024 to 3e-15.
- `tb_eval.py`: scoring, `results/tb_eval.json`.
- `decomp.py`: TB's assumed-correct indicator vs R0's verdict on 1e6 sampled skeletons, by region.
  `results/decomp.json`.

## Status
Done (2026-10-18). **DEAD for threshold designs (pre-registered):** no variant reaches .25 pp, at any cost up to
23k FPE.

## Results
**R0 reference** (1e7 MC; 10 DEV2 models):
- acc_R0 0.883-0.954.
- R0's errors: blind 0.59-0.90%, band23 (g23 < .05) 2.2-4.3%, band12 (g12 < .05) 0.6-2.4%, other 0.6-4.8%.
- Target acc_R0 0.9534 (truth 0.9531).

**TB integration error vs MC of R0** (RMS over the 10 DEV2 models):

| variant (n23, Newton, blind) | charged FPE | integration RMS (pp) | mean (pp) | vs truth RMS (pp) | target integration error (pp) |
|---|---|---|---|---|---|
| (1,1,1) | **3,912** | **2.75** | -2.20 | 2.72 | -2.36 |
| (1,2,1) | 6,792 | 1.27 | -0.78 | 1.33 | +0.19 |
| (2,1,1) | 5,352 | 2.70 | -2.09 | 2.72 | -2.53 |
| (2,2,1) | 9,672 | 1.19 | -0.60 | 1.30 | +0.14 |
| (2,2,2) | 10,392 | 1.20 | -0.61 | 1.31 | +0.14 |
| (3,2,2) | 13,272 | **1.16** | -0.56 | 1.27 | +0.22 |
| (4,3,2) | 23,352 | 2.96 | -2.07 | 2.94 | -2.07 |

- **At <= 4k FPE the integration error is 2.75 pp RMS, 11x the go line.**
- Spending more does not converge. The error plateaus at ~1.2 pp, and a third Newton step makes it *worse*: past a
  gate flip or competitor switch the frozen-gate Newton step overshoots.

**Why: R0's error set is not a threshold set** (`decomp.py`; mass in pp, mean over the 10 models):

| region | (1,1,1): TB correct & R0 wrong | TB wrong & R0 correct | (2,2,2): TB correct & R0 wrong | TB wrong & R0 correct |
|---|---|---|---|---|
| blind | 0.10 | 0.12 | 0.15 | 0.10 |
| band23 | 0.99 | 1.31 | 1.01 | 0.98 |
| band12 | 0.53 | 0.51 | 0.57 | 0.43 |
| other | 1.06 | 2.96 | 1.04 | 1.81 |

- TB and R0 disagree on about 6-7% of the skeleton mass. The net error of 1.2-2.7 pp is what remains after
  cancellation.
- The disagreement sits in every region. Inside the gap bands (1 pp each way), a per-triple threshold that depends
  on v2 alone misplaces the boundary, which also moves with v3 and g12.
- In the bulk ("other", 1.0 pp), R0 is wrong where no tie threshold predicts it.
- **What a passing design would need [inferred]:** the R0 boundary resolved in all three value dimensions per
  triple. That means boundary-finding along at least two directions at several (v3, g12) points, plus the bulk
  error cells. Even a few anchors per dimension is >= 30-50 evaluations per triple, i.e. >= 20-40k FPE, 5-10x over
  budget, before counting gate-flip corrections.

## Compute accounting
- Charged (estimator): the table above, FLOPs as implemented (value + tangent = 2 forwards; GL nodes and Phi
  evaluations charged; GL is ~290 FPE of each total).
- Diagnostic: MC 11 x 1e7 R0 forwards plus decomp 2 x 10 x 1e6. About 15 CPU-min in all.

## Claims (unverified)
- [measured] The deterministic threshold-boundary integral of R0 has integration error 2.75 pp RMS at 3.9k FPE
  and >= 1.16 pp at any cost up to 23k FPE (10 DEV2 models). **Dead for threshold designs.**
- [measured] R0's misclassified set disagrees with every per-triple threshold model tried on ~6% of the mass, in
  all regions including the bulk.
- [inferred] A deterministic R0 integral within 4k FPE (5.6 evaluations per triple for a 3-D piecewise-constant
  function) is not plausible with boundary-finding. The .51 pp oracle floor is not reachable cheaply this way.

## Alternatives not ruled out
- Sharing boundary information across triples (by symmetry or a weight-derived position map) could cut the cost
  per triple. For example, the 2nd-vs-3rd threshold may depend mainly on (p2, p3) and age. Not tried.
- An exact per-triple 3-D decomposition into R0's linear pieces (R0 is piecewise linear in (v1, v2, v3) except
  through mu(v3)) would be exact, but would cost per piece.
