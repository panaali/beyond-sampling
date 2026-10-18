# blind11 — T293 BLIND-11: a deductive estimate of the blind region's error contribution (Fable r2 C3)

↑ [TASKS](../../../TASKS.md) · Fable r2: branch claude/project-handoff-priorities-zpzfm8, review/fable_review_r2/README.md C3

**All claims unverified.**

## Pre-registration (written 2026-10-18, before any T293 number was computed)
**Target.** Blind region B = {x_(2) <= 0}, P(B) = 11/1024 exactly, error contribution c = P(B) P(err | B). Cases: q = none (all
negative, mass 1/1024) or exactly one positive at position q (mass 1/1024 each). Within a case the label is the position j of the
largest negative input, uniform over the n = 10 (case none) or 9 negatives by exchangeability.

**Method (deductive; no witness verdict, the T287 trap).** For each (case, j):
1. Gate pattern: the trained net's gates on the case's canonical input. That input uses the model-free conditional means: x_q =
   E[|Z|], x_j = E[max of the n negatives], others = E[a negative below the max], placed at their positions. Only the *gates* are
   used, never the verdict.
2. With the gates frozen, logits are linear: L = A x (A from the frozen recurrence; deductive linear algebra on the weights).
3. Margins m_i = (A_j - A_i) . x for i != j. Their mean and covariance come from model-free conditional moment tables of x
   given (case, j). By exchangeability these are a mean and variance for x_q, for x_j, and for the "other negatives" (common
   variance and covariance), plus cov(x_j, other).
4. P(correct | case, j) = P(all m_i >= 0) under a Gaussian closure of the margins, evaluated deterministically by Clark's (1961)
   moment matching of the max of correlated Gaussians (P(max_i(-m_i) <= 0)).
5. P(err | B) = 1 - mean over (case, j) with weights 1/11 x 1/n; contribution c = (11/1024) P(err | B).
Labels (Hilton): tables model-free; gates from one constructed input (allowed); the affine map and closure are deductive, but the
gate freezing is an approximation, not an extrapolated verdict. Cost: instrumented FLOPs (forward passes, A construction,
moment algebra, Clark), reported in FPE (6000 FLOPs). The target is ~100 FPE; the actual figure is reported.
Baseline: a fixed 8/9 error rate (FINDINGS' argued value).

**Test.** DEV2, 80 open retrains (T192 611796bd hp2/dev2/weights.npz). Sampled reference per model: 2e4 inputs per case drawn
exactly from the case's distribution (2.2e5 per model; SE of c ~1e-3 pp), diagnostic only. Also the 30 LB3 models.
**Go:** RMS over the 80 DEV2 models of (c_pred - c_sampled) <= 0.15 pp. **Stop (dead, scoped to this closure):** RMS > 0.25 pp.
Between the two: inconclusive. If it passes, publish est_blind(model) -> (contribution, cost_flops).

**Amendment (before any number):** in case "none" (all inputs negative) the label is the *second*-largest negative, so that case
is split by the ordered pair (a = largest, j = second largest), 90 pairs of weight 1/90 each, with role moments for "top", "label",
"other". The total is 180 (case, label) units, so the cost estimate is ~1-1.5k FPE, above the ~100 FPE target, and is reported as measured.

## Results (unverified; `blind.py` -> blind.json, blind.log)
Clark vs exact MVN check on a random 9-D case: 0.054 vs 0.047. Cost: **2,759 FPE** per model, dominated by the frozen-map
products (H^3 per step x 180 units). That is 28x the ~100 FPE target; it could be cut ~10x by sharing gate patterns across units.

| panel | RMS (pred - sampled), pp | mean (pred - sampled) | fixed 8/9 RMS | pred sd across models | sampled sd | corr(pred, sampled) |
|---|---|---|---|---|---|---|
| DEV2 (80 retrains) | **0.234** | +0.162 | 0.265 | 0.064 | 0.139 | **-0.30** |
| LB3 (30 models) | 0.120 | +0.083 | 0.123 | 0.033 | 0.078 | -0.07 |

## Verdict (pre-registered rule): **inconclusive by the letter (0.15 < 0.234 <= 0.25), uninformative in substance**
- The closure's prediction hardly varies across models (sd 0.06 pp vs the true 0.14 pp) and is *anti*-correlated with the truth
  on DEV2 (-0.30). It is a slightly worse constant than "8/9" plus noise: even after removing its mean bias by an oracle shift, the
  RMS is 0.169 pp, above the go line.
- Mechanism [inferred]: with gates frozen at one canonical input per (case, label), the closure sees essentially the same affine
  map for every model of the panel's architecture. The model-to-model variation in blind-region error (0.38-0.91 pp on DEV2) comes
  from gate switching *inside* the case, which the frozen map does not see (T218: blind-region prefix regions still grow 12-19x per
  step).
- **Not published as a component** (the go line was not met). The sampled side (diagnostic) shows what is needed: a per-model blind
  rate good to ~0.15 pp, i.e. relative accuracy ~20% of a 0.73 pp contribution.

## Alternatives not ruled out
- Gates per (case, label) *and* per sign-pattern-equivalent sub-cases (e.g. split the label's value into quantile bins: T282 showed
  conditioning halves frozen-gate bias).
- Second-order closure (account for gate switching probability per neuron and step by a Gaussian approximation of each
  pre-activation, i.e. a mean-field ReLU, rather than frozen gates).
- Exact region treatment inside a case is not small (T218), so it was not attempted.

## Compute accounting
110 models x (180 units, ~2.8k FPE) + diagnostic sampling 2.2e5 per model; ~15 CPU-min.
