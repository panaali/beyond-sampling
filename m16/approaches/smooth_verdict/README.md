# smooth_verdict: T319, Fable r4 D3: Gaussian-smoothed closure verdict (oracle floor test)

↑ [Approaches index](../README.md) · [TASKS](../../../TASKS.md) · design: claude/project-handoff-priorities-zpzfm8 review/fable_review_r4/README.md D3

## Pre-registration (written 2026-10-18, before any T319 number)
**Question.** A point closure pins the lows at their conditional mean mu and outputs the hard verdict 1[m > 0]. Does
replacing it with Phi(m / (s ||J||)) cut the oracle floor? Here s^2 is the lows' exact conditional variance given the
skeleton, and J is the margin's gradient with respect to the low positions at the closure point, from the exact
gate-pattern map (an adjoint pass).

**Closures** (skeleton exact, lows pinned):
- **R0:** top-3 exact; the 7 lows at mu(v3) = E[X | X < v3], s^2(v3) = Var(X | X < v3) = 1 - v3 lam - lam^2, lam =
  phi(v3)/Phi(v3).
- **top-4 (T298 b):** top-4 exact; the 6 lows at mu(v4), s^2(v4).

**Verdicts at the closure point x_c.** L = logits(x_c), label j = position of the 2nd largest, rival i* = argmax_{i != j} L_i,
m = L_j - L_{i*}, J = (A_j - A_{i*}) restricted to the low positions, A = the exact local logit map of x_c's gate pattern.
- hard: 1[m > 0] (the closure's own verdict, baseline);
- **smooth (primary, Fable's formula):** Phi(m / (s ||J||));
- smooth-all (secondary, labelled): prod over the 9 rivals i of Phi(m_i / (s ||J_i||)) (independence across rivals).

**Oracle protocol (diagnostic sampling, paired, as T292/T298).** Per DEV2 model, 250k inputs x ~ N(0, I_10) (seed
319000 + index), ok(x) = the model is correct on x. Per-model floor b_m = mean(verdict(closure of x) - ok(x)), in pp.
All 80 DEV2 models, 1e8 truths not needed (paired). Report RMS_m, mean, slope on accuracy (pp/pp, truths from DEV2
truths.json), region split (blind / 2~3 / 1~2 / tail, as T292) and per-model SE.

**Go / stop (DRI line):** per closure, **go** iff RMS_m(smooth) <= 0.70 x RMS_m(hard) AND |slope(smooth)| <= |slope(hard)|
+ 0.01. Otherwise **stop** for that closure. Overall go if either closure goes. I also report Fable's own secondary
line (mean offset halved; stop < 10% cut). Labels: the closure is a declared structural approximation; the Phi
smoothing and J are deductive (exact local map, model-free s); the oracle is diagnostic sampling.
Cap 1 h.
- **Amendment (before any panel data):** 100k inputs per model, not 250k, to fit the 1 h cap (~45 s/model at 250k). Per-model SE is ~.07 pp, and noise-corrected RMS is also reported.

## Method
`sv.py`: per sampled input, the closure point (skeleton exact, lows at their conditional mean). One forward-mode pass
gives the logits and the exact local map A (checked: L = A x_c to 6e-13; no biases). From these: the hard verdict,
Phi(m / (s ||J||)) for the binding rival, and the product over rivals. Paired against ok(x).
`summarize.py` -> `summary.json`. Raw per-model numbers: `results.json`. Run: `python sv.py && python summarize.py`
(~16 CPU-min, 80 models x 1e5 inputs).

## Status
**Done: STOP** for both closures (pre-registered). The overall result is no go.

## Results (80 DEV2 retrains; pp; slope = pp of bias per pp of accuracy)
| closure / verdict | RMS (noise-corr.) | mean | sd | slope | tail part mean / RMS | 2~3 band mean / RMS |
|---|---|---|---|---|---|---|
| R0 hard (T298: .515) | .527 (.521) | +.216 | .48 | -.041 | +.05 / .32 | +.13 / .20 |
| **R0 smooth** Phi(m/(s\|\|J\|\|)) | **1.326** | **-.987** | .89 | **+.119** | -.81 / 1.12 | -.11 / .18 |
| R0 smooth, product over rivals | 11.69 | -8.85 | 7.7 | +.61 | -6.80 / 9.11 | -1.32 / 1.70 |
| top-4 hard (T298: .389) | .399 (.395) | +.291 | .28 | -.045 | +.14 / .22 | +.08 / .12 |
| **top-4 smooth** | **.657** | **-.454** | .48 | **+.065** | -.37 / .56 | -.08 / .11 |
| top-4 smooth, product over rivals | 5.88 | -4.38 | 3.9 | +.36 | -3.25 / 4.47 | -.75 / .96 |

The RMS ratio smooth/hard is 2.52 (R0) and 1.65 (top-4); the go line was <= .70. The slope grows from |.04| to .12 / .07.
On Fable's secondary line, the mean is not halved: it flips sign and doubles. Both closures meet "stop < 10% cut".

## Claims
1. [measured] Replacing the closure's hard verdict by Phi(m / (s ||J||)) makes the oracle floor **worse**, 1.6-2.5x in RMS,
   with a steeper slope on accuracy. The positive offset (+.2 to +.3) becomes a larger negative one (-.45 to -.99).
2. [measured] The damage sits in the soft tail (R0: tail part -.81 mean / 1.12 RMS, against +.05 / .32 hard). The bands
   change little and the blind region not at all.
3. [inferred] It is the same mechanism as the T292 lin failure. ||J|| is the gradient on the closure point's single
   gate pattern, and extrapolated over the lows' full spread it overstates how much the lows move the margin: the
   lows switch gates, and the switches mostly damp their effect, which the first-order law ignores. So the first-order
   band is too wide, and it invents errors where the true error rate is tiny. Treating the rivals independently
   (product) compounds this 9-fold.

## Alternatives not ruled out
- A shrunk smoothing scale (Phi(m / (c s ||J||)) with c < 1) would interpolate between hard (c -> 0) and this. But c would be
  fitted (inductive advice) unless derived. Not tried, by design.
- J restricted to low positions whose perturbation keeps the gate pattern (cone-clipped, as in E21 v4), or an exact
  1-D treatment of the single largest-|J| low. Untried. E21's cone clipping fixed the analogous linear-noise failure.
- 100k inputs per model (SE .05-.07 pp). The ratios are far from the line, so this has no effect on the verdict.

## Failure modes
- Any first-order (Gaussian) treatment of the lows through one local map. On retrains the lows routinely switch gates
  (only 4-12% of low draws keep the anchor's pattern, T292), so the first-order spread is wrong by a large factor.

## Compute accounting
~16 CPU-min of diagnostic sampling (8e6 forward passes + 1.6e7 closure passes with forward-mode maps). For an
estimator, the smoothing itself would cost ~1 extra map per closure point (~10 FPE).
