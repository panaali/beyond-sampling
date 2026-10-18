# T292: oracle bias floor, observer F and error anatomy on the 80 DEV2 retrains (Fable r2 section D)

↑ [Approaches index](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md)

## Pre-registration (copied verbatim from the DRI brief before any data, 2026-10-18)

Panel: DEV2, 80 open retrains, on branch claude/t192-gpu-cost-law under m16/approaches/gpu_cost_law/hp2/dev2/ (weights.npz, manifest.json). Its 1e8-sample truths (truths.json) are being published there now. If they are still missing when you need them, compute 2e7-sample truths yourself (SE ~5e-5) and say so.
1. **Oracle floor.** Run the E31 lin low-input closure (m16/approaches/stoch_margin/diag_cond.py) on 20k sampled skeletons x 256 low draws per model, on all 80 models. This gives a per-model aggregate bias b_m. It is diagnostic sampling only and is not an estimator.
2. **Observer.** Run the E21 observer (m16/approaches/observer/est.py) at configs (1,1,1), (1,1,2) and (2,2,2) on all 80 models. Record e_m, charged cost C_m (instrument FLOPs honestly) and F per config, plus the per-model bias against accuracy.
3. **ANAT-80.** From the same samples, split each model's error into blind / 2~3 band / 1~2 band / soft tail, by top-3 triple and x9 case. Count the "gross-error triples" (error rate > 20% at gap >= .2). Regress accuracy on these parts. The question is what makes a retrain weak.

Go lines (pre-registered):
- **G1:** RMS_m(b_m) <= .40 pp, |mean b| <= .25 pp, and the slope of b_m against accuracy <= .05 pp/pp. Then build C1 (OBS-cheap: the observer at ~5k FPE, as in Fable r2 C1, with a resolution knob).
- **G2:** any E21 config with point F <= 3 on DEV2. Then freeze it with est(model, budget) -> (estimate, cost_flops) and send it to the T281 scorer (session_01UvVWREPpEYpUGudpzkioKd) for the sealed HP2 curve.
- **Dead:** RMS_m(b_m) > .80 pp. Then the lin/observer closures are dead in the cheap regime (scoped).

Report the bias split by blind / band / tail per model, so that any failure names its part.

### Conventions fixed before data (mine)
- Scoring: F = mean_m(e_m^2) * max(C, 1000) / mean_m(p_m(1-p_m)), C in FPE = FLOPs/5948 (Directive 8 amendment 2).
- "Slope of b_m against accuracy" = OLS slope of b_m (pp) on truth accuracy (pp) over the 80 models; the G1 test
  is |slope| <= .05. Between .40 and .80 pp RMS (or G1 failing only on mean/slope) the outcome is "neither".
- Region of an input (ANAT): blind = 2nd largest <= 0; otherwise with g23 = M2-M3, g12 = M1-M2:
  2~3 band = g23 < .05 and g23 <= g12; 1~2 band = g12 < .05 and g12 < g23; soft tail = min(g12, g23) >= .05.
  Tail is also reported at >= .1 and >= .2. x9 case = position 9 is the max / the 2nd / below (as T269).
  Gross-error triple = ordered top-3 position triple whose non-blind error rate at min gap >= .2 exceeds 20%
  (triples with fewer than 30 such samples are not counted and are reported separately).
- The same region split is applied per skeleton for b_m (skeleton = top-3 values, so region is exact per skeleton).
- Truths: if DEV2 truths.json is absent, 2e7-sample numpy truths from my own ANAT pass (seed 292000 + model index).
