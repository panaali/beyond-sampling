# T332: paradigm-level information ceiling from T317e's own per-row outputs (Fable r5 settling test)

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T314](../info_ceiling2/README.md)

## Question
Fable r5 §3 (`review/fable_review_r5/README.md` on claude/project-handoff-priorities-zpzfm8) asks whether any function
of what the R1 geometry measures can reach the strict bar. The test: take T317e's deductive estimate as a fixed
offset, and re-weight T317e's own per-row outputs with a fitted ridge. If the held-out RMS reaches ≤ .50 pp, the
information is present (F .79 at the 2.2k-FPE oracle cost), and a deductive mechanism behind the re-weighting is
worth looking for. If it stays ≥ .62 pp (F 1.22 at 2.2k), no function of these measurements reaches strict even at
the oracle cost; together with T326, strict is then closed for this paradigm (scoped to the measurement class).
**Diagnostic, and inductive by construction:** a fitted predictor is not an estimator ARC would accept.

## Pre-registration (2026-10-18, before any feature is computed)
**Code.**
- `cont_e.py` is a verbatim copy of T317e (claude/t218-tie-tail-labels @01906d33, sha256 `50dd1c9b…fda2b`).
- `features.py` reruns `cont_e.est`'s loop with read-only hooks and asserts its total equals `cont_e.est(model)[0]`
  bitwise. No per-row hook was needed from T218.
- The tie-margin readout recomputes `first_cell` on each line start: an extra prefix-tree pass, which already
  belongs to the walk's own cost.

**Per-row outputs.** Rows are the 720 position triples (p1, p2, p3), each with weight 1/720. Line types:
- 23nb: the g23 line, nonblind region;
- 23b: the g23 line, blind region;
- 12nb: the g12 line, nonblind region (the blind g12 is model-free and has no per-row output).

For each row and line type:
- **deficit** d = (full − actual) contribution to the estimate. For 23nb/23b, full is the integral with I23 at its
  all-correct value (Φ^8 profile) and the row's actual I12. For 12nb, full uses I12 all-correct (1 − Φ) and the row's
  actual I23. Each deficit is integrated against the region's density and Gauss-Legendre weights, exactly as in `est`.
- **tie margin** m = L_k − max_{j≠k} L_j at the line start (the tie point, t = EPS), with k = p2.
- **slope** s = dL_k − dL_{j*} there, with j* the rival at the tie.
- **cells** c = the number of walk cells the row used (≤ 400).

**The 30 features** (per model):

| group | features | count |
|---|---|---|
| deficit by line type × age of p2 | age(p2) = 9 − p2 in bins {0–1}, {2–4}, {5–9}; mean over rows of d (bin rows contribute, others 0) | 9 |
| deficit by line type × age of p3 | the same bins for p3 | 9 |
| tie margin, per line type | mean clip(m, −3, 3); fraction m < 0 | 6 |
| slope, per line type | mean clip(s, −30, 30) | 3 |
| cells, per line type | mean c | 3 |

**Learner (primary, decides the verdict).**
- Target: y = truth − est_T317e, in pp.
- Ridge with intercept on standardized features. The intercept is the fitted offset.
- Alpha from logspace(−3, 9, 13) by inner `KFold(5, shuffle=True, random_state=0)` on the 40 training models only.

**Data (T314 protocol).**
- Train: DEV2 fused_s0..39 (truths 1e8).
- Test: torch_s0..39 (truths 1e8) + LB3 seeds 0–4 (truths 32M), 45 models.

**Reported, not deciding:**
- the offset alone (train-mean residual), which Fable expects to give ~.69;
- raw T317e;
- the torch and seeds parts separately;
- a depth-2 GBT on the same 30 features (T314's settings: lr .05, n_est in {50, 100, 200, 400}, subsample .8, rs 0).

**Verdict (DRI / Fable).** Primary held-out RMS over the 45 test models:
- **≤ .50 pp → GO** (the information exists);
- **≥ .62 pp → STOP** (strict is closed for the paradigm, scoped);
- in between: neither.

**Alternatives first.**
1. With 40 training models, 30 features may overfit. Ridge's nested alpha guards against that, and the offset-only
   row is the floor to beat.
2. A fused → torch kernel shift may cap the held-out RMS regardless of the information present. The seeds row and
   the inner-CV RMS separate the two.
3. A GO could come from a feature that proxies accuracy itself (the bias is concentrated in low-accuracy models,
   T281 subset F). A GO would therefore be followed by an ablation by feature group before any mechanism is claimed.

## Status
**Done. Verdict: NEITHER** (held-out RMS .577 pp, between the .50 go line and the .62 stop line). The
pre-registration is commit f6d36aee. Unverified; diagnostic and inductive by construction.

## Results
To reproduce:
- `PYTHONPATH=<alg_zoo shim> python3 features.py` takes 85 models, ~2 CPU-min, and asserts every per-row total equals
  `cont_e.est` bitwise.
- `python3 learn.py` writes `results/learn.json`.
- `python3 ci.py` writes `results/ci.json`. It gives a 10,000-resample bootstrap over the 45 test models, and F with
  the test mean p(1−p) = .0724.

| predictor | inner-CV RMS (train 40) | **held-out RMS (45)** [95% CI] | torch (40) | seeds (5) | train | held-out mean err | F at 2.2k | F at 5.4k |
|---|---|---|---|---|---|---|---|---|
| raw T317e | — | .817 [.67, .95] | .821 | .786 | .767 | +.53 | 2.03 | 5.01 |
| offset only | .688 | .646 [.55, .74] | .654 | .580 | .673 | +.16 | 1.27 | 3.13 |
| **ridge, 30 features (primary; alpha 10)** | .618 | **.577 [.46, .68]** | .589 | .464 | .430 | +.14 | **1.01** | 2.49 |
| GBT, 30 features (secondary; 50 trees) | .672 | .611 | .627 | .458 | .266 | +.18 | 1.17 | 2.88 |

- **Re-weighting T317e's per-row outputs gains .07 pp over a fitted offset** (.646 → .577).
  - At the oracle cost of 2.2k FPE that is F ≈ 1.0, exactly at the strict line. At T317e's actual 5.4k it is F ≈ 2.5.
  - The CI [.46, .68] spans both the go and stop lines, so 45 test models cannot separate them.
- The offset alone gives .646 held-out and .688 in inner CV, close to Fable's ~.69.
  - A held-out mean error of +.14 to +.16 pp remains after fitting the offset on fused models. The bias moves between
    kernels (fused → torch), which is alternative 2.
- The largest standardized ridge coefficients (pp per sd) are:
  - slope_23nb +.25;
  - def_12nb by p3 age: 5–9 −.22 and 2–4 +.20;
  - def_23nb p2age0–1 +.17;
  - def_23nb p3age2–4 +.13.

  The opposite signs within one deficit family suggest the fit trades bins against each other rather than reading
  one mechanism. No ablation was run, because the result is not a GO (pre-registered).

## Compute accounting
- Features: 85 × (T317e's run + one extra `first_cell` pass per line for the tie readout), ~2 CPU-min, single
  thread, `ulimit -v 2500000`.
- Learners and bootstrap: seconds.
- Feature cost per model: T317e's own ~5.4k FPE plus the tie readout. The readout repeats the first cell's prefix
  pass already counted in the walk, about ≤ 1.2k FPE if charged separately (not measured).

## Claims
- [measured] On 45 held-out models, a nested-CV ridge on 30 re-weightings of T317e's per-row outputs, with T317e's
  estimate as offset, reaches RMS .577 pp [.46, .68]. The pre-registered verdict is **neither** (.50 < .577 < .62).
- [measured] A fitted offset alone reaches .646 pp. Raw T317e is at .817.
- [inferred] Even fitted, the information in this geometry's outputs puts F at about 1.0 at the 2.2k-FPE oracle
  cost and about 2.5 at the realised cost. Strict is not reached at any realised cost of the paradigm. The test
  cannot close strict at the oracle cost either, because the CI is too wide.
- [speculative] A deductive mechanism behind this re-weighting would gain at most ~.07 pp over a de-biasing offset.
  That is small next to the .38 pp spread between the training and held-out residual means and the closure floor
  (.47).

## Alternatives not ruled out
- **Training set size.** 40 training models may be too few for 30 features (train .43 against held-out .58). More
  fused retrains would sharpen the ceiling. HP2/HP3 are sealed and not usable for fitting.
- **Feature choice.** Other per-row aggregations could carry more, such as finer age bins, per-(p2, p3) deficits, or
  margins scaled by slope. The pre-registered 30 are one family.
- **Kernel shift.** This is the fused → torch shift. The seeds-only RMS (.46, 5 models) is below the go line, but
  n = 5.
