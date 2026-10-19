# estT_cheap (T450): a cheaper exact orthant rule for estT at (8,4) and (8,5)

↑ [TASKS](../../../TASKS.md) · [INDEX T443, T428, H-U5, H-U6](../../../INDEX.md) ·
reference engine: `estT_fix/estT.py` from `claude/t443-estT-fix` @ f02558b4, copied verbatim to `estT.py` (sha256[:12] a88f62c41a94, asserted)

## Question
estT_fix computes each simplicial-cone mass by Plackett's reduction to a 1-D integral in t, using a log substitution
(1 - t = e^-v) with 128 Gauss-Legendre nodes. This holds for both the d = 4 branch (T = 4) and `_orth5` (T = 5).
T443 found that a graded rule t = 1 - (1 - s)^3 with 12-24 nodes is within 5e-6 of it at (4,4) on two models, at about
1/4 of the cost. Does a cheap rule hold on every (8,4) and (8,5) zoo model? If so, what do H-U5's exact/recipe cost
ratios and H-U6's recipe-vs-sampling ratios become?

## Pre-registration (2026-10-19, before any run)
**Design (one engine run per model and mode).** In estT the error is a plain sum of `orthant()` values over the
simplices of the error cells, and no branch depends on a mass value. Each `orthant()` call is therefore wrapped. It
returns the unmodified estT_fix value, so the reference run is the reference engine itself. It also evaluates every
candidate rule on the same simplex, each with its own op counter. For each candidate:
- accuracy = 1 - sum of candidate masses;
- ops = reference ops - reference orthant ops + candidate orthant ops.

Linearity check (gate): on (8,4) seed 0, both modes, one full run with the primary candidate substituted into the
engine must equal the wrapper's prediction to <= 1e-12 in accuracy and exactly in ops. A second gate: the reference
accuracies must equal T192's saved estT_fix values (`gpu_cost_law/results/audit/H8T4_*`) and estT_fix's
`run5/zoo0_exact.json` to <= 1e-12.

**Candidates.** Only the t-rule changes, in both the d = 4 branch and `_orth5`. The cofactor/atan2 numerics and the
thin-simplex branch stay as they are.
- **G24 (primary):** graded Gauss-Legendre, p = 3, 24 nodes: 1 - t = (1 - s)^3, weight 3 (1 - s)^2 w.
- G12, G16: the same with 12 and 16 nodes.
- L24, L32: estT_fix's log substitution with 24 and 32 nodes instead of 128.

**Models.** alg_zoo zoo_2nd_argmax(8, 4, s) and (8, 5, s), s = 0..4 (10 models); modes exact and recipe108 (the
recipe used by H-U5 and H-U6). The reference engine runs unmodified. No noisy copies.

**Reported per candidate:**
- max |delta accuracy| over the 20 runs, and the max per-simplex |delta mass|;
- op ratio (candidate total / reference total), per run;
- re-scored H-U5 (exact ops / recipe ops) and H-U6 (recipe MSE ratio vs sampling).

H-U6's MSE ratio uses H-U6's definition: err^2 / (p(1-p) / N), where N = recipe ops / ops per sample (`zoo44.ops_per_sample`) and
truth = exact. It is computed for the reference (gate: it must reproduce H-U6's seed-2 value of 3.4 within rounding)
and for the candidate with the candidate's own exact truth.

**Decision rule (per candidate, over all 20 runs):**
- **EXACT-GRADE:** max |delta accuracy| <= 1e-8. It can replace estT_fix as the truth engine.
- **ESTIMATE-GRADE:** max |delta accuracy| <= 1e-6, at least 10x below the smallest recipe error in H-U5 and below
  sampling's sd at every budget used. It can be used for recipe op counts, but not as the truth.
- **FAIL:** above 1e-6 on any run.

Adopted rule: the cheapest candidate at the best grade it reaches. H-U5 and H-U6 are re-scored with it, alongside the
reference.

**Predictions:**
- G24 is estimate-grade, not exact-grade: |delta| ~1e-6, as at (4,4).
- L32 is close to exact-grade.
- The exact-run op ratio is ~.25-.35 at (8,4); at (8,5), where the mass share of the ops may differ, it is ~.3-.5.
- H-U5's exact/recipe ratios move by < 20%: both modes use the same orthant code.
- H-U6's seed-2 loss stays a loss (3.4 -> ~1-2).

## Method
_(after the runs)_

## Status
Running (pre-registered).
