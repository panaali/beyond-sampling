# T274a: probe v5 and a weight-derived spread variant (Directive 8, lenient bar)

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T269 gap formula](../gap_formula/README.md)

## Question
Can a probe-based estimator (constructed inputs put through the network, with exact model-free extrapolation) reach
the LENIENT bar on every LB3 group at one stated budget? The bar: MSE ratio <= 10 against plain sampling at equal
FLOPs, judged by the cluster-bootstrap CI. If yes, how much of it is deductive?

## Pre-registration (2026-10-18, written before the LB3 panel run)
**What was looked at before freezing (disclosed):**
- `diag.py` — probe v2 vs the true error decomposition on orig, seed0, s0.005_1, s0.02_0;
- `nodes_test.py` — bias of canonical vs real lower ranks on the same four models;
- `bias_scan.py` — v5a bias at 131k nodes on all 30 TEST models;
- `variant_test.py` — four spread variants on six TEST models (orig, seed1, seed4, s0.005_1, s0.01_1, s0.02_4).

The variant choice below was then made on the DEV models only, with a criterion fixed beforehand (`dev_select.py`).
The test-model looks are a contamination risk. They are listed so the reader can discount them.

**Frozen methods** (code `v5.py` at the commit that adds this README; no constants change after this point):
- **probe_v5a = `v5.est`.** The node design, per node:
  - the position triple cycles through all 720 (balanced);
  - the top-3 values come from a fixed scrambled Sobol point (seed 274) through the exact Rényi conditional quantiles
    of the order statistics of 10 i.i.d. N(0,1);
  - ranks 4–10 are canonical values below M3, Q(Φ(M3)·(8−j)/8), on the remaining positions in a fixed permutation;
  - the estimate is the fraction of nodes on which the network returns the rank-2 position.
  - Cost: 1 FPE per node; N = budget. No tunable constant.
- **probe_v5b = `v5.est_spread(rivals=2, n_low=3)`.** The same nodes, but each node contributes
  Π_k Φ(d_k/s_k) over the 2 strongest rival logits. Here d_k is the exact logit margin at the node, and s_k is its s.d.
  when ranks 4–6 vary by their model-free covariance given M3 (delta method on uniform order statistics). The s.d.
  goes through the exact local Jacobian (reverse pass inside the node's gate pattern).
  - Cost: (1 forward + 2 reverse) = 3 FPE per node, charged ×1.01 for the margin algebra.
  - Chosen on DEV: RMS bias .0055 vs v5a .0117 (`results/dev_select.json`).

**Scoring.** LB3 code unchanged (`m16/leaderboard.py` at origin/ccr-6a61f25c-wn80ii d8d6f51f):
- all CURVE budgets (1k, 4k, 16k, 64k, 256k, 1M) on all 30 test models;
- truths from `results/leaderboard_truths.json` (32M samples each);
- 2000-resample cluster-bootstrap CIs;
- the sampling baseline p(1−p)/C. FPE = 6000 FLOPs (T196, multiply-add = 2).

Both methods are deterministic, so each (model, budget) runs once. They are registered locally as class
"quadrature" for the run (`run_lb3.py` injects that in memory; no shared file is edited).

**Go line (pre-registered):**
- PASS for a method if, **at budget 4,000 FPE**, the upper 95% CI of the ratio is <= 10 in all six groups.
- The 4k budget was chosen after the bias scan: at 16k the v5a biases of 1–1.7 pp would exceed the lenient bound in
  the seed and s.02 groups.
- Every other budget is reported with its CI. Passing at other budgets is reported but is not the claim.

**Expected (written before the run).** v5a: point ratios about 2–5 at 4k in every group; fails at 16k in the seeds and
s.02. v5b: better on the noisy groups; worse on orig and seeds (on test models it over-corrected orig by −1.0 pp
and seed4 by −2.1 pp).

**Labels (Hilton, `lit/web_log.md`).**

| step | label |
|---|---|
| exact law of positions and top-3 values (Rényi transform) | deductive (math, model-free) |
| the network's answer on a constructed input | deductive (exact computation) |
| ranks 4–10 at canonical values | approximation |
| averaging an indicator over a fixed node set | **inductive** (project inference: fixed-node averaging) |
| v5b spread | deductive extrapolation from the weights (exact local Jacobian), with a linear-Gaussian approximation |

So a pass is lenient-bar evidence **with the inductive caveat**. v5a is in effect a 3-D QMC design with a canonical
7-D remainder.

**Alternatives (before the run).**
1. A pass reflects the low budget, not quality: at 4k, sampling's own s.e. is 0.3–0.8 pp, so a ~1 pp bias is tolerated.
2. Any deterministic node average is QMC by another name, so the "probe" label adds nothing. Separate it by comparing
   v5a's bias with full 10-D exact-law QMC, which should be unbiased.
3. v5b's gain on DEV comes from noisy copies only; the seeds have no DEV analogue. Expect it to transfer badly to
   the seed group.

## Addendum pre-registration (after the v5a/v5b panel, before any v5q run)
v5a and v5b both FAIL the go line (Results). To test alternative 2, a third frozen method is added as a **control**:
- **probe_v5q = `v5.est_full`**: the same triple-balanced design, but all 10 order statistics come from a 10-D
  scrambled Sobol point (seed 274) by the exact Rényi transform. No canonical values.
- It is fixed-node QMC: inductive in every step except the exact law. It is a lenient-bar control, not a probe method.
- Expected: bias ~0 and ratio <= ~1 at every budget. Go line as before (4k, upper CI <= 10 in all six groups).

## Method
*(after the run)*

## Status
Pre-registered; panel not yet run.
