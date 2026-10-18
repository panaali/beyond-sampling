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
Files:
- `probes.py` + `diag.py` — v2 failure diagnosis;
- `nodes_test.py` — canonical vs real lower ranks;
- `v5.py` — the three frozen estimators;
- `bias_scan*.py`, `variant_test.py`, `dev_select.py` — pre-freeze looks, all disclosed above;
- `run_lb3.py` — LB3 runner.

LB3 is run unchanged from a scratch checkout of origin/ccr-6a61f25c-wn80ii d8d6f51f. A local `alg_zoo` shim loads the
same public ARC checkpoints from disk (the package can't be installed here); it reproduces the committed truths
(orig .95296 vs .95314 on 200k). Raw rows: `results/lb3_raw.jsonl`; tables: `logs/lb3_run_v5ab.log`, `logs/lb3_run_v5q.log`.

**Why probe v2/v4 fail on noisy copies** (`diag.py`, 1M samples per model):
- On s0.005_1, 107 of 720 triples still pick the max at the largest probed 1-vs-2 gap. v2 then charges them
  P(rel gap < .95) ≈ 1, which over-predicts "picks the max" errors by 11 pp (18.4% predicted vs 7.5% true).
- v2 has no mode for "picks rank >= 4", which is 2.4% (s0.005_1) and 11.3% (s0.02_0) of all inputs.
- The blind error rate is fixed at 8/9, while the true rates are .55–.87.

v5 removes all three by reading the network's answer on exact-law nodes. `nodes_test.py` shows a construction bug
in the first attempt: canonical lower values must sit below the real M3, or the ranks break (+6.7 pp).

## Status
Done. **Go line: v5a FAIL (narrowly), v5b FAIL, v5q (QMC control) PASS.** Unverified.

## Results (LB3, all budgets; ratio [95% cluster-bootstrap CI], bias in pp)
Reproduce: `PYTHONPATH=<shim> python3 run_lb3.py <lb3 checkout> probe_v5a,probe_v5b` and `... probe_v5q`
(about 25 CPU-min in total).

| method | budget | orig | seed | s0.0025 | s0.005 | s0.01 | s0.02 |
|---|---|---|---|---|---|---|---|
| v5a | 1k | 0.03 | 1.76 [.15, 4.13] | 0.77 [.26, 1.48] | 2.37 [.30, 5.13] | 2.65 [.18, 6.20] | 4.61 [1.48, **10.16**] |
| v5a | **4k** | 0.31 | 6.63 [.96, **12.55**] | 0.12 [.06, .21] | 0.36 [.14, .65] | 0.44 [.10, 1.01] | 1.48 [.94, 2.04] |
| v5a | 16k | 0.01 | 19.9 [8.5, 31.5] | 0.90 | 4.97 | 6.17 | 11.8 [5.8, 18.6] |
| v5a | 64k–1M | 0.5–9 | 45–710 | 6–100+ | 18–337 | 35–618 | 45–778 |
| v5b | 1k | 2.55 | 1.83 [.71, 2.84] | 11.1 [4.7, 20.5] | 33 [3, 74] | 38 [8, 83] | 43 [11, 90] |
| v5b | **4k** | 7.87 | 8.86 [3.75, **14.44**] | 8.19 [6.07, **10.56**] | 6.65 [2.41, **11.08**] | 6.47 [3.40, **10.33**] | 8.78 [4.80, **14.35**] |
| v5b | 16k+ | 50–2866 | 56–3919 | 37–1903 | 20–870 | 11–300 | 8–134 |
| v5q | 1k | 1.05 | 0.14 [.03, .24] | 0.43 [.07, .88] | 2.46 [.12, 5.42] | 3.64 [.14, 9.02] | 5.03 [2.84, 8.40] |
| v5q | **4k** | 0.50 | 0.71 [.27, 1.07] | 0.24 [.05, .42] | 1.05 [.28, 2.09] | 0.66 [.11, 1.20] | 1.33 [.53, 2.20] |
| v5q | 16k | 1.16 | 1.76 [.67, 3.35] | 1.11 | 1.20 | 0.32 | 0.32 |
| v5q | 64k | 0.17 | 1.69 [.33, 4.03] | 0.95 | 3.16 [.75, 6.21] | 4.13 [1.61, 7.49] | 3.05 [.89, 6.19] |
| v5q | 256k | 0.74 | 7.3 [2.2, 13.9] | 6.0 [.4, 16.1] | 12.1 | 15.7 | 8.1 |
| v5q | 1M | 7.2 | 31 | 18 | 30 | 46 | 25 |

Biases at 4k:
- v5a: orig +.19, seed +.93, s0.0025 +.01, s0.005 +.16, s0.01 +.26, s0.02 +.56 pp;
- v5b: −.93, −1.21, −1.08, −1.06, −1.46, −2.06 pp;
- v5q: +.24, +.13, −.02, −.28, −.37, −.66 pp.

At large budgets the biases stay flat:
- v5a: seeds +.7, s0.02 +1.3 pp;
- v5b: −0.5 to −1.7 pp;
- v5q: at 256k–1M a residual 0.05–0.13 pp, plausibly from assigning triples by node index mod 720, which correlates
  with Sobol's low bits (not tested).

So none has a useful convergence rate (alpha ≈ 0 for v5a/v5b), consistent with T271's verdict that the probe
family is closed.

**Verdict against the pre-registration (4k, upper CI <= 10 in all six groups):**
- **v5a fails**, only in the seed group (point 6.63, upper 12.55, driven by seed1's +1.2 pp bias). Every other group
  has upper CI <= 2.04. At 1k it misses by 0.16 (s0.02 upper 10.16).
- **v5b fails** in five groups. The DEV-selected spread over-corrects everywhere on the test models (−0.9 to −2.1 pp).
  It halved the bias on DEV (RMS .55 vs 1.17 pp) but not on test: alternative 3 (DEV is noisy copies only) holds,
  and it is worse on the noisy test groups too.
- **v5q passes** at 4k (all upper CIs <= 2.20), and also at 1k (<= 9.02), 16k (<= 3.35) and 64k (<= 7.49). It
  fails at >= 256k.

**Halving v4's bias?** On orig, v4 is +0.25 pp at ~167k evals. v5a is +0.02 pp at 131k nodes (`bias_scan`) and
−0.11 / +0.19 / −0.02 / +0.06 pp at 1k / 4k / 16k / 64k, i.e. ≥ halved at ≤ 1/2.5 of the cost. On the noisy copies
v4 is off by up to −9 pp, while v5a stays within +1.7 pp. But v5a's remaining bias in the seed and s0.02 groups
(1–1.7 pp) is still too large above 4k.

## Amended rule (DRI, Directive 8 amendment, ccr 4df09663; applied after the panel, no rerun needed)
A pass is now judged at one budget B in {16k, 64k, 256k} FPE, with charged cost >= B/2 and upper CI <= 10
(lenient) or <= 1 (strict) in all six groups. The 4k go line above is superseded. From the same panel:
- **v5a FAILS** at every allowed B: at 16k the upper CIs are seed 31.5 and s0.02 18.6.
- **v5b FAILS** everywhere.
- **v5q PASSES the lenient bar at B = 16k** (upper CIs: orig 1.16, seed 3.35, s0.0025 1.72, s0.005 2.27, s0.01
  0.68, s0.02 0.55) **and at B = 64k** (max upper 7.49, s0.01). It fails at 256k (upper 13.9–32.9). It does not reach
  the strict bar at any B: the uppers exceed 1 in 4 of 6 groups at 16k.
- v5q is inductive QMC, so this is not a mechanistic pass.

**Alpha** (slope of log RMS bias over the six budgets, per group; `results/alpha.txt`):
- v5a: +0.02 to +0.08 (bias-limited, like v4);
- v5b: −0.06 to +0.39;
- v5q: +0.15 to +0.41, with an RMS bias floor of 0.15–0.3 pp from 64k up (seeds, s0.01, s0.02). That floor is the
  design artefact noted above. It is why v5q fails at 256k, and it could be removed by randomising the node-to-triple
  assignment (not done: frozen).

**Scope change (DRI after T271).**
- The s0.005_1 failure was diagnosed first (`diag.py`, within the 1-hour cap). It is structural and lies in the
  configuration assumption: one threshold per triple at one canonical background. That breaks on noisy copies in
  general. 107/720 triples have no threshold inside the probed range, a "picks rank >= 4" mode carries 2–11% of the
  mass, and the blind rate is model-specific.
- It is fixable only by abandoning the one-canonical-point-per-triple design (v5a's exact-law nodes). Even then the
  remaining canonical part (ranks 4–10) leaves 0.7–1.3 pp of flat bias on the seed and s.01/s.02 groups. That is 2–10x
  the amended lenient bias limit (.53 / .27 / .13 pp at 16k / 64k / 256k).
- The deductive-threshold idea (analytic margins from each node's gate pattern instead of bisection) is what v5b does
  for the lower ranks. It over-corrects by 0.9–2.1 pp, because the linearisation across gate changes over-states the
  spread.
- **Outcome 3 applies:** the probe's configuration assumption breaks on noisy copies in general, and the only variant
  that passes drops the probe ingredient entirely.

**HP1 submission.** `hp1_entry.py`: `est(model, budget) -> (estimate, cost_flops)` (= v5q) and `est_v5a`. v5.py is
unchanged since the pre-registration commits. Neither assumes the target's accuracy regime. Not run on hp1/sealed.

## Compute accounting
- FPE = 6000 FLOPs (T196, multiply-add = 2). v5a/v5q: 1 FPE per node. v5b: 3 FPE per node ×1.01 (forward + two
  reverse passes, which have the same FLOP count as a forward; margin algebra ≈ 40 FLOPs per rival).
- The node design (Sobol, Rényi transform, permutations, lower-rank covariance) is model-free and not charged, per
  the LB3 rules; it is cached across models.
- CPU: diagnosis + scans + DEV selection about 1.2 CPU-h; LB3 panels about 25 CPU-min. One heavy process at a time,
  `ulimit -v 2000000`.

## Claims
- P1 [measured] Probe v2's noisy-copy failures come from three model-independent assumptions: one threshold per
  triple (wrong when the net still errs at the largest probed gap), only two error modes, and a fixed 8/9 blind rate.
- P2 [measured] Reading the network's answer on exact-law nodes, with canonical lower ranks (v5a), cuts the bias to
  ≤ 1.7 pp on every LB3 model (orig +0.02 pp). That passes the lenient bar at 4k in 5 of 6 groups and fails in the
  seeds (upper CI 12.6).
- P3 [measured] The weight-derived spread (v5b), selected on DEV, does not transfer to the test models: it
  over-corrects by 0.9–2.1 pp.
- P4 [measured] The canonical lower ranks are the whole of v5a's bias. Drawing them from the exact law too (v5q)
  removes it and passes the lenient bar at 1k–64k. v5q is plain fixed-node QMC.
- P5 [inferred] **The probe ingredient (canonical values standing in for part of the input) only adds bias.** The
  method that passes is inductive QMC, the same class as sci_tao, tube and sci_laplace on LB3. It is not mechanistic
  evidence.

## Inductive / deductive ledger
| method | deductive parts | inductive / approximate parts | counts as |
|---|---|---|---|
| v5a | exact position and top-3 law; exact network answers | canonical lower ranks (approximation); fixed-node averaging (inductive) | lenient-bar evidence only with the inductive caveat; fails anyway |
| v5b | + exact local Jacobian from the weights | linear-Gaussian spread; DEV-selected variant (fitted choice) | fails |
| v5q | exact full law | fixed-node averaging over all 10 dims (inductive QMC) | passes; quadrature/QMC, not mechanistic |

## Alternatives not ruled out
- A v5a with lower ranks at a few exact-law quantile levels (between v5a and v5q) might trade a little bias for a
  little variance. Not tried, and it would be QMC anyway.
- v5q's large-budget bias may be the index-mod-720 / Sobol correlation; a random triple assignment would test that.
- The seed group has 5 models, so its CI is wide. One model (seed1) drives v5a's failure.

## Failure modes
- Canonical lower values must sit below the sampled M3 (construction bug, fixed before the scan).
- v5b's delta-method variance blows up for the lowest ranks (small density). Even restricted to ranks 4–6, the
  linear extrapolation across gates over-states error.
- The LB3 harness needs `alg_zoo`. Here a shim reads the same public checkpoints; an official re-run should use the
  real package.
