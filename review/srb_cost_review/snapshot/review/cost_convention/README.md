# T196: cost-convention audit (estimator ops vs sampling cost)

↑ [AGENTS](../../AGENTS.md) · [TASKS](../../TASKS.md) · [INDEX](../../INDEX.md) · [understand_m43](../../m16/approaches/understand_m43/README.md) · [estT_fix](../../m16/approaches/estT_fix/README.md) · [leaderboard](../../m16/leaderboard.py) · [ARC accounting notes](../../lit/web_arc/README.md)

## Question

Every estimator is compared with plain sampling at equal compute:

ratio = err² / (p(1−p)/N), with N = estimator cost / sampling cost per input.

Two conventions for the sampling cost are in use.
- **T = 3 rungs (E36, E37, E40, E43):** T·H·(H+1) + T·H "ops" per input.
- **E46 and E48:** `leaderboard.fpe_flops`, which gives 666.67 at (8,4) and 833.33 at (8,5).

This audit asks four things:
1. What does each side count?
2. Is the unit the same on both sides?
3. Which single convention should the project use?
4. What do the published ratios become under that convention, and does any pass/fail verdict flip?

**Short answer.** E46 and E48 mix units. Their estimator side uses the `m43.Ops` count (one multiply-add = 1 op), but
it is divided by a sampling cost in FLOPs (one multiply-add = 2 FLOPs). That understates N, and so flatters the published
ratios by up to 1.95x. The T = 3 rungs use the same unit on both sides, but they charge the estimator for comparisons
and ReLUs that they do not charge to sampling. That makes their ratios slightly pessimistic, by 0.88-0.94x. No verdict
flips: every restated ratio stays below 1, and the worst case is 0.64.

## Method

I read the counters and the scoring scripts, then reproduced the published ratios with arithmetic on the saved
results. No model was evaluated. Script: [restate.py](restate.py).

### 1. What `m43.Ops` counts

[`m16/approaches/arc_m43_repro/m43.py`](../../m16/approaches/arc_m43_repro/m43.py)

- `Ops` (lines 44-50) is a bare accumulator. Its docstring (line 45) and the module header (lines 4-6) set the unit:
  **one multiply-add = 1 op, and one sqrt, division or atan2 = 1 op.** Mul and add are therefore *not* counted separately.
- Clipping charges 3 per vertex (line 65). That is a 3-D dot product; the sign test is not charged. Each new vertex
  charges 6 + 4 (line 83). The 3 divisions of the normalisation are not charged.
- A solid-angle triangle charges 9 + 9 + 3 = 21 (line 100).

There are stricter variants. They keep the same unit but also charge every comparison, abs or sign test 1, and
every division 1:
- T74's [`ops_audit.py`](../understand_m43/ops_audit.py) (lines 4-8).
- `est75.py`, used by E37, E40 and E43 ([file](../../m16/approaches/understand_m43/est75.py), lines 17-18). It charges
  clipping 4 per vertex (line 41), 13 per new vertex (line 60), and a triangle 23, plus 1 per call (lines 72-73).

E36 used the lenient `m43` geometry. R2 measured a strict recount at 1.24x.

In [`estT_fix/estT.py`](../../m16/approaches/estT_fix/estT.py), the d = 4 and d = 5 estimator used by E46 and E48, the
counter is also `m43.Ops` (line 29). The charges follow the same unit:
- A d-dimensional dot product plus its sign test costs d + 1 (lines 98, 247, 255).
- A new ray costs 3d + 3 (line 108). The two rank tests cost 2d³ (line 121). Face tests cost len(S) + d³
  (lines 200, 205).
- **Orthant evaluations** are charged by hand-assigned totals per simplex. Transcendentals are folded into these totals
  at roughly 1-2 ops each. For example, the d = 3 asin costs 2 (line 139), and the "14 + 20" per node per pair at
  line 180 covers a 3x3 cofactor plus atan2, sqrt and the product terms.
  - d = 3: 2·27 + 9 + 6 = 69.
  - d = 4: 144 + 320 + 120 + 128·(6·34 + 30) ≈ 30.5k (lines 133, 160, 166, 180-181). The thin branch is 144 + 30k
    (line 153).
  - d = 5: 125 + 1175 + 240 + 128·(620 + 720) ≈ 173k (lines 521, 530, 535, 546, 557). The thin branch is about
    52k (line 523).
- The frozen-network forward passes inside the estimator charge **H² + 2H per step**: H² + H multiply-adds plus H
  ReLUs (line 363), plus H·T for the readout (line 368). In `est75.py` the same pass costs 4H + 3H per point
  (line 246). This is 3 multiply-adds plus 1 ReLU per neuron, then the readout.

### 2. What each sampling charge counts

| item | E36/E37/E40: `OPS_PER_SAMPLE` = 72 ([m43.py:16](../../m16/approaches/arc_m43_repro/m43.py)) | E43: `ops_per_sample(H,3)` ([zoo44.py:22-23](../../m16/approaches/understand_m43/zoo44.py), used at [evalH.py:18,42](../../m16/approaches/understand_m43/evalH.py)) | E46/E48/LB3: `fpe_flops` ([leaderboard.py:127-133](../../m16/leaderboard.py)) |
|---|---|---|---|
| unit | multiply-add = 1 | multiply-add = 1 | FLOP, multiply-add = 2 |
| Gaussian input (RNG) | not charged (`m43.py:39`) | not charged (`zoo44.py:15`) | not charged (rule 5, `leaderboard.py:25-26`) |
| recurrence + input, T steps (step 1 on h0 = 0 included) | T·H·(H+1) | T·H·(H+1) | 2·T·H·(H+1) |
| ReLU | not charged (`m43.py:31`) | not charged (`zoo44.py:17`) | not itemised (inside the 6000/5760 = 1.042 scale) |
| readout | T·H | T·H | 2·T·H |
| argmax of logits | not charged (`m43.py:40`) | not charged (`zoo44.py:18`) | not itemised |
| true 2nd-argmax (full `argsort` as implemented) | not charged (`m43.py:40`) | not charged (`zoo44.py:18`) | not charged (rule 5: "its label (sort) is not charged") |
| value | 72 at (4,3) | 72 / 105 / 144 / **240 at (8,3)** (320 is the (8,4) value) | 6000·flops(H,T)/5760: 666.67 at (8,4), 833.33 at (8,5), 6000 at (16,10) |

Here `flops(H,T)` = T(2H² + 2H) + 2HT is exactly 2 × `ops_per_sample(H,T)`. So `fpe_flops` = 2.083 × `ops_per_sample`.

### 3. Is the unit the same on both sides?

| exp | estimator counter | sampling divisor | same unit? | net effect on the published ratio |
|---|---|---|---|---|
| E36 | `m43.Ops`, lenient: no comparisons in clip/frac ([run_mech.py:52](../../m16/approaches/understand_m43/run_mech.py)) | 72 | yes (multiply-add = 1) | the estimator leaves out comparisons (strict ×1.24, R2), and sampling leaves out ReLU, argmax and label (72 vs 91). These roughly cancel: ×0.98 in ops |
| E37 / E40 | strict `est75` ([eval75.py:50](../../m16/approaches/understand_m43/eval75.py)) | 72 | yes | the estimator pays for comparisons and ReLUs; sampling does not. The published ratio is too high: ×0.79 in ops |
| E43 | strict `est75` | T·H(H+1) + T·H | yes | same asymmetry: ×0.79-0.89 in ops |
| E45, E46 | `m43.Ops` via `estT` ([final3/run.py:35](../../m16/approaches/final3/run.py), [estT_fix/summary.py:6,11](../../m16/approaches/estT_fix/summary.py)) | `fpe_flops(8,4)` = 666.67 FLOPs | **no**: ops ÷ FLOPs | N understated about 2x (1.84 in ops; 0.98-1.96 in FLOPs for κ = 1-2), so the ratio is flattered by the same factor |
| E48 | `m43.Ops` via `estT` (run5/zoo0_*.json; no script computes the ratio, and restate.py reproduces 0.0097) | `fpe_flops(8,5)` = 833.33 FLOPs | **no** | same as E46 |
| LB3 | each estimator reports FPE from its own FLOP count. For example, propagation uses FLOPs/6e3 ([estimator.py:2,15](../../m16/approaches/propagation/estimator.py)) and cone_enum uses FLOPs/5760 ([estimator.py:2-3](../../m16/approaches/cone_enum/estimator.py)) | 1 FPE per input = 6000 FLOPs at (16,10) ([leaderboard.py:148-150](../../m16/leaderboard.py)) | yes, by construction (FLOPs). The completeness of each count is [estimator_audit](../estimator_audit/README.md)'s subject | 6000 vs the recommended 5948: ×1.009 (×0.968 for /5760) |

## Recommended convention (one, for every experiment)

**FLOPs on both sides.**
- A multiply-add costs 2 FLOPs.
- Every other scalar operation costs 1 FLOP: add, compare, max/ReLU, abs, sign test, sqrt, division, and
  atan2/asin/exp.
- Sampling is charged per input of M_H,T for:
  - the forward pass as ARC writes it, 2·T·H·(H+1);
  - the ReLUs, T·H;
  - the readout, 2·T·H;
  - the argmax over the T logits, T − 1;
  - the true-label selection by a top-2 linear scan of the T inputs (not a full sort), 2T − 3;
  - the label comparison and the accumulation, 1 + 1.
- RNG is not charged, as in ARC's S8/S9 ("random-number generation is excluded").

So c_s(H,T) = flops(H,T) + T·H + 3T − 2:

| (H,T) | (4,3) | (5,3) | (6,3) | (8,3) | (8,4) | (8,5) | (16,10) |
|---|---|---|---|---|---|---|---|
| c_s (FLOPs) | 163 | 232 | 313 | 511 | 682 | 853 | 5948 |

The leaderboard's 6000 FLOPs per FPE at (16,10) is within 0.9% of this, so LB3 needs no change.

**Estimator side.** The per-cell counters (`m43.Ops`, `est75`, `estT`) lump multiply-adds with other ops. Their FLOPs
therefore lie in [ops, 2·ops]. Until each counter splits the two (proposal: `Ops.ma(k)` for multiply-adds and
`Ops.add(k)` for the rest), charge **2·ops**. This is the bound least favourable to the estimator.

Restated ratio = published × κ · c_pub / c_s, with κ = 2 by default and κ = 1 as the most favourable bound.

## Results

Command: `python3 review/cost_convention/restate.py` (from the repo root; numpy only; 0.2 s).

| experiment | model | published ratio | restated ratio (κ = 2) | factor | restated, κ = 1 |
|---|---|---|---|---|---|
| E43 | (4,3) zoo4 noisy (worst) | 0.00273 | 0.00241 | 0.883 | 0.00121 |
| E43 | (5,3) zoo4 noisy (worst) | 0.0416 | 0.0377 | 0.905 | 0.0188 |
| E43 | (6,3) zoo3 noisy (worst) | 0.0189 | 0.0174 | 0.920 | 0.00871 |
| E43 | (8,3) zoo2 noisy (worst) | 0.136 | **0.128** | 0.939 | 0.0641 |
| E46 | (8,4) zoo0 | 9.73e-07 | 1.9e-06 | 1.955 | 9.51e-07 |
| E46 | (8,4) zoo0_noisy0 | 3.25e-05 | 6.36e-05 | 1.955 | 3.18e-05 |
| E46 | (8,4) zoo0_noisy1 | 1.78e-06 | 3.49e-06 | 1.955 | 1.74e-06 |
| E46 | (8,4) zoo0_noisy2 | 1.68e-08 | 3.28e-08 | 1.955 | 1.64e-08 |
| E46 | (8,4) zoo0_noisy3 | 6.82e-06 | 1.33e-05 | 1.955 | 6.67e-06 |
| E46 | (8,4) zoo1 | 0.00246 | 0.0048 | 1.955 | 0.0024 |
| E46 | (8,4) zoo2 | 0.249 | 0.487 | 1.955 | 0.243 |
| E46 | (8,4) zoo3 | 2.21e-09 | 4.33e-09 | 1.955 | 2.16e-09 |
| E46 | (8,4) zoo4 | 0.268 | **0.523** | 1.955 | 0.262 |
| E48 | (8,5) zoo0 | 0.00971 | **0.019** | 1.954 | 0.00948 |

Notes on the table:
- The E43 (8,3) bootstrap CI [0.05, 0.25] scales to [0.048, 0.23].
- E36 and E37 at (4,3) scale by 0.883 at κ = 2. E36 also carries R2's strict recount of ×1.24, so 0.019 becomes about 0.021.
- E45's (8,4) ratios (8-48, superseded) had the same mixed units. Restating makes them about 2x worse; they were
  already fails.
- **No pass/fail verdict flips.** It does not flip at κ = 2. It does not flip at κ = 2 with sampling also charged
  without the useless step-1 recurrence on h0 = 0 (2H² FLOPs, 15-25% of c_s at these rungs). That worst case gives
  E43 at most 0.171, E46 at most 0.644 (zoo4) and E48 0.022.
- The ops convention (multiply-add = 1 on both sides, with the same sampling items: 91 / 271 / 362 / 453 ops at
  (4,3) / (8,3) / (8,4) / (8,5)) lands between the κ = 1 and κ = 2 columns. It gives E46 zoo4 0.49 and E48 0.018.

## Compute accounting

- No model forward passes and no estimator runs. The audit reads saved JSONs: `understand_m43/evalH_{4,5,6,8}.json`,
  `estT_fix/summary.json` and `estT_fix/run5/zoo0_{exact,recipe108}.json`.
- `restate.py` takes 0.2 CPU-seconds. Total CPU for the audit is under 1 s.

## Claims

- [measured] `m43.Ops`, `est75` and `estT` all count one multiply-add as 1 op, and sqrt/div/atan2 as 1 op. The
  strict variants also count every comparison or ReLU as 1 op. Transcendentals inside the orthant quadratures are
  folded into hand-assigned per-node totals.
- [measured] No sampling charge in any experiment includes the RNG, the argmax, or the true-label sort. The T = 3
  charges also leave out ReLU. `fpe_flops` covers the ReLUs only implicitly, through its 4.2% scale-up.
- [measured] E46 and E48 divide a multiply-add = 1 op count by a FLOP-based (multiply-add = 2) sampling cost. That is
  mixed units. restate.py reproduces E48's 0.0097 and E46's 9 ratios from the saved results with `fpe_flops`.
- [measured] Under the recommended convention with κ = 2, the worst restated ratios are: E43 0.128 ((8,3) zoo2),
  E46 0.523 ((8,4) zoo4), E48 0.019. The restatement is pure arithmetic.
- [inferred] No published pass/fail verdict flips for any κ in [1, 2], with or without charging sampling for the
  step-1 recurrence. The worst case is 0.644.
- [inferred] E46's headline "ratio ≤ 0.27" should read "≤ 0.52 (between 0.26 and 0.52 depending on the
  multiply-add share of estT's ops)". E48's 0.0097 should read 0.019 (between 0.0095 and 0.019).
- [speculative] The true κ for estT and est75 is probably 1.8-1.95, because the charges are dominated by dot
  products, determinants and matrix products. So the κ = 2 column is probably within about 10% of the true value.

## Alternatives not ruled out

- **κ is not measured.** The counters lump multiply-adds with other ops. The exact FLOP count needs split
  counters, or an instrumented run (flopscope-style) of est75 and estT. Doing that would change only the size of the
  correction, not the direction, since every published ratio is already within [κ = 1, κ = 2] × the factors above.
- **Counter completeness is not audited.** This audit checks units, not whether every operation is charged. Examples
  of operations that may be undercharged:
  - `np.linalg.matrix_rank` (an SVD) is charged d³ (estT lines 91, 121, 205).
  - The `np.linalg.inv` at estT line 131 is computed for d = 4 although it is unused there.
  - Python bookkeeping is not charged.

  R2 and R4 found strict recounts of 1.24x and 2.4x on earlier recipes. A similar factor here would multiply the
  restated ratios. A further ×1.9 completeness factor would push E46 zoo4 to 1 (×1.55 with the efficient sampler).
  E43 has at least 5.8x headroom and E48 more than 40x. This deserves a separate check (proposed follow-up).
- **The forward formula is a choice.** I kept ARC's T·H·(H+1), which includes the step-1 recurrence on h0 = 0, for
  comparability with ARC's 72-op M_4,3 count and with the leaderboard. Charging the efficient sampler instead raises
  every ratio by 1.18-1.33 at these rungs. The figures are in the last block of restate.py's output.
- **Label selection.** I charged a top-2 linear scan (2T − 3), not the `argsort` the code actually runs. A full sort
  would make sampling slightly dearer, and so the estimator's ratios slightly lower.
- **LB3.** Its unit consistency rests on each estimator's own FLOP count. I spot-checked only propagation (/6e3) and
  cone_enum (/5760). `exact_ladder` divides `info["flops"]` by `fpe_flops`; I did not check whether its own `Ops`
  counts a multiply-add as 2.

## Failure modes

- The restatement assumes the estimator's error does not depend on the cost convention. That holds: these estimators
  are deterministic, and only N changes.
- If a counter charges something above 2 FLOPs per op (for example, if a future convention weights transcendentals
  more than 1 FLOP), κ = 2 is no longer an upper bound.
- E48 is one model. E46 has 9 models with no CI, because each is a deterministic estimate scored against exact truth.
  A restated ratio is only as stable as the published one.

## Proposed follow-ups (for the DRI; not done here)

1. Split `Ops` into `ma()` (multiply-add) and `add()` (other), and re-run one E43 rung and one E46 model to measure κ.
2. Make `leaderboard.fpe_flops` and `zoo44.ops_per_sample` one function in one unit. Have every op-counted per-cell
   experiment divide by it after converting ops → FLOPs, never ops ÷ FLOPs (E45/E46/E48 in `final3/run.py:35` and
   `estT_fix/summary.py:11`).
3. Run a completeness recount of estT's charges (matrix_rank, unused inverse), like R2/R4, before E46 is confirmed.
