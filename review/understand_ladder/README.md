# T97: independent check of E38 / H-U3 (fibre recipe F2L2 on the T = 3 ladder)

↑ [TASKS](../../TASKS.md) · [INDEX](../../INDEX.md) · reviewed: [understand_ladder](../../m16/approaches/understand_ladder/README.md) · previous reviews: [R2](../understand_m43/README.md)

Reviewer: DRI sub-agent (Claude session 011hwZqHr8rxiHqJ7yTBUF1z), 2026-10-17. I did not edit the reviewed folder.
The scripts here import the author's `fibre.py` read-only (the reference F2L2, commit 46b67b9, unchanged by the later
T95 commit). Where I needed a change, I made a separate copy (`fibre_strict.py`, `fibre_fix.py`). The T95 fast path
(`fibre_fast.py`) is **not** reviewed here.

## Question

Does E38 hold up? E38 claims that the frozen deductive recipe F2L2 beats sampling at equal ops on every model at
rungs (H,T) = (4,3), (5,3) and (6,3). That covers all 5 zoo seeds individually and noisy copies (ARC absolute
σ = 0.05; relative σ 0.0025-0.02). It also claims a cost of ~5k ops per linear region (10-20x exact enumeration),
with ops ∝ H^2.0 and regions ∝ H^1.84 (H-U3: the cost scales like the region count, not like the boundary families).

## Verdict (summary)

| item | verdict | key number |
|---|---|---|
| 1. Independent truth | **confirmed** | My own spherical-polygon enumeration equals the author's truth to ≤ 1.1e-15 on all 15 zoo models, and the region counts are identical. 1e7-sample MC of the actual torch modules: \|z\| ≤ 1.42 on all 15 |
| 2. Fresh noise seeds | **confirmed** | 184 fresh copies per rung (abs 0.05 seeds 101-102 ×64, abs 0.10 seed 103 ×32, rel seed 7 ×24): **0 per-model failures**. Worst per-model ratio 5.8e-3 / 1.9e-4 / 3.7e-9 at H = 4/5/6 |
| 3a. Determinism, nothing fitted | **confirmed, with one implementation bug** | No RNG; constants are numerical tolerances and caps (never hit). Zoo-seed results reproduce bit-identically. **But a `id()`-reuse bug in the cache makes F2L2 over-count error sometimes** (negative errors down to −3.6e-3 at L1, −7.4e-5 at L2). This contradicts the README's "error is always one-sided". Fixing it changes no pass/fail |
| 3b. Look-ahead choice (L = 2 chosen after seeing seeds) | **low risk** | L = 3 is better on every group at +3-7% ops (worst model 8e-4). L = 1 passes on the author's count (worst 0.53, M_4,3 seed 1) but would fail that seed on a strict count (~1.3) |
| 4. Op recount and cost law | **confirmed; cost understated by ~2.45x** | Strict count / author count = 2.39-2.53 (32 models): ~11-12k ops per region, not 4.5-5k. Ops slope 1.98 (strict) / 1.99 (author) vs regions 1.84, traces 0.97. F2L2 = 15-17x my own enumeration, ~40x the author's exact-enumeration count |
| 5. Adversarial weights | **robust inside ARC's noise scale, breaks well outside it** | Hill-climb inside the RMS-0.05 shell around the best seed: worst ratio 0.020 (H = 4), 4.5e-5 (H = 5), ~0 (H = 6). Unconstrained (RMS Δ 0.27-0.41, accuracy ~0.4): **F2L2 fails** with ratio 1.43 (H = 4) and 9.47 (H = 5), from missed islands. L = 3 fixes both (0.36, 0.06) |

**Recommendations to the DRI.**
- **E38 accuracy claim → confirmed (scoped).** Scope: at rungs (4,3), (5,3) and (6,3), on the 5 zoo seeds and on
  noisy copies of the best seed (absolute σ ≤ 0.1, relative σ ≤ 0.02), with exact truth, the frozen F2L2 beats
  sampling at equal ops on every model. This holds even with ops counted strictly: the worst model, M_4,3 seed 1,
  goes from 0.149 to ~0.38.
- Record these caveats in the ledger:
  - (a) The reference implementation has a cache bug, described in item 3a below. It produces over-counts, so the
    error is *not* one-sided, and the output depends in principle on Python memory reuse. A one-line fix exists
    (`fibre_fix.py`), and all rungs still pass after it.
  - (b) The author's op count is ~2.45x too low on a strict count.
  - (c) L = 2 is not robust far from the trained weights. A random σ = 0.2 copy of M_5,3 seed 0 already fails
    (1.6). L ≥ 3 is the safer frozen choice.
- **H-U3 → keep "refuted at T = 3", now independently checked.** At T = 3 and H = 4-8 the cost scales like the
  region count (or slightly faster): ops ∝ H^1.98-1.99, against regions ∝ H^1.84 (H^1.84 also up to H = 16 on my
  counts). The number of traces grows like H^0.97, but each trace costs O(T·H). This is scoped to this algorithm;
  T95's Ω(H²) argument is the author's and not reviewed here.

## Method and results

Environment: `source .venv/bin/activate`, `cd review/understand_ladder`, every run with `OMP_NUM_THREADS=1` and
`ulimit -v 2500000` (1 core, < 2.5 GB). Total compute for this review is about 45 CPU-min.

### 1. Independent truth: confirmed

`rtruth.py` was written from scratch:
- it loads weights from `alg_zoo` directly;
- it builds 8 octant triangles × 6 orderings;
- it recursively splits each by the sign of every step-2/3 pre-activation row (Sutherland-Hodgman on the sphere);
- it clips by L_truth − L_d > 0, and gives a dead pattern to class 0;
- it measures solid angles with the Van Oosterom-Strackee formula.

It shares no code with `arc_m43_repro/exact.py`.

- `python mc15.py` (→ `mc15.out`, `mc15.json`): on all 15 zoo models at H = 4, 5, 6, my exact accuracy minus the
  author's `truth` lies within [−6.7e-16, +1.1e-15], and my region counts equal the author's on all 15.
- On 1e7 N(0, I) inputs, the **actual torch module** with alg_zoo's own `task_2nd_argmax` labels gives z-scores in
  [−1.24, +1.42] against my exact value. The rms z is ≈ 0.9.
- At H = 16 my region counts (776-898, mean 838) also match the author's T95 table.

### 2. Fresh noise seeds: confirmed

`python score.py H 1,2,3` for H = 4, 5, 6 (→ `score_H{H}.{out,json}`, ~3-5 CPU-min per H). The noise seeds were
never used by the author: absolute σ = 0.05 with seeds 101 and 102 (64 copies each); absolute σ = 0.10 with seed 103
(32 copies, beyond ARC's σ); relative σ ∈ {0.0025, 0.005, 0.01, 0.02} × RMS(W) with seed 7 (6 copies each). All
are copies of the best-of-5 seed, plus the 5 zoo seeds. Truth is `rtruth.exact`. Ratios use the author's
convention: sampling MSE at equal ops = p(1−p) / (ops / (3H(H+1) + 3H)).

F2L2 (LOOK = 2), group MSE ratio / worst single model:

| rung | zoo seeds | abs 0.05 (128) | abs 0.10 (32) | rel 0.0025 / 0.005 / 0.01 / 0.02 (6 each) | per-model failures |
|---|---|---|---|---|---|
| M_4,3 | 3.5e-2 / 0.149 (seed 1) | 3.6e-4 / 5.8e-3 | 3.5e-4 / 2.3e-3 | 7e-6 / 8e-6 / 2.0e-4 / 6e-6 (worst 1.1e-3) | 0 / 189 |
| M_5,3 | 2.6e-3 / 3.4e-3 (seed 0) | 8.9e-7 / 1.9e-4 | 9.8e-7 / 2.5e-5 | ≤ 7e-10 | 0 / 189 |
| M_6,3 | ~1e-23 | 2.4e-11 / 3.7e-9 | ~0 | ~0 | 0 / 189 |

The zoo-seed numbers equal the author's bit for bit (estimates and op counts). The fresh-copy ratios agree in order
of magnitude with the author's (abs 0.05: 3.4e-4 / 2.9e-7 / 2.5e-10).

### 3a. Determinism, no fitting, and the cache bug

- **No fitting.** `fibre.py` and `fiber3.py` contain no RNG and no data-derived constant. The only numbers are the
  tolerances 1e-300, 1e-15, 1e-13, 1e-12 and 1e-9, the 64-piece loop cap and the trace budgets of 2e5/4e5. Traces
  per model are ≤ 1k, so the budget never binds.
- **Determinism.** Repeated runs, and the original vs. my instrumented copy (`recount.py` asserts equality on 32
  models), give bit-identical outputs.
- **Bug** (found from negative errors, which a missed-island-only method cannot produce). In `trace_F2`, the
  clamp-side state `stc = State(...)` is a temporary. `_walk` caches its successors under the key `(id(stc), gate)`.
  Once `stc` is freed, CPython reuses its address for the next fibre's `stc`, which may be a *different* linear
  piece. The cache then returns a wrong successor state.
  - Evidence: `diag_neg.py` / `diag_neg3.py`, on the adversarial H = 6 model. F2L2 over-counts the family
    ((0,1,2), s = −1, truth 1) by 2.1e-3: it reports 0.05530 where FX and brute force give 0.05316. When the same
    integration is re-run with extra allocations (`diag_neg3.py`), it gives 0.05281 instead: same inputs,
    different output.
  - `fibre_fix.py` changes one line, keeping `stc` alive in the cache. `python fixcheck.py` (→ `fixcheck.out`)
    removes almost all negative errors, e.g. M_4,3 abs0.05_n101_35 at L1: −3.6e-3 → +4.1e-5. Two M_4,3 copies keep
    a small residual at L2 (min −2.4e-6), so a second, smaller over-count source exists. I did not chase it.
  - `python scorefix.py H` (→ `score_fix_H{H}.*`) re-scores all 567 models with the fix. There are still 0
    failures, and the worst models are unchanged (0.149 / 3.4e-3 / 3.7e-9).
- In the unfixed reference, the bug sometimes **cancels** missed islands. The reported accuracy is therefore
  slightly luckier than the method; with the fix it is ≤ 2e-4 worse on the affected copies, and it still passes.

### 3b. Look-ahead 1 / 2 / 3 (the knob chosen after seeing zoo seeds)

These come from the same `score_H{H}.json` runs. Worst per-model ratio over all 189 models, with mean ops:

| rung | L = 1 | L = 2 (frozen) | L = 3 |
|---|---|---|---|
| M_4,3 | 0.53 (seed 1), 270-300k | 0.149 (seed 1), 295-330k | 8.2e-4 (seed 4), 303-333k |
| M_5,3 | 0.078 (seed 0), ~390k | 3.4e-3 (seed 0), ~450k | 3.9e-7, ~470k |
| M_6,3 | 4.9e-4, ~500k | 3.7e-9, ~600k | ~1e-22, ~655k |

L = 2 is not a knife-edge pick: L = 3 dominates it for 3-7% more ops, and every L ≥ 2 passes. L = 1 passes on the
author's count but not with margin: M_4,3 seed 1 has 0.53 now, and ~1.3 at the strict factor of item 4 [inferred].

### 4. Strict op recount and cost law: confirmed, cost understated

`python recount.py` (→ `recount.out`, `recount.json`) uses 5 zoo seeds + 3 abs-0.05 copies at each H ∈ {4, 5, 6, 8}.

`fibre_strict.py` is a verbatim copy that additionally charges every operation the author leaves uncharged, into a
separate counter:
- comparisons and abs in gate filtering (`ahead`, `same`, `ok`);
- the subtractions that build each order constraint;
- the 16 elementwise ops per constraint actually executed in `Cons.interval` (6 are charged);
- piece_ok re-evaluations and midpoints;
- the trig of θ and p(θ);
- root filtering in `trace_arc_rows`;
- the full cost of `_Sint`.

The asserted outputs and the author counts are identical to the original.

| H | regions (own) | author ops | strict ops | strict / author | strict ops / region | F2L2 strict / own enumeration |
|---|---|---|---|---|---|---|
| 4 | 66 | 295k | 729k | 2.40-2.53 | 11.2k | 11.8-17.6x |
| 5 | 97 | 462k | 1.14M | 2.42-2.49 | 11.8k | 14.9-19.0x |
| 6 | 134 | 644k | 1.58M | 2.39-2.47 | 11.8k | 14.7-16.5x |
| 8 | 236 | 1.18M | 2.89M | 2.42-2.46 | 12.3k | 14.7-16.7x |

My own enumeration is counted loosely (5 ops per vertex per clip, 26 per cut, 25 per solid-angle triangle, plus row
compositions). It costs 48k-188k ops, i.e. 2.7-3x the author's exact.py count, so F2L2 is ≥ 15x any honest
enumeration count here.

Log-log slopes over H = 4, 5, 6, 8 (leave-one-out range in brackets):

| quantity | slope |
|---|---|
| regions | 1.84 [1.77, 1.89] |
| author ops | 1.99 [1.92, 2.01] |
| strict ops | 1.98 [1.91, 1.99] |
| own enumeration | 1.98 [1.84, 2.09] |
| traces | 0.97 [0.90, 1.07] |

The author's law (ops ~H^2.0, regions ~H^1.84, traces ~H) is reproduced. Strict ops per region rise slowly
(11.2k → 12.3k). The cost claim is therefore **qualitatively confirmed and numerically understated**: ~11-12k strict
ops per region, not 4.5-5k.

Effect on the accuracy claim [inferred, since ratio ∝ ops]: multiplying ratios by ≤ 2.53 leaves every model passing.
The worst is M_4,3 seed 1 at ~0.38. Sampling is still charged the author's 3H(H+2) per sample, which does not count
the ReLU compares or the label; a strict sampling count would move the ratios back down.

### 5. Adversarial weights

- **Inside ARC's noise scale** (`python adv_ball.py 4 120 3 304`, `5 100 2 305`, `6 60 2 306` →
  `adv_ball_H{H}.*`). The search starts from the best seed + δ and greedily hill-climbs the F2L2 per-model ratio,
  keeping RMS(δ) = 0.05 exactly (60-120 steps, 2-3 restarts). Worst ratios found: 0.020 (H = 4, err +5.4e-4),
  4.5e-5 (H = 5), 6e-20 (H = 6). The fixed code gives the same values. **No break.**
- **Unconstrained** (`python adv.py H 100 150 20H`, → `adv_H{H}.*`). Phase 1: 100 random absolute perturbations of
  the 5 zoo seeds at σ ∈ {0.05, 0.1, 0.2, 0.4}. Phase 2: 150 greedy steps of σ = 0.03 from the worst.

  | H | phase-1 worst (σ) | after hill-climb | RMS Δ vs RMS W | accuracy | L1 / L2 / L3 / L4 / L6 ratio |
  |---|---|---|---|---|---|
  | 4 | 0.26 (σ 0.4) | **1.43** (err +8.5e-3) | 0.41 / 3.5 | 0.414 | 1.42 / 1.43 / 0.36 / 0.044 / 0.034 |
  | 5 | **1.60** (σ 0.2, no climbing) | **9.47** (err +2.2e-2) | 0.27 / 4.4 | 0.402 | 9.44 / 9.47 / 0.062 / ~0 / ~0 |
  | 6 | 2e-3 | 0.098 (err −2.1e-3 from the bug; fixed: +3.5e-4) | 0.47 / 5.1 | 0.511 | 0.23 / 0.098 / ~0 / ~0 / ~0 |

  These are the predicted failure mode: islands more than 2 pieces from both fibre ends. One is shown pointwise in
  `diag_neg2.py`: on the H = 6 model F2L2 has no wrong interval at θ = 2.6269, while FX and brute force show one at
  u ∈ (0.176, 0.217). A larger look-ahead removes the observed cases. At H = 5, phase 1 at σ = 0.1 had a worst ratio
  of 0.64 on the author's count, ~1.6 on the strict count [inferred]. That model is a copy of the undertrained seed
  0, not of the best seed used by the protocol.

## Compute accounting

Everything here is review compute; none of it is charged to the estimator. It comprises:
- exact enumerations: ~2.4k models × ~0.1 s;
- 1.5e8 MC forward passes of the torch modules (~1 CPU-min);
- ~2.2k F2L2/F2L1/F2L3 evaluations (0.4-1 s each);
- brute-force fibre checks (2e5 forward passes per θ sample).

The total is about 45 CPU-min, run on one core.

## Claims

- [measured] My exact truth equals the author's to ≤ 1.1e-15 on the 15 zoo models, and 1e7-sample MC of the torch
  modules agrees (|z| ≤ 1.42).
- [measured] F2L2 beats sampling at equal (author-counted) ops on all 567 models with fresh noise seeds at H = 4, 5,
  6. With the id-reuse fix it still does.
- [measured] Strict op count = 2.39-2.53x the author's. Strict ops per region are 11.2-12.3k, and F2L2 costs ≥ 15x
  enumeration.
- [measured] Ops ∝ H^1.98-1.99, regions ∝ H^1.84, traces ∝ H^0.97 (H = 4-8).
- [inferred] All rung passes survive the strict count (worst ~0.38).
- [measured] The reference `fibre.py` over-counts error through `id()` reuse of a temporary cache key, so errors
  are not one-sided.
- [measured] L = 2 fails on adversarial weights far from training (ratios 1.43, 9.47), while L = 3 passes them.
  Inside the RMS-0.05 shell no break was found (worst 0.02).

## Alternatives not ruled out

- My strict recount is one reviewer's convention: comparisons count 1, and the 16 ops per constraint are as
  implemented. A tighter implementation could get back much of the 2.45x, which is what T95 does; I did not review
  T95.
- The hill-climbs are short and greedy. A stronger in-shell adversary (gradient-free optimisation with more
  restarts, or aiming directly at gate switches inside wedges) might find a break inside σ = 0.05. I found none in
  ~550 evaluations.
- The residual over-count after the fix (−2.4e-6 on two M_4,3 copies) is undiagnosed. A validity-interval
  incompleteness is the likeliest cause.
- I did not run H = 8 noise sets beyond 3 copies, and I did not test T ≥ 4.

## Failure modes found

- `id()`-keyed cache with temporary objects (`fibre.py` `trace_F2`/`_walk`).
- Fixed look-ahead L = 2 under heavy perturbation: wrong islands detached from both fibre ends.

## Proposed change to the reviewed code (for the author; not applied)

In `fibre.trace_F2`, after `stc = State(...)`, add `cache.setdefault("_alive", []).append(stc)`, or key the cache by
the gate pattern instead of `id()`. Re-run `run_t3.py`. Consider freezing L = 3.
