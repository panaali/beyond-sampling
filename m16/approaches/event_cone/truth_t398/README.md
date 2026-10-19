# T398 worker: ground-truth rates of E1(t0) on orig (conditional MC along exact v3 lines)

↑ [T373 event cone](../README.md) · [TASKS](../../../../TASKS.md) · [INDEX](../../../../INDEX.md)

Worker for T398 (certified upper bound on rare mistakes), under T398's board reservation (lead session
session_01LhvQ3Abu3KY6KE3AELyQUg). Writes only in this folder.

## Question
What are P(E1(t0)) on orig = M_16,10 for t0 in {2.0, 2.25, 2.5}, with relative s.e. <= 10%, where
E1(t0) = {network picks the position of the 3rd-largest input (p3), and v2 - v3 > t0}? T398 needs these as the
reference values for its certified upper bounds.

## Pre-registration (written and pushed before any number is produced; 2026-10-19)

**Method (sampling-class, unbiased; not mine).** T362's conditional Monte Carlo along exact v3 lines,
`m16/approaches/rare_events/cmc.py` from branch `claude/t286-cheap-frontier` @535d5bde, with its imports `rare.py`,
`../cheap_frontier/cf.py`, `../closure_floor/cont_g.py`, copied verbatim into `src/` (sha256 431c349e..., f3ce1dfe...,
72fb793a..., d34dde36...). For x ~ N(0, I_10), given everything except the value at p3, x_p3 is N(0,1) restricted to
(v4, v2); so Z = P(pick = p3, v2 - x_p3 > t0 | rest) is computed exactly by `rare.walk3` (the exact gate-cell walk
along x_p3 from v2 down to v4, with the T340 simultaneous-crossing fix) and E[Z] = P(E1(t0)). Inputs with
v2 - v4 <= t0 cannot be in the event and get Z = 0 without a walk.

**Only change:** `truth.py:run_e1(model, t0, ...)` is `cmc.run(model, "E1(1.0)", ...)` with t0 a parameter instead of
the literal 1.0; every other line is T362's. Check before the runs: `run_e1(t0=1.0)` must reproduce
`src/cmc.run("E1(1.0)")` bitwise on the same seed.

**Model.** orig via `m16/leaderboard.models(16, 10)["orig"]` (= `m16/common.py`, `alg_zoo.example_2nd_argmax`), float64.

**Chunks and seeds.** Chunks of `L` walked lines; chunk i uses `np.random.default_rng(398000 + i)`. Index ranges:
t0 = 1.5: i = 0..999; 2.0: 1000..1999; 2.25: 2000..2999; 2.5: 3000..3999. Each chunk appends one JSON line (sums of Z
and Z^2, inputs drawn, nonzero Z values, FLOPs, seconds) to `runs/t<t0>.jsonl`. The pooled estimate is
sum Z / n_drawn over all chunks, s.e. = sample sd of Z / sqrt(n_drawn). No chunk is ever discarded.

**Validation (gate).** t0 = 1.5, run to rel. s.e. <= 10% (cap 2 CPU-h, not counted in the 8 CPU-h). Reference:
T340's CPU plain-MC truth 2.16e-7 (1e9 inputs, ~216 events, s.e. ~1.5e-8). Pass iff |est - 2.16e-7| <= 2 s.e. of the
difference, sqrt(se_est^2 + se_ref^2) with se_ref = sqrt(2.16e-7 / 1e9). Also reported, not gating: the strict
|est - 2.16e-7| <= 2 se_est, and the GPU truth 2.09e-7 (1e10 inputs, s.e. 4.6e-9). If the gate fails I stop and
report to the lead; nothing is tuned to make it pass.

**Runs.** t0 in {2.0, 2.25, 2.5}: chunks until rel. s.e. <= 10% (checked after every chunk), or until 8 CPU-h total
across the three are spent. Allocation: each t0 starts with a cap of 8/3 CPU-h; time a t0 does not need goes to the
others. At most 2 processes at once, each under `ulimit -v 2500000`.

**Disclosed before running:** I have seen T364's conditional-law truths for orig, t0 = 1.75: 3.93e-8 and
t0 = 2.0: 6.83e-9 (`rare_events/README.md` on the T286 branch). They are an independent cross-check (different
sampler), reported after the fact, never used to stop or tune.

**Output.** `results.json`: est, se, n_drawn, n_lines, sec per t0 (plus the validation run).

## Status
Pre-registered; not run yet.
