# T398 worker: ground-truth rates of E1(t0) on orig (conditional MC along exact v3 lines)

↑ [T373 event cone](../README.md) · [TASKS](../../../../TASKS.md) · [INDEX](../../../../INDEX.md)

Worker for T398 (certified upper bound on rare mistakes), under T398's board reservation (lead session
session_01LhvQ3Abu3KY6KE3AELyQUg). Writes only in this folder.

## Question
What are P(E1(t0)) on orig = M_16,10 for t0 in {2.0, 2.25, 2.5}, with relative s.e. <= 10%, where
E1(t0) = {network picks the position of the 3rd-largest input (p3), and v2 - v3 > t0}? T398 needs these as the
reference values for its certified upper bounds.

## Method
See the pre-registration below (sampling-class; T362's conditional MC with t0 as a parameter). Key files:
`truth.py` (run_e1 and chunk driver), `make_results.py` (pooling and the validation gate), `src/` (verbatim copies).

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
Done (2026-10-19). Validation passed; t0 = 2.0 / 2.25 / 2.5 stopped by the CPU budget, at 11.6% / 18.3% / 44% rel. s.e.
(none reached the 10% target).

## Results
`results.json`, pooled over all chunks (Z = 0 for screened inputs; s.e. = sd(Z) / sqrt(n_drawn)):

| t0 | est P(E1(t0)) | s.e. | rel. s.e. | nonzero lines | walked lines | inputs drawn | CPU s |
|---|---|---|---|---|---|---|---|
| 1.5 (validation) | 2.41e-7 | 2.34e-8 | 9.7% | 204 | 3.30e6 | 1.10e8 | 2,988 |
| 2.0 | **7.57e-9** | 8.8e-10 | 11.6% | 136 | 9.00e6 | 2.39e9 | 9,517 |
| 2.25 | **1.12e-9** | 2.0e-10 | 18.3% | 47 | 6.96e6 | 5.97e9 | 9,395 |
| 2.5 | **6.6e-11** | 2.9e-11 | 44% | 8 | 4.24e6 | 1.29e10 | 9,569 |

**Validation (gate): PASS.** t0 = 1.5: 2.41e-7 vs T340's CPU truth 2.16e-7 (s.e. 1.47e-8): z = 0.91 on the
s.e. of the difference (pre-registered gate), z = 1.07 on my s.e. alone (also within 2), z = 1.35 vs the GPU truth
2.09e-7. Code-path check: `run_e1(t0 = 1.0)` reproduced `cmc.run("E1(1.0)")` bitwise (Z, n_drawn, FLOPs, cells;
3000 lines, seed 362000).

**Cross-check (disclosed in advance, not used for stopping):** T364's conditional-law MC truth for orig t0 = 2.0 is
6.83e-9; mine 7.57e-9 +- 0.88e-9, z = 0.84.

Commands (from this folder, `.venv` = the repo's venv, each under `ulimit -v 2500000`):
```
python truth.py 1.5 0 100000 3600 & python truth.py 1.5 500 100000 3600     # validation, 2 processes
python truth.py 2.0 1000 100000 9600 & python truth.py 2.25 2000 10000 9400
python truth.py 2.5 3000 10000 4800 & python truth.py 2.5 3500 10000 4800
python make_results.py                                                       # -> results.json
```
Chunk i used seed 398000 + i; chunks per t0 are listed in `runs/t<t0>_<first chunk>.jsonl` (one line per chunk,
with the nonzero Z values). Logs in `logs/` (the t1.5 / t2.0 / t2.25 logs stop early: a git autostash replaced the
files while the processes were writing; the jsonl run files are complete, as they are reopened per chunk).

**Deviation (disclosed):** the first t0 = 2.25 launch used 1e5-line chunks and died of MemoryError (about 860 draws
per walked line, so a 650 MB array of screened zeros) after ~3 min, before writing any chunk
(`logs/t2.25_2000_oom.log`). It was relaunched with 1e4-line chunks from the same first index 2000 and a 9,400 s cap.
Nothing was discarded. Also, the validation was first launched with a 7,200 s cap per process (4 CPU-h, above the
pre-registered 2); I killed those two PIDs within a minute, before any chunk was written, and relaunched at 3,600 s.

## Compute accounting
- Model work: exact walk3 gate-cell walks, one per walked line (the T362 FLOP count, as executed, in FPE = 5948 FLOPs):
  t0 = 1.5: 3.21e7 FPE; 2.0: 9.67e7; 2.25: 7.79e7; 2.5: 4.92e7 (`C_fpe` in results.json).
- Input generation: n_drawn standard-normal 10-vectors plus an argsort each (table above). At t0 = 2.5 this dominates
  (3,032 draws per walked line).
- CPU time (sum of chunk wall times, one core each): validation 2,988 s; t0 = 2.0 to 2.5: 28,481 s, plus ~200 s for
  the crashed launch = ~7.97 CPU-h, inside the 8 CPU-h budget.

## Claims
- [measured] P(E1(2.0)) = 7.57e-9 +- 0.88e-9, P(E1(2.25)) = 1.12e-9 +- 0.20e-9 on orig.
- [measured] P(E1(2.5)) = 6.6e-11 +- 2.9e-11, from only 8 nonzero lines: treat it as an order of magnitude
  (the s.e. itself is poorly estimated with 8 events; a Poisson 95% interval on the count alone is about 3e-11 to 1.3e-10).
- [measured] The t0-parametrised code reproduces T340's t0 = 1.5 truth (z 0.91) and T362's t0 = 1.0 code bitwise.
- [inferred] The estimator is unbiased (exact conditional expectation along x_p3, given the walk is exact; T340's walk
  fix was validated against dense forwards to 5e-4 in t by the T286 child).

## Alternatives not ruled out
- A walk error on rare lines (e.g. a missed gate flip) would bias every t0 the same way; it is checked only through
  the t0 = 1.5 gate and the T364 cross-check at 2.0 (both pass), not at 2.25 or 2.5.
- Heavy tail in Z: at 2.5 the estimate rests on 8 lines, and the largest Z = 0.29 carries a large share; the s.e.
  can be underestimated.
- Not done: an independent sampler at 2.25 / 2.5 (T364's conditional-law sampler would be the natural check).

## Failure modes
- The cost per event grows fast with t0: the screen v2 - v4 > t0 passes 1 in 266 / 858 / 3,032 inputs at
  2.0 / 2.25 / 2.5, and the fraction of walked lines with Z > 0 falls (1.5e-5 at 2.0, 1.9e-6 at 2.5). Reaching 10%
  at 2.5 would need roughly 20x this budget. A tighter screen or a conditional-law sampler of g23 > t0 (T364) would
  be the fix; not tried here, per the task (same code path).
- Memory: screened zeros are materialised per chunk; chunks must shrink as t0 grows (1e4 lines at t0 >= 2.25).
