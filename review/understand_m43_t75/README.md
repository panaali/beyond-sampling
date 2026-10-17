# T84: independent check of E37 / H-U2 (T75 generic resolve-by-impact recipe, all 5 seeds of M_4,3)

↑ [TASKS](../../TASKS.md) · [INDEX](../../INDEX.md) · reviewed: [understand_m43 (T75 sections)](../../m16/approaches/understand_m43/README.md) · protocol and truth: [arc_m43_repro](../../m16/approaches/arc_m43_repro/README.md) · previous review: [R2 / T74](../understand_m43/README.md)

Reviewer: DRI sub-agent (Claude session 011hwZqHr8rxiHqJ7yTBUF1z), 2026-10-17. The reviewed folders were not edited.
The reviewed code is `est75.py` with `RECIPE = dict(K=(4, 4), rule="adaptive", eps=0.1)`, as committed in d406c17.
The T85 agent is extending `est75.py` in place, but it only adds a separate `rule="margin"` path. My independent
re-implementation (below) matches the current file's `RECIPE` output to ≤ 7.8e-15 on every model I evaluated, so
the frozen recipe has not changed.

## Question

The E37 / H-U2 claim is this. One frozen generic recipe beats plain sampling at equal (strictly counted) ops on all 5
zoo seeds of M_4,3 and on their noisy copies, under ARC's protocol (absolute σ = 0.05 on all 32 weights). The recipe
resolves gates by impact with a 10% threshold, inside 24 half-chambers. It is deductive and its coefficients come
from the weights. The claimed MSE ratios on noise seeds 20-23 are 0.0002, 0.054, 0.003, 0.007 and 0.25.

## Verdict (summary)

| item | verdict | key number |
|---|---|---|
| 1. Fresh noise seeds 30-33, own truth | **confirmed** | ratios **0.0001 / 0.040 / 0.003 / 0.005 / 0.28 [0.23, 0.33]** (seeds 0-4, 1024 new copies each). exact.py vs own 1e7-sample MC on 45 models: rms z 0.97, max \|z\| 1.92 |
| 2. Selection risk (threshold, impact definition) | **10% is not a lucky pick for the threshold. The impact definition matters.** | With T75's impact, every ratio is < 1 on all seeds for thresholds 2-30%, on both the dev set and fresh data (worst: seed 3 at 30%, 0.81). With "depth only" impact (no weight factor), seed 3 fails (1.2-2.9) for thresholds ≥ 10%. On the dev set, 10% was not the best threshold (15% had a lower worst-seed ratio) |
| 3. Strict op recount, CPU time | **ops confirmed. On CPU time, sampling wins on 3 of 5 seeds** | Automatic count of every executed flop and comparison (both sides counted the same way) gives 1.04 × T75's ratio. CPU time: 23-35 ms per model vs 0.2 ms for sampling at equal ops. At equal CPU time the ratio is 0.01 / 8.2 / 0.69 / 1.03 / 66 |
| 4. Robustness | **partly** | σ from 0.005 to 0.1: every ratio < 1, but seed 4 reaches **0.69 [0.51, 0.87]** at σ = 0.015-0.02. The recipe is **not invariant to function-preserving rescalings**: seed 3 goes from 0.004 to 0.17, and single models from 0 to 5 pp. Accuracy-preserving box \|Δw\| ≤ 0.1 attack: worst 0.39 pp. A margin-aware step-3 impact (post hoc, mine) cuts seed 4 to 0.048 (σ 0.05) and 0.004 (σ 0.02) |
| 5. No random calls, no fitted constants | **confirmed** | Ran with all RNG entry points raising: no call. Bit-identical repeats. Every literal is a convention or a tolerance, apart from eps = 0.1 (item 2) |
| context | **dominated by exact enumeration** | exact.py is deductive and has MSE 0 at 18-22k strict ops. T75 costs 23-34k and is not exact. The recipe's own eps = 0 limit is exact at 33-39k |

**Recommendation for H-U2: confirmed, with scope.** On M_4,3 (all 5 trained seeds) under ARC's protocol (absolute
σ ≤ 0.1, ops counted strictly), the single frozen recipe beats plain sampling at equal ops on every seed. Fresh-seed
ratios are 0.0001-0.28, and seed 4 is the weakest (up to ~0.7 at σ ≈ 0.02). The ledger should keep these caveats.

- (a) On M_4,3 the recipe is **dominated by exact enumeration**: exact.py is cheaper and has zero error. So this is
  not evidence that understanding beats brute force here. The open question is cost scaling (T85).
- (b) The **impact definition** is a load-bearing design choice. Depth-only impact fails on seed 3. The recipe is
  also **not invariant to function-preserving neuron rescalings**, because the step-2 weight factor
  max_j |Wh[j,k]| depends on the arbitrary scale of the step-3 neurons. The ratio still stays < 1 under rescalings
  up to e^±2.
- (c) On **CPU time**, this numpy implementation loses to sampling on seeds 1, 3 and 4.
- (d) Seed 4 remains the weak point. Its margin is thinnest at **small** σ, not at large σ.

## Method and results

Environment: `source .venv/bin/activate`, `cd review/understand_m43_t75`, every run with `OMP_NUM_THREADS=1`
(1 core) under `ulimit -v 2500000`, and at most 2 of my processes at once.

**New code.** [recipe.py](recipe.py) re-implements the recipe in pure Python scalars. I wrote it from the T75
README's description, not by copying. It builds its own 24 half-chambers from the definition: ordering chamber ×
chart, split by x_max + x_min = 2 x_med. Their masses sum to 1.0000000000000004. It agrees with `est75` (RECIPE)
to at most **7.8e-15 on all 11,008 fresh models** evaluated (column `reimpl maxdiff`). That confirms the code does what
the README says. [mine.py](mine.py) holds my own weight loader (it asserts there are no biases), my own ARC noise
draws, my own forward pass (it matches torch: argmax agrees on 1e5 inputs, |Δlogit| ≤ 5e-4) and my own MC.

### 1. Fresh seeds and independent truth: confirmed

*Truth* (`python truth_mc.py`, `truth_mc.log`). I ran a 1e7-sample MC with my own forward pass against exact.py on
45 models: the 5 clean seeds; for each seed, 4 copies at σ = 0.05 (seed 30), 2 at σ = 0.1 (seed 40) and 2 at
σ = 0.02 (seed 50). Mean z = +0.20, rms z = 0.97, max |z| = 1.92, so exact.py is correct at the 1e-4 level on all 5
seeds.

*Fresh noise seeds 30, 31, 32, 33* (256 copies each per zoo seed, σ = 0.05). Command:
`python fresh.py 0.05 256 30 31 32 33`, table also in `report.log`. Ops are est75's own strict charge. The ratio is
MSE / (mean p(1−p) / (ops/72)). The CI is a bootstrap over copies, 1000 resamples.

| zoo seed | mean acc | RMSE | bias | max \|e\| | ops | **MSE ratio [95% CI]** | per noise seed | author (seeds 20-23) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.878 | 0.01% | −0.000% | 0.0% | 34.3k | **0.0001 [0.0000, 0.0001]** | 0.0001, 0.0001, 0.0000, 0.0001 | 0.0002 |
| 1 | 0.718 | 0.47% | +0.17% | 4.3% | 25.1k | **0.040 [0.030, 0.051]** | 0.049, 0.034, 0.038, 0.038 | 0.054 |
| 2 | 0.735 | 0.12% | −0.04% | 1.3% | 31.7k | **0.003 [0.003, 0.004]** | 0.004, 0.004, 0.002, 0.003 | 0.003 |
| 3 | 0.785 | 0.13% | −0.02% | 2.3% | 30.5k | **0.005 [0.003, 0.008]** | 0.004, 0.009, 0.004, 0.003 | 0.007 |
| 4 | 0.577 | 1.42% | +0.21% | 6.7% | 23.3k | **0.28 [0.23, 0.33]** | 0.34, 0.22, 0.32, 0.22 | 0.25 |

These reproduce the author's numbers within their CIs. Seed 4's worst single noise seed is 0.34 (author's worst:
0.28).

### 2. Selection risk: the threshold is fine; the impact definition is load-bearing

Command: `python sweep.py`, table in `report.log`. It uses recipe.py, whose charge is 0.92-1.02× est75's. It runs on
T75's dev set (noise seed 0, 32 copies per zoo seed, where eps and impact were chosen) and on fresh noise seed 30
(128 copies). Worst-seed ratio:

| variant | eps 0.02 | 0.05 | **0.10** | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|---|
| T75 impact (depth × max\|ΔW\|, scale = Σ max) — dev | 0.43 (s4) | 0.45 (s4) | **0.42 (s4)** | 0.24 (s4) | 0.30 (s3) | 0.36 (s3) |
| T75 impact — fresh | 0.31 (s4) | 0.40 (s4) | **0.33 (s4)** | 0.30 (s4) | 0.66 (s3) | 0.81 (s3) |
| depth only (w ≡ 1) — fresh | 0.31 | 0.42 | **1.18 (s3)** | 2.59 (s3) | 2.86 (s3) | 2.82 (s3) |
| depth × L2\|ΔW\| — fresh | 0.30 | 0.39 | 0.34 | 0.36 | 0.72 (s3) | 0.79 (s3) |
| scale = vertex mean — fresh | 0.31 | 0.43 | 0.39 | 0.36 | 0.33 | 0.28 |
| eps = 0 (exact limit) | ratio 0, at 33-39k ops | | | | | |

- **The threshold.** Every T75-impact setting from 2% to 30% beats sampling on every seed. The worst case is 0.81
  (seed 3, eps 0.3). So the claim does not hinge on 10%. The per-seed values do move a lot with eps: seed 3 goes
  from 0.003 at eps ≤ 0.1 to 0.14-0.81 at eps ≥ 0.15. 10% was also not the dev optimum by worst seed (15% was,
  0.24). So the pick is not "lucky", though it is not principled either.
- **Seed 4 is not monotone in eps.** Larger eps helps it (0.10 at eps 0.3). Smaller eps does not: it stays at
  0.31-0.45 down to eps = 0.02. Its error comes from gates whose impact is below 2% of the "surely-on" scale and
  that still matter. The large Wi0 = 45 inflates that scale while the logits cancel. Resolving more at step 3 only
  (eps = (0.1, 0.03)) or at step 2 only (0.03, 0.1) does not fix it either (0.35, 0.40).
- **The impact definition is load-bearing.** Dropping the downstream-weight factor breaks seed 3 at eps ≥ 0.1
  (ratio > 1). The L2 and vertex-mean variants are about equivalent to T75's.
- **Margin-aware step-3 impact** (mine; it compares the impact with the smallest frozen logit margin in the cell;
  `mmode="margin"`) gives a worst-seed ratio of 0.025-0.05 at eps 0.1-0.3, with every seed ≤ 0.05. It costs 29-42k
  ops, which is more than T75 and more than exact.py. I chose it after seeing seed 4's failure, so it is post hoc.

### 3. Op count and CPU time

*Automatic recount* (`python count.py 4`, `count.log`). recipe.py runs with every model-dependent number wrapped in
a counting type. Every executed `+ − × ÷`, abs, comparison, sqrt and atan2 counts 1, with no hand-assigned
constants. The same counter runs one sampling step: 3 RNN steps, ReLU, readout, argmax, the 2nd-argmax label and
the comparison, which totals 114 FLOPs. Over 25 models (5 clean and 20 noisy):

- The recipe's FLOPs are 1.64× T75's strict charge. In sample-equivalents that is 403 (FLOPs / 114) against 388
  (charge / 72).
- **Counting both sides the same way multiplies T75's ratios by 1.04.** The strict charge is therefore fair.
- Counting multiplies only (21.7k vs 56 per sample) gives about the same result.
- RNG cost for sampling is not counted, as in ARC's convention.

*CPU time* (`python cpu.py`, `cpu.log`, with the machine loaded at about 9). The reviewed est75 takes 23-35 ms per
model. My pure-Python re-implementation takes 6-8 ms. Vectorised sampling at equal ops (320-480 samples) takes
0.2 ms, which is 130-160× faster than est75. At equal CPU time, sampling would draw 72k-95k samples.

| zoo seed | ratio at equal CPU time (est75) | with recipe.py's time (≈ ÷ 4.5) |
|---|---|---|
| 0 | 0.01 | ≈ 0.002 |
| 1 | 8.2 | ≈ 1.8 |
| 2 | 0.69 | ≈ 0.15 |
| 3 | 1.03 | ≈ 0.23 |
| 4 | 66 | ≈ 15 |

ARC's protocol counts ops, so this does not refute H-U2. It does mean the advantage exists only under op counting,
in Python.

### 4. Robustness: partly

*Other σ* (fresh seeds; `python fresh.py <σ> 128 <seeds>`; seed 4 only, 256 copies: `ZOO=4 python fresh.py <σ> 256 <seed>`;
all in `report.log`):

| σ (abs.) | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 [CI] |
|---|---|---|---|---|---|
| 0.005 (52, 53) | 0.0008 | 0.013 | 0.001 | 0.017 | 0.032 [0.025, 0.039] |
| 0.01 (50, 51) | 0.0004 | 0.009 | 0.002 | 0.012 | **0.66 [0.47, 0.87]** |
| 0.015 (54, seed 4 only) | | | | | **0.69 [0.51, 0.87]** |
| 0.02 (40, 41) | 0.0002 | 0.008 | 0.002 | 0.006 | **0.64 [0.49, 0.83]** |
| 0.02 (56, seed 4 only) | | | | | 0.68 [0.51, 0.88] |
| 0.03 (55, seed 4 only) | | | | | 0.50 [0.37, 0.63] |
| 0.05 (30-33) | 0.0001 | 0.040 | 0.003 | 0.005 | 0.28 [0.23, 0.33] |
| 0.1 (40, 41) | 0.0000 | 0.086 | 0.031 | 0.008 | 0.28 [0.19, 0.39] |

- Every cell is < 1, and every upper CI is ≤ 0.88.
- Seed 4 is closest to failing at σ ≈ 0.01-0.02. There its RMSE is 1.7-2.0% with max errors around 7 pp.
  Copies near the clean model keep its doubled x1-vs-x0 switch (1.046 / 1.047) and its logit cancellation, while
  p(1−p) is still high.
- The margin-aware variant (`python seed4_margin.py`) brings seed 4 to 0.0036 (σ 0.02) and 0.0034 (σ 0.015), with
  max error 0.5 pp, at 29k ops.

*Function-preserving reparametrisations* (`python adversarial.py` part 1, `python rescale_eval.py`):

- Hidden permutations leave the estimate unchanged (≤ 2e-16).
- Positive per-neuron rescalings leave the truth unchanged (≤ 1.4e-15) but **change the estimate**. The step-2
  weight factor w2_k = max_j |Wh[j,k]| scales with the arbitrary scale of the step-3 neurons j.
  - Random log c ~ U(−1, 1) moves single estimates by up to 1.36 pp.
  - A hill climb over log c ∈ [−2, 2] takes two seed-3 noisy copies from 0.00 to **4.5 and 5.0 pp** error.
  - On 256 fresh copies per seed, rescaling each copy with log c ~ U(−2, 2) moves seed 3 from 0.004 to **0.17**,
    seed 1 from 0.049 to 0.058, and leaves seeds 0, 2 and 4 about the same.
- Because ARC fixes the parametrisation (zoo weights + noise), this does not break the claim. It does show that the
  impact proxy is not a property of the function.

*Weight attacks inside |Δw| ≤ 0.1 = 2σ on all 32 weights* (part 2; hill climbs of 3 × 100 steps per seed):

| zoo seed | accuracy-preserving worst (acc ≥ clean − 1 pp) | unconstrained worst (acc there) |
|---|---|---|
| 0 | 0.03 pp | 0.06 pp (0.831) |
| 1 | 0.22 pp | 1.43 pp (0.770) |
| 2 | 0.39 pp | 0.61 pp (0.680) |
| 3 | 0.28 pp | 0.53 pp (0.576) |
| 4 | 0.23 pp | **7.56 pp** (0.592) |

I found no accuracy-preserving perturbation that pushes gates into a damaging near-switch state. The T74
"τ-slope ≈ 1" window failure does not recur: the author's `window75.py` shows ±0.04 pp, and I did not re-scan it.

### 5. Deductive, no fitted constants: confirmed

`python audit.py` (`audit.log`):

- **RNG.** Every numpy and stdlib RNG entry point is replaced with a function that raises. est75 then runs on 35
  models (all 5 seeds, at σ = 0.05 and 0.1) and nothing raises. Repeated runs are bit-identical.
- **Literals.** The numeric literals in est75.py are 0, 1, 2, 3, 4, 5, 12, 13, 16 and 23 (op-charge constants and
  dimensions), 1e-9, 1e-12 and 1e-300 (tolerances, the latter two only in the T85 margin path), and **0.1 (eps)**.
- **Inputs.** est75 opens no file and imports only `HALVES` (model-free geometry) from est_mech. The token "exact"
  appears only in a docstring.
- **Coefficients.** Gates, A2, A3, logit maps and impact weights are closed-form functions of (Wi, Wh, Wo). The only
  free choices are eps = 0.1, the impact proxy and the arc-midpoint freeze state. Item 2 covers their selection
  risk. My re-implementation reproduces est75 exactly, without access to any truth.

## Compute accounting (this review)

- Truths: exact.py for about 15k models.
- MC truth: 45 × 1e7 forward passes (4.5e8 samples).
- Estimator runs: est75 and recipe.py on 11,008 fresh models; 34 sweep variants × 800 models; about 4k
  adversarial and rescaling evaluations.
- Total: about 50 CPU-min on 1 core per process, at most 2 processes at once, all well under 2 GB.

## Claims

- [measured] On 5120 never-used copies (noise seeds 30-33, σ = 0.05) the frozen recipe's MSE ratios are 0.0001,
  0.040, 0.003, 0.005 and 0.28 [0.23, 0.33]. exact.py agrees with a 1e7-sample MC on 45 models (max |z| 1.92).
- [measured] A re-implementation written from the README reproduces est75 to ≤ 7.8e-15 on 11,008 fresh models.
- [measured] With T75's impact, every eps in 2-30% gives ratio < 1 on all seeds. With depth-only impact, seed 3
  fails for eps ≥ 10%.
- [measured] A uniform FLOP+comparison count of both sides changes the ratios by ×1.04. CPU time is 130-160× that
  of sampling at equal ops. At equal CPU time, sampling wins on seeds 1, 3 and 4.
- [measured] For every σ from 0.005 to 0.1 the ratio is < 1 on every seed. Seed 4 peaks at 0.69 [0.51, 0.87]
  at σ = 0.015.
- [measured] The recipe is not invariant to function-preserving rescalings: seed 3 goes from 0.004 to 0.17, and
  single models by up to 5 pp.
- [measured] A margin-aware step-3 impact gives seed 4 0.048 (σ 0.05) and 0.004 (σ 0.02). This is post hoc and costs
  more ops.
- [inferred] On M_4,3 the recipe is dominated by exact enumeration (MSE 0 at 18-22k strict ops, from eval75.log;
  T75 costs 23-34k).

## Alternatives not ruled out / checks not done

- The hill climbs are short local searches (3 × 100 steps), so the worst cases are lower bounds.
- The sweep uses 128 fresh and 32 dev copies per seed. Ratios near 1 (seed 3 at eps ≥ 0.2) have wide CIs.
- I did not test relative noise, a compiled implementation for wall-clock time, or larger H and T, where the cost
  question lives (T85).
- The margin-aware variant is mine and post hoc. It differs from T85's `RECIPE85`, which I did not review.
- Truths for most models rely on exact.py, validated on 45 models across all seeds and three σ levels.

## Failure modes found

1. Seed 4 at small σ (0.01-0.02): ratio about 0.65-0.69, with max error about 7 pp.
2. Non-invariance to positive neuron rescalings: a function-identical model can get 5 pp error.
3. The impact proxy without the weight factor fails on seed 3.
4. Wall-clock time: it loses to vectorised sampling on 3 of 5 seeds.
5. On this model it is no cheaper than exact enumeration.

## Proposed changes (for the T85 owner, not made here)

- Make the step-2 weight factor scale-invariant. For example, use the step-3 effect propagated to the readout,
  max_o |Σ_j (Wo[t,j] − Wo[o,j]) Wh[j,k]| over the surely-on j. Then test it under random rescalings.
- Adopt a margin-aware step-3 impact, frozen before new fresh seeds.

## Files

[recipe.py](recipe.py) (independent re-implementation and variants), [mine.py](mine.py) (weights, noise, forward
pass, MC), [count.py](count.py) / `count.log` / `count.json` (automatic op count),
[fresh.py](fresh.py) / `fresh_*.json` / `fresh_*.log` / `fresh4.log`, [truth_mc.py](truth_mc.py) / `truth_mc.log` /
`.json`, [sweep.py](sweep.py) / `sweep.json`, [adversarial.py](adversarial.py) / `adversarial.log` / `.json`,
[rescale_eval.py](rescale_eval.py) / `rescale_eval.log` / `.json`, [seed4_margin.py](seed4_margin.py) /
`seed4_margin.log`, [cpu.py](cpu.py) / `cpu.log`, [audit.py](audit.py) / `audit.log`, and
[report.py](report.py) / `report.log` (all tables rebuilt from the JSONs; some original logs were lost when another
agent's checkout replaced files mid-run).
