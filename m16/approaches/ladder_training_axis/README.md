# T101: the training-amount axis of the ladder, M_4,3 at n_seqs = 2^28 down to 2^20

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T75 recipe (E37)](../understand_m43/README.md) · [fibre recipe (E38)](../understand_ladder/README.md) · [R4 review / cache fix](../../../review/understand_ladder/README.md) · [R3 review / strict ops](../../../review/understand_m43_t75/README.md)

## Question

At the solved rung M_4,3 = `alg_zoo.zoo_2nd_argmax(4, 3)`, do the two frozen mechanism-derived recipes keep beating
sampling as the model becomes **less trained**? The models are `zoo_2nd_argmax(4, 3, n_seqs=2**k, seed=s)` for
seeds 0-4. They are checkpoints of the same five training runs, so k = 30 is the fully trained model of E37. The
ladder protocol (TASKS.md) says: one step at a time, stop at the first failing rung, diagnose, fix, re-check earlier
rungs. The two recipes are:

- **T75**: `est75.RECIPE = dict(K=(4,4), rule="adaptive", eps=0.1)` (E37/H-U2). It resolves the gates that matter by
  impact inside 24 half-chambers and freezes the rest.
- **F2L3**: the fibre recipe F2 with look-ahead L = 3 (E38), with R4's `id()` cache fix.

The first training-amount rung, 2^29, was done by Codex in T90 (board note: passed all 5 seeds with an independent
T55 check; folder `m16/approaches/training_axis_ladder/`, commit 6d47ac7). **That folder and commit are not on this
branch (`ccr-6a61f25c-wn80ii`), so I could not read or reuse its exact panels.** Following the DRI's instruction I did
not redo 2^29. I used the E37 panel convention, described below, and report k = 30 as the harness check.

**Short answer.** Yes, on every rung tested: k = 30 (harness check), 28, 27, 26, 25, 24, 23, 22, 21 and 20. Both
frozen recipes beat sampling on all 5 seeds, on their 1024-copy noisy panels and on the clean models. **No rung
failed, so there was no failure to diagnose, fix or stop at.** The tightest case is T75 on seed 4 at k = 24: MSE ratio
0.59 with a 95% CI upper bound of 0.66. 18% of individual copies there exceed 1. It is diagnosed below as a near miss.
Both recipes stay dominated by exact enumeration in cost at every k: T75 costs 1.7-2.1x and F2L3 about 40x.

## Method

**Models per rung k.** For each zoo seed s = 0-4 there is the clean model plus a frozen noisy panel. The panel uses
the ARC protocol: absolute σ = 0.05 on all 32 weights, noise seeds 20, 21, 22, 23 × 256 copies, which is 1024 copies
per seed. This is `understand_m43/zoo43.noisy_copies`, the E37 fresh-seed convention. **The same noise draws are added
at every k** (common random numbers), so rungs differ only in the base weights. Truth is exact enumeration
(`arc_m43_repro/exact.py`) for every model.

**Recipes and the copies they are scored on.**
- T75 is scored on all 1024 copies per seed. It runs from `est75_e37.py`, a pinned copy of `understand_m43/est75.py` at
  commit d406c17, the version that produced E37. Another lane edits `est75.py` during this work (T85/T112). The pinned
  copy gives bit-identical estimates and op counts to the live `RECIPE` on 140 models (k = 30/27/24/21 × 5 seeds × 7
  models).
- F2L3 and the linear-region counts are scored on the first 256 copies per seed at k = 30 and on the first 128 per seed
  for k ≤ 28 (noise seed 20). The smaller panel is because of CPU limits (about 0.4 s per model). It runs from
  `review/understand_ladder/fibre_strict.py` (R4's strict recount), with R4's one-line cache fix from `fibre_fix.py`
  patched in at import (`lt.py`), and LOOK = 3.

**Ops.**
- T75 uses est75's own strict charge, the T74/R3 convention. R3's automatic recount found it fair to within 1.04×.
- F2L3 uses R4's strict count, n + x.
- Sampling costs 72 ops per sample (ARC). MSE ratio = MSE / (mean p(1−p) / (ops/72)).

**Pass rule (ladder protocol, E37/E38 aggregate convention).** A recipe passes rung k if, on every zoo seed, both
conditions hold:
- (a) the panel MSE ratio is < 1;
- (b) the clean model's single-model ratio is < 1.

A bootstrap 95% CI over copies is reported for each ratio, along with the share of individual copies whose own ratio
is ≥ 1. That share is reported for information only: a single copy can exceed 1 while the recipe still beats sampling
on average.

**Mechanism tracking** (`mech.py`, all k, clean models). Exact enumeration is used here only to describe the models;
no estimator uses it. Quantities tracked:
- the step-2 switch ratios r_k = x1/x0 at which each neuron switches, per chart sign(x0);
- which neurons store x0 at step 1;
- the error mass broken down by the number of active step-3 neurons, including the dead (all-off) share;
- the share of error touching a true boundary;
- the number of step-3 gates cutting a step-2 piece.

**Diagnostics** (development data only: noise seed 0, never the 20-23 panel).
- `diag_t75.py` attributes T75's error to the step-2 vs step-3 freezes.
- `diag_neuron.py` attributes the step-3 freeze error to individual neurons.

Files:
- [lt.py](lt.py): loaders, panel, recipe wrappers.
- [run_rung.py](run_rung.py): scores one rung and writes `rung_k{k}.json`.
- [verdict.py](verdict.py): pass/fail.
- [ladder.sh](ladder.sh) / [ladder2.sh](ladder2.sh): driver (stop at first failure) and `ladder.log`.
- [report.py](report.py): tables.
- [mech.py](mech.py): `mech_k*.json`, `mech.log`.
- [clean_scan.py](clean_scan.py): `clean_scan.json/.log`, a clean-model preview over all k.
- [diag_t75.py](diag_t75.py), [diag_neuron.py](diag_neuron.py).
- [est75_e37.py](est75_e37.py): the pinned T75.

## Status

Done (2026-10-17). The ladder ran to the bottom of the zoo's training axis (k = 20) with no failure. k = 29 is Codex's
T90 and was not redone. The container restart hit after the driver had already finished, and every rung JSON is
complete: 5 seeds, 1024 T75 copies and 128/256 F2L3 copies each.

## Results

Commands (from this folder, one core):
- `ulimit -v 2500000; OMP_NUM_THREADS=1 python run_rung.py 30 256`, then `python run_rung.py K 128` for K = 28 … 20
  (`ladder2.sh` does this in order and stops at the first failing rung). Each rung takes about 20-35 CPU-min.
- `python verdict.py K` gives pass/fail.
- `python report.py` gives the tables below.
- `python mech.py 30 29 … 20` and `python emergence.py` track the mechanism.

### k-ladder table

How to read the columns:
- MSE ratio = panel MSE / sampling MSE at equal ops, per zoo seed 0 / 1 / 2 / 3 / 4.
- Exact ops = exact.py's own count, panel mean. T75 ops are strict, panel mean. F2L3 ops are R4's strict count.
- Regions = linear regions, i.e. distinct (sign x0, g2, g3) patterns. The table gives the mean over the clean seeds,
  with the noisy-subset mean in parentheses.
- Both recipes also pass the clean-model criterion at every k. The largest clean ratios are T75 0.029 (k = 30,
  seed 4) and F2L3 0.18 (k = 23, seed 4).

| k (n_seqs=2^k) | clean acc, seeds 0-4 | regions clean (panel mean) | panel acc | exact ops | T75 ops | **T75 panel MSE ratio, seeds 0-4** | T75 worst CI hi | F2L3 ops | **F2L3 ratio, seeds 0-4** | T75 / F2L3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 0.985 / 0.893 / 0.759 / 0.904 / 0.945 | 66 (66) | 0.739 | 17.1k | 29.0k | 5.6e-05 / 0.054 / 0.0033 / 0.0069 / 0.249 | 0.296 | 755k | 0.00016 / 0.067 / 5.4e-26 / 0.00075 / 0.0007 | pass / pass |
| 28 | 0.983 / 0.890 / 0.758 / 0.902 / 0.939 | 66 (66) | 0.754 | 17.1k | 29.2k | 5.3e-05 / 0.040 / 0.0029 / 0.0052 / 0.345 | 0.408 | 751k | 8e-05 / 0.062 / 2.6e-25 / 0.00016 / 0.00047 | pass / pass |
| 27 | 0.972 / 0.888 / 0.758 / 0.900 / 0.912 | 64 (66) | 0.767 | 17.2k | 29.8k | 4.8e-05 / 0.026 / 0.0076 / 0.0011 / 0.429 | 0.494 | 746k | 0.00048 / 0.034 / 4.1e-11 / 0.00029 / 0.0004 | pass / pass |
| 26 | 0.977 / 0.864 / 0.760 / 0.895 / 0.898 | 66 (66) | 0.787 | 17.5k | 31.2k | 6.4e-05 / 0.028 / 0.0052 / 0.00043 / 0.337 | 0.402 | 740k | 9.8e-05 / 0.0093 / 1.4e-25 / 5.1e-05 / 0.00052 | pass / pass |
| 25 | 0.971 / 0.861 / 0.758 / 0.899 / 0.920 | 66 (67) | 0.791 | 17.5k | 31.5k | 4.6e-05 / 0.012 / 0.0049 / 0.00053 / 0.425 | 0.498 | 740k | 9.3e-05 / 0.0097 / 4.5e-26 / 2.2e-05 / 0.00046 | pass / pass |
| 24 | 0.968 / 0.845 / 0.757 / 0.901 / 0.924 | 66 (67) | 0.782 | 17.6k | 31.5k | 4.6e-05 / 0.017 / 0.001 / 0.00041 / 0.588 | 0.661 | 747k | 2.2e-07 / 0.0066 / 7.4e-20 / 5.3e-05 / 0.027 | pass / pass |
| 23 | 0.968 / 0.861 / 0.753 / 0.898 / 0.907 | 66 (66) | 0.795 | 17.7k | 31.4k | 2.3e-05 / 0.0067 / 5.1e-05 / 0.00018 / 0.278 | 0.319 | 754k | 0.00019 / 0.00028 / 1.3e-25 / 4.7e-05 / 0.047 | pass / pass |
| 22 | 0.955 / 0.845 / 0.747 / 0.893 / 0.785 | 64 (64) | 0.802 | 17.1k | 32.3k | 6.1e-06 / 0.0059 / 1.1e-05 / 0.00014 / 0.0055 | 0.0068 | 698k | 9.8e-08 / 2.3e-25 / 2.9e-25 / 0.00021 / 4.1e-05 | pass / pass |
| 21 | 0.922 / 0.815 / 0.739 / 0.857 / 0.767 | 63 (62) | 0.792 | 16.8k | 32.7k | 2.4e-06 / 0.0037 / 1.3e-08 / 7.3e-05 / 0.00023 | 0.0045 | 681k | 3.6e-25 / 1.3e-24 / 5.9e-26 / 2.2e-09 / 0.016 | pass / pass |
| 20 | 0.805 / 0.682 / 0.625 / 0.570 / 0.581 | 59 (58) | 0.644 | 14.6k | 30.8k | 0.00032 / 0.00039 / 7.6e-06 / 0.00011 / 5.6e-06 | 0.00045 | 583k | 6.8e-26 / 3.5e-25 / 4.6e-05 / 7.1e-27 / 3.4e-26 | pass / pass |

Mechanism (clean models, `mech.py`): min over neurons of |r - 1| for the step-2 switch ratio r = x1/x0, chart x0>0 / x0<0, seeds 0-4; dead-cell error share

| k | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | dead (all-off) error share, seeds 0-4 |
|---|---|---|---|---|---|---|
| 30 | 0.19 / 0.19 | 0.01 / 0.23 | 0.20 / 0.04 | 0.00 / 0.01 | 0.04 / 0.05 | 11% / 0% / 66% / 18% / 10% |
| 29 | 0.20 / 0.20 | 0.01 / 0.23 | 0.19 / 0.04 | 0.00 / 0.01 | 0.05 / 0.03 | 11% / 0% / 66% / 17% / 13% |
| 28 | 0.21 / 0.21 | 0.00 / 0.23 | 0.22 / 0.04 | 0.01 / 0.01 | 0.05 / 0.02 | 11% / 0% / 65% / 16% / 13% |
| 27 | 0.20 / 0.21 | 0.03 / 0.26 | 0.18 / 0.06 | 0.00 / 0.02 | 0.06 / 0.00 | 9% / 0% / 65% / 16% / 6% |
| 26 | 0.21 / 0.23 | 0.13 / 0.18 | 0.15 / 0.05 | 0.01 / 0.01 | 0.06 / 0.00 | 13% / 0% / 64% / 14% / 23% |
| 25 | 0.24 / 0.22 | 0.11 / 0.14 | 0.21 / 0.03 | 0.00 / 0.01 | 0.07 / 0.00 | 11% / 0% / 66% / 16% / 21% |
| 24 | 0.24 / 0.23 | 0.11 / 0.10 | 0.23 / 0.06 | 0.01 / 0.03 | 0.03 / 0.11 | 12% / 0% / 63% / 16% / 2% |
| 23 | 0.27 / 0.26 | 0.14 / 0.17 | 0.27 / 0.05 | 0.04 / 0.02 | 0.08 / 0.21 | 10% / 0% / 63% / 16% / 2% |
| 22 | 0.30 / 0.30 | 0.13 / 0.29 | 0.32 / 0.09 | 0.09 / 0.08 | 0.10 / 0.27 | 9% / 0% / 59% / 14% / 1% |
| 21 | 0.33 / 0.40 | 0.09 / 0.43 | 0.36 / 0.17 | 0.03 / 0.41 | 0.66 / 0.26 | 4% / 0% / 48% / 6% / 1% |
| 20 | 0.62 / 0.63 | 0.05 / 0.54 | 0.36 / 0.23 | 1.00 / 0.45 | 0.07 / 0.08 | 3% / 0% / 15% / 28% / 1% |


More detail, from `rung_k*.json`:
- **Share of individual copies whose own ratio is ≥ 1.** For T75 this happens only on seeds 1 and 4. On seed 4 the
  share is 8% / 10% / 12% / 9% / 11% / **18%** / 8% / 0 / 0 / 0 at k = 30, 28, … , 20. For F2L3 it is 0 on every
  panel.
- **T75's worst single copy** is 6.8 pp at k = 30, 8.1 pp at k = 28 and 7.2 pp at k = 24. From k = 22 down it is
  ≤ 0.7 pp.
- **T75 cells per model** range from 103 to 175 and do not grow as the model is less trained.

### What changes in the mechanism as training decreases (plain words; `mech.log`, `emergence.log`)

- **Weights grow with training; the overall layout does not change much.**
  - From k = 20 to k = 30, |Wi| grows 3-50x, and ||Wo|| grows 3-30x.
  - The number of linear regions stays at 52-74 for every seed at k ≥ 21. At k = 20 it is 36-69, lower because the
    models are smaller and simpler.
  - At k ≥ 21 most of the error still touches a true boundary (82-100%; the exceptions are seed 2 at k = 21, 26 and 27, at
    46-64%).
  - About 2 step-3 gates cut each step-2 piece at every k (1.6-2.1).
  - So the four ingredients that T75 relies on are there from about k = 21. They do not appear late.
- **The step-2 "x1 vs x0" switch is built early and then sharpened.**
  - Seed 0 has its mirror two-pair storage (neurons 0, 1 vs 2, 3) already at k = 20. Its switch band brackets
    x1 = x0 from the start, but it is wide: 0.38-1.92 at k = 20, 0.70-1.40 at k = 22 and 0.73-1.31 at k = 24. It then
    narrows slowly to 0.81-1.24 at k = 30. Accuracy rises from 0.80 to 0.985 as the band narrows.
  - Seed 1 has a switch at 1.05 in the x0 > 0 chart already at k = 20. It moves to 0.97-1.00 only at k ≥ 27, together
    with a jump in |Wi|max from 3.7 to 7.4 between k = 26 and 27.
  - Seed 2 gets its x0 < 0 switch near 1 at k = 22 (0.91). It is 0.95 from k = 23 on.
  - **Seed 3 at k = 20 forgets x0 entirely when x0 > 0.** All four Wi there are ≤ 0, so no neuron stores x0, and the
    accuracy is 0.57. At k = 21 neuron 1 starts storing x0 > 0, and the switch at 1.03 appears (accuracy 0.86). The
    x0 < 0 switch at 0.98 appears at k = 23.
  - Seed 4 shows the most structure forming during training.
    - At k = 20-21 x0 > 0 is stored by a single neuron, neuron 1.
    - At k = 22 neurons 0 and 1 store it, and neuron 0's input weight starts to explode: Wi0 = 2.9 → 8.2 → 15 → 18 →
      45 at k = 22, 23, 24, 25, 30.
    - The sharp doubled x1 = x0 switch in the x0 < 0 chart (r = 1.00/1.08) forms at **k = 25**. At k = 24 it sits at
      1.11.
    - At k = 22 the error is mostly 1-active cells (66%). At k ≥ 23 it is 3-active cells.
- **So the structure the recipes rely on (a near-x1 = x0 switch, errors at true boundaries, few cutting gates) is
  present by k ≈ 21-23 on every seed. Training mostly sharpens it and scales up the weights.** The exceptions are
  seed 3 at k = 20, where x0 is lost when x0 > 0, and seed 4, whose large-input neuron and doubled switch form at
  k = 22-25.

### The near miss: T75 on seed 4 at k = 23-30 (diagnosis on dev noise seed 0, never the test panel)

Commands: `python diag_t75.py K 4 64 0`, `python diag_neuron.py K 4 64 0`, `python diag_rescale.py 24 4 0 1,0.2,0.05,5`
(→ `diag_seed4.log`, `diag_rescale.log`).

| k | T75 frozen | all step-2 gates resolved (step-3 frozen) | all step-3 gates resolved (step-2 frozen) | T75 + step-3 neuron 2 always resolved | margin-aware step 3 (RECIPE85, live est75) |
|---|---|---|---|---|---|
| 30 | 0.50 | 0.64 | 0.028 | 0.037 | 0.044 |
| 27 | 0.79 | 1.06 | 0.0017 | 0.011 | 0.0026 |
| 24 | 0.49 | 0.68 | 0.0017 | 0.010 | 0.0026 |
| 22 | 0.0047 | 0.0033 | 0.0003 | 0.0043 | 0.0005 |

(64 dev copies each. These dev MSE ratios are noisier than the 1024-copy test panel. On dev, k = 27 is the worst at
0.79; on the test panel k = 24 is the worst at 0.59.)

- **What fails is one frozen gate:** the step-3 gate of neuron 2. In seed 4, neuron 2 stores |x0| when x0 < 0, and
  its readout column is (1.1, 2.8, −3.2) at k = 24. Resolving only that gate removes 95-98% of the error. Resolving
  neurons 0, 1 or 3 instead removes nothing. Freezing at step 2 is harmless.
- **Why it is frozen.** Two causes contribute.
  - The impact threshold compares neuron 2's possible logit shift with the summed impacts of the surely-on neurons,
    and part of that sum is neuron 0's large weights. A function-preserving rescaling of neuron 0 by 0.2 halves the
    ratio (0.49 → 0.22). Shrinking it further gives the same 0.22. Rescaling neuron 2 changes nothing.
  - The rest is consistent with logit cancellation among the surely-on neurons 1 and 3, whose readout columns are
    nearly opposite: (0.3, 2.9, −2.6) and (−0.4, −3.8, 3.4). This is R3's explanation. A summed-impact scale then
    overstates the real margin. The margin-aware rule, which compares the shift with the actual frozen logit margin,
    removes it (0.0026).
- **Why it appears only at k ≥ 23 and is gone at k ≤ 22.** At k ≤ 22 seed 4 has no large-input neuron 0 and no sharp
  x0 < 0 switch at r = 1. Its readout columns are not in the cancelling (1, 3) arrangement: at k = 22 they are
  (−1.2, 2.2, −1.2) and (1.4, −3.2, 1.8). With no such cancellation, frozen neuron 2 costs almost nothing (0.0047).
  The error family is created by training, during the same steps (k = 23-25) that build seed 4's final mechanism.
  It then partly relaxes at k = 26-30 (0.25-0.43 on the panel).

### Exhaustion checklist (ladder protocol rule 3)

No rung failed, so the protocol's "fix and re-run" branch was never triggered.

| item | status |
|---|---|
| Diagnose the closest call (T75, seed 4, k = 24) | done: one frozen step-3 gate, neuron 2. Mechanism above |
| Candidate fix: margin-aware step 3 (T85 RECIPE85) | dev-only check: 0.49 → 0.0026 at k = 24 and 0.79 → 0.0026 at k = 27, at about 1.7× T75's ops. **Not adopted.** The frozen recipe passes, and changing it would need a full re-run of every earlier rung on fresh noise. That is left to a backlog row |
| Re-check k = 30 / 29 | k = 30 re-run here; it reproduces E37 exactly. k = 29 is Codex's T90 (not on this branch) |
| F2L3 near misses | none: worst 0.067 (seed 1, k = 30) |
| Panels with fresh noise seeds other than 20-23 | not done. The test noise is the E37 panel, reused at every k |

## Compute accounting

- **Per model (strict ops):**
  - T75: 29-33k ops (panel means per rung).
  - exact.py: 15-18k ops on its own count. R3's strict recount is about 1.2x higher, so T75 is about 1.6-1.8x exact
    on a like-for-like count.
  - F2L3: 0.58-0.76M ops, about 40x exact.
  - Sampling: 72 ops per sample.
- **The recipes stay deductive** and make no random calls. The only randomness is the noise draws that build the
  panels.
- **Totals for this lane:**
  - 10 rungs × 5 seeds × 1025 models = 51,250 exact + T75 evaluations.
  - 6,410 F2L3 evaluations and 6,410 region counts.
  - About 4.5 CPU-hours on one core, at a machine load of about 10.
  - Diagnostics: about 15 CPU-min on dev noise seed 0.

## Claims

- [measured] Both frozen recipes pass every training rung k = 28 … 20 (and k = 30). Here "pass" means aggregate panel
  MSE ratio < 1 and clean ratio < 1 on all 5 seeds, with exact truth on 1024 (T75) or 128 (F2L3) ARC copies per seed.
  The worst T75 seed ratio is 0.59 [CI up to 0.66] (seed 4, k = 24). The worst F2L3 ratio is 0.067 (seed 1, k = 30).
- [measured] The pass margin on the training axis is not monotone. T75's worst seed (4) runs 0.25 → 0.35 → 0.43 →
  0.34 → 0.43 → **0.59** → 0.28 → 0.006 → 0.0002 → 6e-6 from k = 30 down to k = 20. Less-trained models are
  **easier** for both recipes below k = 23.
- [measured] Seed 4's near miss is carried by one frozen step-3 gate (neuron 2). Neuron 0's scale accounts for about
  half of it (rescaling test). A margin-aware step-3 rule removes it on dev data.
- [measured] The x1-vs-x0 step-2 switch exists from k = 20 on seeds 0 and 1, from k = 21-23 on seeds 2-3, and in its
  final sharp form on seed 4 from k = 25. Training mainly sharpens it: seed 0's band narrows from 0.38-1.92 to
  0.81-1.24.
- [inferred] The rest of seed 4's error comes from logit cancellation between surely-on neurons 1 and 3. This is
  consistent with the opposite readout columns and with the margin rule fixing it, but I did not isolate it with its
  own test.
- [inferred] Low-k rungs are easier partly for a reason unrelated to understanding. The noise σ = 0.05 is absolute,
  while the weights shrink 3-50x, so low-k copies are relatively much noisier and their accuracies sit nearer 0.6-0.8
  (larger p(1−p)). At the same time T75 resolves nearly every cutting gate there.
- [speculative] The hard part of the training axis for these recipes is the window where a seed builds its final
  readout (k ≈ 23-25 for seed 4). This is when large weights and cancelling readout pairs appear. Larger models may
  have such windows at different k.

## Alternatives not ruled out

- **Relative vs absolute noise.** With absolute σ = 0.05 the low-k panels are not comparable in difficulty to the k =
  30 panel. A relative-σ panel (σ ∝ RMS(W)) would test whether the low-k "easy" verdict survives. Not done.
- **Panel reuse.** The test noise seeds 20-23 are E37's. They were used for T75's original evaluation, though not for
  its development. Fresh seeds at k = 24 (the tightest rung) were not run.
- **F2L3 panel size.** F2L3 used 128 copies per seed, not 1024, because of CPU limits. Its margins are 10-1000x, so
  this is unlikely to matter. The bootstrap CIs are in the JSONs.
- **k = 29 not reproduced here.** I rely on T90's report, and its folder is not on this branch.
- **The CI is over noisy copies of one base model per seed.** It says nothing about other training runs.

## Failure modes

- T75: a frozen gate whose logit effect is masked by a large or cancelling surely-on scale (seed 4, k = 23-30). The
  error is one-sided per copy, up to 8.1 pp.
- F2L3: misses wedges detached from both fibre ends (R4). On the clean seed-4 models it misses 0.06-0.11 pp at
  k = 21-24, still ratio ≤ 0.18.
- Neither recipe is cheaper than exact enumeration at any k on M_4,3.
