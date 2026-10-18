# T241: quantitative per-neuron account of M_16,10

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [FINDINGS](../../FINDINGS.md)

## Question
Can the qualitative account of the 16 neurons (running max n2, record detector n4, x9 comparators n1/6/7/8,
clocks n0/10/11/12/14/15, late-x8 n5/n9; FINDINGS.md, H-M1..H-M8) be made quantitative: a causal map, explicit
encoding formulas with held-out R^2, and a replacement model built from those formulas that reproduces the
network's answers (pre-registered targets: >= 99% agreement, accuracy within 0.1 pp, all 16 replaced)?

This is interpretability with sampling (not an ARC estimator). Every fit is scored on held-out inputs.

## Method
Exact float64 forward passes of the original checkpoint (`m16_weights.npz`, exported once from ARC's public
`.pth`, sha256 `498bd9a8...`; base accuracy on 1M fresh inputs 0.95332, truth 0.95318). `nacommon.py` holds the
forward pass, case masks and order statistics. Notation: h_t = state after reading x_{t-1}, t = 1..10.
Cases (by the true input): *x9max* (x9 is the max, 10.0%), *x9second* (10.0%), *x9below* (80.1%);
*blind* = at most one positive input (1.06%, overlaps the others).

- **Phase A, causal map** (`phaseA.py`, `phaseA_modes.py`, `reportA.py`): for each neuron j and step t,
  replace h_t[j] by 0 (zero), by its pilot-set mean (mean; pilot disjoint from eval) or by h_t[j] of another
  random input (resample), then run the remaining steps exactly. Also ablate neurons/groups at every step, or
  at the final state only. Modes: for the 5 most common final gate patterns, eigen-decompose D_g W_hh and
  resample each mode's oblique coordinates (from left eigenvectors) from a donor input with the same final
  gate pattern, at h_10 (readout) or at h_9 (then step 10 run exactly).
- Phase B, encoding: *running*.
- Phase C, replacement: *pending*.
- Phase D, residual: *pending*.

## Status
Running. Phase A done (2026-10-18).

## Results

### Phase A: causal map (N = 200,000 eval inputs, seed 241001; pilot 50,000, seed 241002)
Reproduce: `python3 phaseA.py 200000 && python3 reportA.py > results/phaseA_tables.md` (about 4 CPU-min);
`python3 phaseA_modes.py` (N = 400,000, seed 241011; about 1 CPU-min). Full 16x10 matrices for every metric
(accuracy, agreement, four case accuracies) x three ablation types are in `results/phaseA.json` and
`results/phaseA_tables.md`. Sampling s.e. of an accuracy here is about 0.05 pp; differences are paired.

**Resample-ablation of one neuron at one step: drop in accuracy (pp).** Base accuracy 0.9539.

| neuron | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 | t=9 | t=10 |
|---|---|---|---|---|---|---|---|---|---|---|
| n0 | -0.0 | 2.3 | 5.6 | 8.3 | 14.9 | 23.3 | 28.0 | 35.6 | 37.2 | 42.5 |
| n1 | 5.9 | 12.4 | 18.8 | 26.0 | 32.7 | 46.3 | 52.4 | 58.7 | 62.0 | 64.0 |
| n2 | 0.0 | 29.8 | 42.4 | 49.7 | 50.4 | 48.7 | 44.5 | 39.2 | 36.2 | 28.3 |
| n3 | 0.0 | 0.3 | 0.3 | 2.9 | 3.7 | 4.5 | 5.7 | 6.5 | 19.7 | 22.5 |
| n4 | 27.4 | 24.3 | 20.7 | 17.3 | 14.1 | 11.4 | 8.8 | 6.3 | 4.3 | 4.4 |
| n5 | -0.0 | 10.7 | 10.5 | 12.7 | 15.3 | 17.2 | 19.6 | 20.7 | 21.1 | 21.3 |
| n6 | 8.7 | 12.4 | 22.6 | 34.2 | 44.7 | 50.0 | 53.6 | 55.2 | 59.9 | 61.1 |
| n7 | 9.3 | 24.4 | 33.2 | 40.8 | 47.0 | 54.2 | 59.3 | 63.2 | 67.0 | 66.8 |
| n8 | 8.6 | 12.9 | 18.8 | 25.6 | 33.7 | 39.7 | 47.5 | 59.6 | 61.6 | 65.2 |
| n9 | 0.0 | 1.7 | 3.6 | 4.6 | 5.4 | 6.3 | 6.8 | 6.6 | 5.4 | 1.1 |
| n10 | 0.0 | 9.0 | 11.3 | 15.5 | 15.8 | 17.2 | 32.2 | 33.5 | 37.5 | 42.1 |
| n11 | 0.0 | 11.2 | 18.4 | 17.2 | 19.4 | 32.1 | 29.7 | 28.8 | 35.0 | 37.4 |
| n12 | 0.0 | 8.0 | 9.3 | 16.3 | 19.0 | 19.1 | 23.2 | 33.0 | 40.4 | 43.7 |
| n13 | -0.0 | 0.9 | 2.4 | 6.5 | 11.4 | 12.1 | 13.9 | 17.7 | 21.9 | 25.0 |
| n14 | 0.1 | 3.3 | 3.6 | 7.0 | 13.2 | 19.2 | 24.4 | 27.4 | 29.4 | 38.5 |
| n15 | -0.0 | 0.2 | 0.7 | 0.6 | 0.4 | 2.0 | 4.5 | 19.1 | 28.5 | 31.0 |

Zero- and mean-ablation give the same ordering (tables in `results/phaseA_tables.md`); zero is harsher on the
clocks, mean is milder on n2 late (19.5 vs 28.3 pp at t=10).

**Case drops at t=10 (resample, pp)** — x9max / x9second / x9below / blind:
n0 35/48/43/10 · n1 60/68/64/12 · n2 28/20/29/10 · n3 21/13/24/5 · **n4 36/7/0/1** · n5 20/12/23/4 ·
n6 57/70/61/10 · n7 66/71/66/10 · n8 57/72/65/12 · n9 0/1/1/1 · n10 37/34/44/6 · n11 25/33/40/9 ·
n12 39/42/45/9 · n13 24/12/27/9 · n14 31/32/40/5 · n15 32/17/33/6.

**Groups, final-state-only ablation (accuracy in x9max / x9second / x9below)**

| ablated | mean | resample |
|---|---|---|
| none | .934 / .952 / .957 | |
| comparators n1,6,7,8 | .840 / **.000** / .089 | .478 / .067 / .168 |
| clocks n0,10,11,12,14,15 | .125 / .433 / .231 | .158 / .181 / .195 |
| record n4 | .876 / .821 / .962 | **.570** / .887 / .955 |
| n5,n9 | .807 / .928 / .792 | .740 / .834 / .730 |
| n3,n13 | .715 / .890 / .678 | .628 / .777 / .577 |
| n2 | .692 / .823 / .759 | .653 / .749 / .664 |

The mean-ablation rows for comparators and clocks reproduce FINDINGS.md's table (.840/.000/.085 and
.127/.437/.229) with independent code and data.
All-step resample of any single neuron leaves 13–64% accuracy (n9: 64%, n3: 53%, n15: 48%, others 13–37%).

**Modes of the most common final gate pattern 64975 (29.0% of final states; acc 0.958 inside it).**
Resample of one mode's coordinates at h_10 (accuracy inside the pattern; agreement similar):

| |lambda| | period | main neurons | energy share | acc after resample h_10 | after resample h_9 |
|---|---|---|---|---|---|
| 1.010 | – (integrator) | n12, n0, n14, n11 | .455 | .905 | .780 |
| 0.973 | – (slow) | n0, n12, n11, n14 | .115 | .842 | .728 |
| 0.951 | **8.42 (clock)** | n12, n10, n14, n11 | .313 | .864 | .809 |
| 0.937 | **4.32** | n15, n13, n3 | .063 | **.491** | .532 |
| 0.928 | 2 (Nyquist) | n13, n3, n15 | .007 | .824 | .836 |
| 0.837 | 2.89 | n3, n15 | .044 | .553 | .604 |
| 0.735 | 8.37 (damped) | n14, n15, n3 | .042 | **.390** | .419 |
| 0.231 | – | n14, n8 | .006 | .807 | .935 |
| ~0 | – (input-only) | n7, n8, n6, n1 | .289 | .838 | .925 |

Patterns 63951 (8.6%) and 62927 (6.0%) behave similarly (every mode resample costs 15–70 pp). In patterns
65007 and 65519 the eigenbasis is badly conditioned: oblique mode projections have energy shares 3–120x the
state energy, so their modal ablations are not interpretable (kept in `results/phaseA_modes.json`).

## Compute accounting
Phase A: 480 single-step ablations x 200k inputs x (10−t) steps + 66 group/all-step runs; total about 1.5e8
sequence-forwards (~1e11 FLOPs), ~5 CPU-minutes single-threaded. Modes: 5 patterns x ~9 modes x 2 sites on
≤116k inputs, ~1 CPU-minute.

## Claims
- A1 [measured] No neuron is redundant in the late steps except n9: resample-ablating any other single neuron at
  t=10 costs 4–67 pp (n4: 4.4, others >= 21 pp); n9 costs <= 6.8 pp at every step.
- A2 [measured] Roles are time-local: n4 matters most at t=1 (27 pp) and fades (4 pp at t=10); n2 peaks at
  t=4–5 (50 pp); comparators and clocks rise with t; n3 and n15 matter only from t=8–9 on; n5 is ~10–21 pp
  from t=2 onward (so not only "after a near-record x8").
- A3 [measured] Case selectivity holds for the comparators (mean final ablation: x9second .000 vs x9max .840)
  and for n4 at the final step (x9max loses 36 pp, others <= 7), not for the clocks, n2, n3/n13, n5 (all cases hit).
- A4 [measured] In the dominant final pattern, the period-4.32 and damped period-8.37 modes (n3/n13/n14/n15)
  are the most damaging modes to resample (to .49 and .39) although they carry 6% and 4% of state energy; the
  8.42 clock and the integrator cost 5–9 pp.
- A5 [inferred] The single-neuron damage is mostly an off-manifold effect of incoherent states plus the
  readout's large-cancelling weights (FINDINGS §3), so ablation size measures sensitivity, not "amount of role".

## Alternatives not ruled out
- Large drops from resampling a single coordinate may reflect off-distribution states (the 16 neurons are
  strongly correlated), not the neuron's own information content. Conditional resampling (donor matched on the
  hypothesised variables) is the stronger test; it is Phase C's job.
- Oblique mode coordinates depend on the gate pattern; donor states from the same final pattern can still
  have different earlier gates.

## Failure modes
- Mode ablations are only meaningful in well-conditioned patterns (64975, 63951, 62927).

## Proposed shared-code changes
None so far.
