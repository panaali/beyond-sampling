# Results: AlgZoo legibility versus understanding

↑ [Project README](README.md) · [STATUS](STATUS.md) · [INDEX](INDEX.md) · [PREREGISTRATION](PREREGISTRATION.md)

The AlgZoo addenda were committed before any AlgZoo neuron was inspected. This file retains the historical E4 results; consult [INDEX](INDEX.md) for current estimator results.

## AlgZoo, legibility vs understanding (pre-registered E4)


Each model got a blinded investigator agent with code access. It wrote a label and a formula f_i(x) per
neuron (no access to the weights in the formula, at most 5 constants per neuron). Scored on held-out data
([`algzoo/score.py`](algzoo/score.py)).

| model | params | model acc | neurons legible (A/B, R^2 >= 0.8) | labels rebuilt -> accuracy | gap recovered | verdict |
|---|---|---|---|---|---|---|
| M_4,3 (positive control) | 32 | 98.4% | 4/4 | 98.5% | 100% | legible: pass, mechanism: pass |
| **M_16,10 (Hilton's challenge)** | **432** | 95.4% | **14/16** | **54.1%** | **52%** | **legible: pass, mechanism: FAIL** |
| M_32,10 | 1,376 | 99.4% | stopped (draft in [`algzoo/labels_32_10.py`](algzoo/labels_32_10.py), unscored) | | | |

The protocol works when understanding is achievable (positive control). On the 432-parameter model the
neurons are individually legible, but the labels together lose about half the model's skill. **Caveat:**
this is largely driven by the 5-constants-per-neuron limit. A 216-state lookup of per-neuron linear maps on
the same features rebuilds 98%, and replacing even one neuron by its label drops accuracy to 0.79-0.95
despite R^2 ~0.9, because the readout amplifies small errors. It shows "short labels lose skill", not "no
compact description exists". The size ladder has one data point beyond the control (M_32,10 was not scored).

Deeper first-principles analysis of M_16,10 ([`m16/FINDINGS.md`](m16/FINDINGS.md)):
- In the most common gate pattern, the recurrence has damped rotating modes; the slowest mode's phase
  encodes how long ago the max arrived (measured -42.7 deg/position vs 42.8 predicted). **Corrected:** the
  "harmonic series / Fourier memory" reading does not survive robustness checks. That pattern covers only
  10-16% of states, other common patterns give non-harmonic periods, and the damping differs across modes. 13
  of the 16 neurons were ablated, in groups, with mean-ablation (n2, n5 and n9 were not). The "n3/n13 are
  harmonics" attribution used a non-orthogonal decomposition and is weak.
- Errors grouped by what the model picks: blind region (2nd largest <= 0) 0.82%, picks the 3rd 2.14%, picks
  the max 0.52% (when x9 is below both), plus a ~0.27% remainder picking rank >= 4. (The 99.96% accuracy on
  rank-canonical inputs is close to true by construction: canonical inputs never fall in the blind region.)
- ARC's open challenge (estimating accuracy mechanistically as well as sampling) is **not met**. After about 30
  attempts, every deductive method loses to sampling (8x to >300x). The best verified estimator is sampling-based
  ("srb", ~1.6-2.4x better than plain sampling on all 18 leaderboard models). Calculation does beat sampling by up
  to ~500x on short sequences (T=3-4) and collapses around T=5-8. See [`STATUS.md`](STATUS.md) and [`INDEX.md`](INDEX.md) section 4b.
