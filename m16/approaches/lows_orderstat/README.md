# T341: do the late margins read the low inputs through their order statistics (running max)?

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T338](../mech_closure/README.md) · T327 (claude/t286-cheap-frontier, closure_floor/)

## Question
T338 [speculative]: the late margins read the lows through the order-statistic neurons (running max, leave-one-out
max), not through any single low. Closure c's flat μ(v4) background then understates the lows' running max, and T327's
±σ spread works because it restores a realistic one.

The spread lowered the oracle floor from .481 to .343 pp, and to .255 with the reverse pattern. If the running max is
what matters, closures that restore the true running max should do at least as well.

## Pre-registration (2026-10-18, before any run)
**Notation.** The free positions F are the 7 positions outside the triple. The free values are x_F. v4 = max x_F,
v5 = the second-largest of x_F. μ(u) = E[N(0,1) | < u]; σ(u) = its sd.

**Oracle closures** (all built from each input x's own values; v1–v3 exact at their positions):
- (c) **closure c**: v4 at the latest free position, the other 6 at μ(v4). Baseline.
- (i) **v5 at its true position**. v4 is at the latest free position; if v5 is there, v4 moves to the next-latest. The
  other 5 are at μ(v5).
- (ii) **v5 at a fixed rule position**: v4 at the latest free position, v5 at the second-latest, the other 5 at
  μ(v5). There is no tree split: positions are fixed per triple, and the extra value is one more node of a 1-D law.
- (iii) **true running max**. Walk F in time order. A free position whose value is a new record (≥ all earlier free
  values) keeps its true value. A non-record position gets μ(running max so far).
  - This keeps the whole running-max path of the lows, v4 at its true position included.
  - It needs the record positions, which are random (about H_7 ≈ 2.6 records), so it splits the tree. It is the
    information test, not a design.
- (iv) **running-max profile rule** (realisable, no split).
  - The k-th free position in time order (k = 1..6) gets r_k = E[max of k i.i.d. N | < v4]. The latest free
    position gets v4.
  - Every position is a record, and the running max follows its expected path given v4.
  - The r_k come from a model-free 1-D quadrature per v4.
- (s+) / (s−): **T327's spread**, re-implemented. v4 at the latest free position, and the other free lows at
  μ(v4) ± (−1)^j σ(v4), where j is the absolute position. It is a calibration against T327's .343 (s+) and .255 (s−).

**Protocol (T298).**
- 80 DEV2 models; x ~ N(0, I_10), N = 500k per model, seed 341000 + index.
- b_m = mean(ok_closure − ok_true).
- RMS_m b_m, mean, slope; a paired bootstrap over models (10,000) of RMS differences against (s−).

**Go (DRI):** a realisable variant, (ii) or (iv), with an oracle floor below (s−) as measured here (≈ .255) and the
upper bootstrap CI of the difference < 0. (iii) is reported for its information content.

**Reading of the outcomes:**
- (iii) ≤ (s−), and (iv) or (ii) ≈ (iii): the running max explains the spread's gain, and a cheap rule captures it.
- (iii) ≤ (s−) but (ii) and (iv) are worse: the information is real but not cheaply placeable.
- (iii) > (s−): the running-max hypothesis does not explain the spread. Some other feature of alternation matters,
  for example sign-alternating recurrent modes (T241's eigenmodes with period ~2), and the T338 reading is refuted.

**Alternatives first.**
1. The spread works through second-order effects (the variance of the lows entering a ReLU neuron), not the max. Then
   (iii) helps little and (s±) stays best.
2. The orientation asymmetry (.343 against .255) points to a parity or recency effect. The last free position's
   parity decides whether it sits above or below μ. Report (s±) by the parity of the latest free position.

## Status
**Done. NO GO, and the T338 running-max reading is REFUTED** (pre-registered reading 3). Unverified; the floors are
diagnostic sampling.

## Results
To reproduce: `python3 os_mc.py` (80 DEV2 models × 500k inputs, ~9 CPU-min), then `python3 analyze.py`. Outputs are
`results/os_all.json` and `results/summary.json`.

| closure | oracle floor RMS (pp) | mean (pp) | RMS − (s−) [95% CI] | splits the tree? |
|---|---|---|---|---|
| (c) closure c | .472 | −.151 | +.224 [+.152, +.300] | no |
| (i) v5 at its true position | .314 | +.029 | +.066 [+.010, +.120] | yes |
| (ii) v5 at the second-latest position | .483 | −.187 | +.235 [+.162, +.312] | no |
| (iii) true running max (records exact) | .359 | **+.315** | +.111 [+.055, +.168] | yes |
| (iv) expected running-max profile | .602 | +.352 | +.354 [+.257, +.452] | no |
| (s+) spread μ + (−1)^j σ | .333 | −.046 | +.085 [+.028, +.144] | no |
| **(s−) spread μ − (−1)^j σ** | **.248** | −.019 | — | no |

- **Calibration.** The re-implemented spread reproduces T327: .333 against .343 for (s+), .248 against .255 for (s−).
  Closure c reproduces .472 (T338 .473, T327 .481).
- **No go.** Neither realisable variant beats (s−). Both are worse even than closure c: v5 at a fixed position, .483,
  and the expected running-max profile, .602.
- **The running-max hypothesis does not explain the spread** (pre-registered reading 3).
  - Imposing the true running max (iii) reaches .359, worse than the zero-cost spread (.248), with a large positive
    bias (+.31 pp). It makes the model look more accurate than it is.
  - So restoring a realistic running max is not what the spread does. The T338 [speculative] reading is refuted.
- **v5 carries real information, but only at its true position.** (i) at .314 beats closure c by .16 pp, but moving v5
  to a fixed slot (ii) removes the whole gain. Where the second-largest low sits matters, so it is not a cheap
  order-statistic summary. It is still worse than (s−).
- **Parity check (alternative 2).** The mean bias of (s−) differs by the parity of the latest free position: +.058
  when even, −.044 when odd. (s+) shows no parity effect (−.048 / −.045). There is a parity component, but it is small
  against the floors.
- **What remains [speculative].** The spread's gain is not explained by any order statistic of the lows tested here.
  Its orientation matters (.248 against .333), and the per-parity bias is asymmetric. Both point to a sign-alternating
  response of the recurrence to the lows: T241's eigenmodes with a period of about 2. The response depends on the
  pattern of deviations across time, not on their max.
  - Cheapest test (not run): project the T241 period-2 mode onto each retrain's W. Check whether the (s−) and (s+)
    gains across models correlate with that mode's magnitude.

## Compute accounting
`os_mc.py`: 80 models × 500k inputs × 8 forward passes (truth + 7 closures), plus a model-free 6 × 851 table for (iv).
About 9 CPU-min, single thread, `ulimit -v 2500000`.

## Claims
- [measured] On 80 DEV2 models, the oracle closure floors are: closure c .472, v5-true .314, v5-fixed .483, true
  running max .359, expected running-max profile .602, spread (s+) .333, spread (s−) .248.
- [measured] No order-statistic closure beats the zero-cost reverse spread. The running-max explanation of the
  spread's gain is refuted for this closure family.
- [inferred] The position of v5 carries information that a fixed placement loses.

## Alternatives not ruled out
- Other order-statistic placements (v5 at the earliest free position, or at the parity-matched position) were not
  tried.
- (iii)'s non-record values sit at μ(running max). A spread around that running max was not tried.
- The eigenmode reading is untested.
