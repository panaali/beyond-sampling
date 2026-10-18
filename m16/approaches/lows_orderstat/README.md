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
Pre-registered; not run.
