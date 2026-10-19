# T410: independent check of T402's targeted repair, plus the random 12-weight control

↑ [T402 repair (branch claude/t372-rare-gradient @ f110e852, m16/approaches/rare_gradient/repair/)](../../../TASKS.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md)

## Question
T402 reports that a 12-weight edit of M_16,10 (L2 norm .003), chosen from the weight-gradient of the rare far-gap
error E1(2.0) (rank-3 pick with g23 = v2 - v3 > 2) and the accuracy curvature, makes E1(2.0) 16.8x rarer (ratio
.059 +- .004) at an accuracy cost of -.0116 +- .0002 pp, while picks of the largest input at g23 > 2 rise 9.3%.
(a) Do these three numbers reproduce with a different agent, different code and different seeds?
(b) Does targeting matter: what do random edits of the same 12 weights at the same norm do?

## Method (pre-registered 2026-10-19, before any score exists)
Sampling-based (verification truths; this is not an estimator under test). Own code: `check.py`; T402's estimator,
sampler (T364 truthc) and forward are not used.

**Models (`python check.py models`; `results/models.json` holds every weight change).**
- base and T402's frozen edit from T402's `edit.json` (copied to `t402_edit.json`), sha256 of `w_edit` asserted
  (6c9ad06a...); asserted that only the 12 weights of S change and ||dW|| = .003.
  No alg_zoo install in this container, so base integrity is checked by its accuracy against `m16/truth.txt`
  (.95317945, 2e7 inputs): must agree within 3 combined s.e.
- **Random control (primary): 40 edits** (the task minimum is 20; the first 20 are also reported alone):
  z ~ N(0, I_12) on the same 12 weights S, scaled to L2 .003. numpy `default_rng(410001)`, drawn in order.
- **Sign control (secondary): 10 edits** with T402's per-weight magnitudes and random signs (T402's own sign pattern
  excluded), `default_rng(410002)`. Tests whether the gain is in the choice of weights/magnitudes or the direction.

**Samplers.**
- Slab (conditional law given g23 > t0, own derivation, v3 first): density of v3 = b is
  G(b) = 360 phi(b) Q(b + t0)^2 Phi(b)^7; b from a piecewise-constant grid proposal with exact weight G/q (unbiased
  for any grid; grid [-8, 5], 4e5 points); the two high values i.i.d. N given > b + t0, seven lows i.i.d. N given < b,
  uniform random positions. P(E) = E_q[w 1_E].
- Plain: i.i.d. N(0, I_10).
- Sampler check (base model only, run first, not a score of any edit): `plain 1.0 2e8 410400` compares plain-MC
  P(g23 > 1) and P(E1(1.0)) with the slab sampler's; must agree within 3 s.e.

**Runs (all models on the same draws in a job = common random numbers; per-1e6-draw block sums stored).**
- Slab t0 = 2.0, 3e8 draws for base + T402: seeds 410201-410209 (3e7 each, pair only) and 410210-410211 (1.5e7 each,
  all 52 models; so every control gets 3e7 paired slab draws).
- Accuracy, 1e9 inputs for base + T402: seeds 410301-410319 (5e7 each, pair only) and 410320-410321 (2.5e7 each,
  all 52 models; every control gets 5e7 paired inputs).
- Per draw: E1 = pick == position of v3; R1 = pick == position of the max; the picked rank histogram.

**Statistics.** Ratio r = sum w 1_E(edit) / sum w 1_E(base); its s.e. by delete-one-block jackknife over the 1e6-draw
blocks (T402 used a per-draw delta method). Accuracy change = mean paired difference, s.e. by batch means over blocks.
Largest-input pick shift = sum w 1_R1(edit) / sum w 1_R1(base) - 1, jackknife s.e.

**Decision lines (fixed now).**
- (a) Agreement with T402 on each of the three numbers: |mine - T402| <= 2 sqrt(se_mine^2 + se_T402^2). T402 gives no
  s.e. for the pick shift; I use se_T402 = sqrt(679k) / 7.26M (~.011%, flips assumed one-way), i.e. essentially mine.
  Disagreement on any one is reported as such and investigated (not explained away).
- (b) Control "success" for an edit: E1(2.0) ratio <= 0.1 AND accuracy change >= -.02 pp (-2e-4). Reported: the
  fraction of the 40 (and of the first 20) random edits that succeed, the distribution of their ratios (min,
  quartiles, max), their accuracy changes and pick shifts, and the same for the 10 sign edits.
  - **Targeting matters** if <= 2/40 random edits succeed AND T402's ratio is below the smallest random ratio.
  - **Targeting not needed** if >= 10/40 succeed.
  - Otherwise: partial (reported with the numbers).
- Cost: forward passes of this check reported. This is a verification, not an estimator of the accuracy, so the
  F bars (Directive 8) do not apply to its numbers; the F-relevant statement is T402's design cost in FPE vs the
  plain-MC cost of seeing E1(2.0) (restated, not re-scored).

**Alternatives to keep in mind.** Agreement could be shared error (same base weights file; same event definition):
base integrity is checked against truth.txt; the event definition is re-derived from the task text. A random-edit
success rate near zero could reflect a too-small norm rather than targeting: reported with the ratio distribution and
the sign control (same magnitudes, different direction).

## Status
Pre-registered; runs not started.

## Results
(pending)

## Compute accounting
(pending)

## Claims
(pending)

## Alternatives not ruled out
(pending)

## Failure modes
(pending)
