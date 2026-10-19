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
**Done.** (a) All three of T402's numbers reproduce (within 2 s.e.). (b) Pre-registered verdict: **targeting matters**.
None of the 40 random edits succeeds, and none of the 10 sign edits does. T402's ratio is 3.2x below the best random ratio.
Unverified (one agent); I do not confirm my own claim.

## Results
Reproduce: `python check.py models; ./run.sh` (jobs in `results/jobs.txt`, log `results/run.log`), then
`python summarize.py` (-> `results/summary.json`, per-model table in `results/summary.txt`).

**Integrity and sampler checks (pre-registered, all pass) [measured].**
- Base accuracy is .953121 (1e9 inputs, s.e. ~7e-6). `m16/truth.txt` gives .953179 +- 4.7e-5, so z = 1.2.
- Sampler check, `plain 1.0 2e8 410400`. P(g23 > 1): plain .044454 +- 1.5e-5 vs slab .044428, z = 1.8.
  P(E1(1.0)): plain 5.43e-6 +- 1.6e-7 vs slab 5.20e-6 +- 3.4e-7, z = 0.6.
- P(g23 > 2) is 5.776e-4. Base P(E1(2.0)) is 6.81e-9; T364's truth is 6.81e-9, also matched.

**(a) Re-measurement of T402 (3e8 paired slab draws; 1e9 paired inputs) [measured].**

| number | mine | T402 | z | agree (2 s.e.) |
|---|---|---|---|---|
| E1(2.0) ratio, edit / base | **.0518 +- .0037** (6.81e-9 -> 3.52e-10, 19.3x rarer) | .059 +- .004 | -1.3 | yes |
| accuracy change | **-.0113 +- .0002 pp** | -.0116 +- .0002 pp | +0.9 | yes |
| largest-input picks at g23 > 2 | **+9.33 +- .02%** | +9.3% | +1.2 | yes |

**(b) Control: same 12 weights, same L2 norm .003, 3e7 paired slab draws and 5e7 paired inputs each [measured].**

| set | success (ratio <= .1 and d acc >= -.02 pp) | ratio <= .1 | ratio min / q25 / median / q75 / max | d acc (pp) min / median / max | pick shift min / median / max |
|---|---|---|---|---|---|
| T402 edit | yes | yes | .052 | -.011 | +9.3% |
| 40 random directions | **0/40** | 0/40 | .165 / .46 / .80 / 1.56 / 2.44 | -.27 / -.068 / +.024 | -2.6% / +3.2% / +21.8% |
| first 20 random | 0/20 | 0/20 | .165 / .46 / .86 / 1.35 / 2.44 | -.25 / -.068 / +.024 | -2.6% / +3.7% / +13.6% |
| 10 sign flips of T402 | 0/10 | 0/10 | .49 / .88 / .93 / 1.85 / 2.28 | -.077 / -.036 / -.008 | -3.6% / +1.1% / +9.2% |

- The best random edit is rand_17: ratio .165 +- .017, d acc -.011 pp. That is a 6x drop, 3.2x short of T402's ratio.
- 7/40 random edits reach ratio < .3. Of those, only rand_3 (.256, -.002 pp) and rand_17 cost <= .02 pp.
- 15/40 random edits make E1(2.0) MORE frequent; the worst makes it 2.4x more frequent.
- Random edits hurt accuracy more than T402's edit: median -.068 pp vs -.011 pp.

Verdict by the pre-registered line: **targeting matters**. 0/40 random edits succeed (the line allows <= 2), and T402's
.052 is below the smallest random ratio, .165.

**Pick shift [measured].** Random edits also shift picks towards the largest input, by a median of +3.2%; 3/40 shift
more than T402's +9.3%. So the shift is partly a generic response to perturbing these 12 weights. T402's shift is large
for its accuracy cost but not exceptional among random edits.

## Compute accounting
- Model forward passes: 6.6e9 (one input on one model = 1 FPE): slab 2.1e9, accuracy 4.5e9, sampler check ~1e7.
- Sampler draws (input generation): 3e8 slab draws, 1e9 + 5e7 plain inputs, 2e8 for the sampler check.
- Wall time: about 2.5 h on 2 single-threaded processes, each under `ulimit -v 2500000`.
- F bars (Directive 8): not applicable. This is a verification of an edit, not an estimator of accuracy.
  For context, the plain-MC cost of seeing E1(2.0): at p = 6.8e-9, a 10x drop needs about 30 events on the edited model,
  i.e. about 9e10 plain inputs. The slab sampler needed 3e8 draws (a 1/p_gap = 1730x saving).
  T402's design cost of ~2e9 FPE is restated, not re-scored.

## Claims
- [measured] T402's edit makes E1(2.0) 19.3x rarer (ratio .052 +- .004) at -.0113 +- .0002 pp accuracy, and raises
  largest-input picks at g23 > 2 by 9.33%. All three agree with T402 within 2 s.e., using independent code, sampler,
  seeds and statistics.
- [measured] Random 12-weight edits of the same norm reach no 10x drop: the best is 6x, the median 1.25x, and 15/40 go
  up. Within the same accuracy budget the best is 6x (rand_17). Targeting matters by the pre-registered line.
- [measured] Keeping T402's per-weight magnitudes but randomising the signs does no better: the best is 2x, the median
  ~1.08x. The effect is in the direction, not just in which weights or how much they move.
- [inferred] P(E1(2.0)) is very sensitive to these 12 weights: random edits of norm .003 move it by up to ~2.5x
  either way. This fits T372/T402's large log-gradient (|d log P / dW| ~ 950). A gain of 10-20x needs the direction
  to be aligned with that gradient and, at the same time, to stay off the accuracy curvature; T402's design does both.

## Alternatives not ruled out
- Shared inputs with T402: the same base and edit weight file. Base integrity is checked only through accuracy vs
  truth.txt, because alg_zoo is not installed here.
- Only isotropic random directions in S, and sign flips. Not tried:
  - random directions scaled per weight (relative to |w|);
  - random sets of 12 weights other than S;
  - larger norms.
  A random search over many directions, or the best of N random edits, could find a 10x edit; the control tests a
  random edit, not a search.
- The control's own s.e. near the .1 line is ~.02. No control is near .1 (the minimum is .165 +- .017, 3.8 s.e. above it).
- The accuracy-cost line (.02 pp) is T402's design limit. At the go line's .05 pp the result is the same: 0/40 reach
  a ratio of .1 at all.

## Failure modes
- The slab sampler is exact only for events defined on g23 > t0. The grid truncation [-8, 5] for v3 drops mass
  < 1e-20.
- Paired counts at ratio ~.05 rest on ~1000 edit-side weighted events. Event counts, not draws, set the s.e.
- Precision: float64 forward. T402's float64 path differs in operation order, so a tie-level disagreement is
  possible but not visible at these s.e.
