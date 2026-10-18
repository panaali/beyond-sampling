# T358: does variant A's rare-event win on the interior error set (E2) hold on 20 more models and rarer events?

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T350](../rare_check/README.md)

## Question
T350's one surviving rare-event result: variant A, an uncut T317 line-23 walk at T317's region transverse point,
estimates E2(t) = P(the network is wrong, g12 ≥ t and g23 ≥ t). On 6 fresh DEV3 models it beat sampling (or the
zero answer) at equal compute in 7 of 9 decided cases, at 14–21k FPE.

This task strengthens or kills that result, on 20 further models and at rarer thresholds.

## Pre-registration (2026-10-18, before any truth for these models is requested or read)
**Estimator.** It is frozen and unchanged: `../rare_check/rare.py`, `est_E2` (variant A), sha256
`e7db88752c8c1139793eb23994017a8685a36003dc03553103d54d8a9bc06b97`, called with t ∈ {.5, .75, 1.0, 1.25}. It is
deductive, and its cost is charged as executed (as in T350).

**Events.** E2(t) = {argmax ≠ p2} ∩ {g12 ≥ t} ∩ {g23 ≥ t}, for t ∈ {.5, .75, 1.0, 1.25}.

**Models.** The next 20 models in DEV3's manifest order: **dev3_s506 … dev3_s525** (manifest positions 7–26).
**Truths.** GPU MC with ≥ 1e9 float64 inputs per model, all four events in one pass, requested from T192. I will not
read DEV3's open truths.json or any count before my estimates are committed.

**Scoring (T350's).**
- r = |est/p − 1|.
- s = sqrt((1 − p)/(pC)) if pC ≥ 1, else s = 1 (the zero answer).
- Win if r < s − .05, loss if r > s + .05, tie otherwise.
- Cases with p = 0 in the MC count are "undefined" and reported separately.

**Go.** Both of the following must hold:
- (1) A wins ≥ 2/3 of the decided (non-tied) cases overall, out of 80;
- (2) the win rate does not fall as t rises: the win share of decided cases at t ∈ {1.0, 1.25} pooled is ≥ the share
  at t ∈ {.5, .75} pooled, minus .10.

Also reported:
- the win share per t;
- the subset with pC ≥ 1, where the comparison is with real sampling;
- the est/truth ratios, against T340's bands (within 2x and within 3x).

**Alternatives first.**
1. **The metric eases as t rises.** As pC falls below 1 the competitor becomes the zero answer (s = 1), and any
   0 < est < 2p wins. Criterion (2) can then pass with no improvement in accuracy. The pC ≥ 1 subset and the ratio
   bands are reported for this reason.
2. **A's accuracy may come from one transverse point that happens to suit E2 at t ≈ .5–.75.** At larger t the
   interior islands may sit away from the line, and the ratio collapses toward 0, as variant A did on E1. In that
   case A still "wins" against the zero answer, and the ratio bands will show it.

## Results
**Order of steps.**
- Estimates frozen at 1247a8cf (`results/est.json`, sha256 4e54933f…7866). The scoring script was committed before
  the truths, at b7200b60.
- Truths: T192 GPU MC, claude/t192-gpu-cost-law @f86b0f40, t358/rare_mc.json, 1e9 float64 inputs per model. The weights
  in t358/weights.npz are identical to dev3/weights.npz.
- To reproduce: `python3 score.py results/est.json <t358 rare_mc.json> results/score.json` (log in `logs/score.log`).
  Estimator cost: 14–26k FPE per model.

**The pre-registered verdict is GO, but it is carried by the zero-answer competitor (alternative 1), and the accuracy
collapses with t (alternative 2).**

| t | win | tie | loss | win share of decided | cases with pC ≥ 1 (win / tie / loss) | est/truth median | within 2x | within 3x |
|---|---|---|---|---|---|---|---|---|
| .5 | 12 | 2 | 6 | .67 | 18 (10 / 2 / 6) | .69 | 16/20 | 19/20 |
| .75 | 15 | 1 | 4 | .79 | 9 (5 / 1 / 3) | .52 | 10/20 | 15/20 |
| 1.0 | 14 | 5 | 1 | .93 | 1 (1 / 0 / 0) | .22 | 6/20 | 6/20 |
| 1.25 | 8 | 10 | 2 | .80 | 0 | .05 | 1/20 | 2/20 |
| **all** | **49** | 18 | 13 | **.79** | 28 (16 / 3 / 9): **.64** | | | |

- **Criterion (1)** passes: .79 ≥ .67. **Criterion (2)** passes: the share at t ≥ 1 is .88 and at t ≤ .75 it is .73.
- **The wins at t ≥ 1 are almost all against the zero answer.** Sampling at 14–26k FPE sees under one event there. Any
  estimate in (.05p, 1.95p) wins under the rule, even when it is 5–20x too low.
- **A's estimate collapses as the event gets rarer.** The median est/truth falls .69 → .52 → .22 → .05. At t = 1.25 it
  is within 2x on only 1 of 20 models.
  - It is one-signed low: the walk at T317's single transverse point misses the deeper interior islands, as variant A
    did on E1 in T340/T350.
- **Where the comparison is with real sampling (pC ≥ 1, mostly t = .5)**, A wins 16 of 25 decided cases (.64). That
  is just under the 2/3 line.

**Honest reading.**
- A is a usable equal-compute estimate of the interior error set only at moderately rare levels (t ≈ .5, p ~ 1e-4 to
  1e-5). There it is within 2x on 16/20 models and beats sampling about 2/3 of the time.
- It is not a rare-event estimator. At p ≲ 1e-6 it under-estimates by 5–20x. The rule scores it a "win" only because
  sampling at that cost usually returns 0.
- The pre-registered GO should not be read as "A estimates rarer events well". A metric that judges a 20x
  under-estimate at p ~ 1e-7 against the zero answer is too lenient for that question. Here that leniency decides
  the outcome, as the pre-registration's alternative 1 warned.

## Claims
- [measured] On 20 fresh DEV3 models, the pre-registered rule gives GO: 49 wins, 18 ties, 13 losses (.79 of
  decided); t ≥ 1 .88 against t ≤ .75 .73.
- [measured] Against real sampling (pC ≥ 1), A wins 16 of 25 decided cases (.64).
- [measured] The median est/truth falls from .69 (t = .5) to .05 (t = 1.25). A under-estimates the rarer interior
  errors by 5–20x.
- [inferred] A beats the zero answer, not sampling, at p ≲ 1e-6. It is not evidence of a working rare-event
  estimator at those levels.

## Alternatives not ruled out
- Variant B, or several transverse points, may track the deeper islands. Untested here; T340/T350 found that B
  over-estimates E2 at higher cost.
- A stricter metric for rare events, for example a log-ratio band independent of the competitor, would score A
  as failing at t ≥ 1. Its definition is left to the DRI.

## Compute accounting
Estimates: 20 models × variant A, about 3 CPU-min, single thread. Truths: T192 GPU.

## Status
**Done.** Formally GO; substantively, A only works where events are moderately rare. Unverified.
