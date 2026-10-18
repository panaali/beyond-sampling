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

## Status
Pre-registered.
