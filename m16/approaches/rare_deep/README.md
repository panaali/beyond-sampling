# T368: independent re-check of the deep-tail rare-event claim (H-R1 extension) with my frozen T360 code

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T360](../rare_k/README.md)

## Question
The H-R1 extension says that K-node walks are within 1.3x on 10/10 fresh (model, threshold) pairs from p = 1.8e-10 to
7e-7 (T364 + T366). That rests on the T218 child's estimator code. Does my independent implementation (T360, K3),
run unchanged, reproduce it?

## Protocol (2026-10-18, before any run)
**Disclosure.** The truths are already public: T192's conditional-law MC, t364/ and t366/tail_mc.json on
claude/t192-gpu-cost-law. My estimator is frozen, so this checks the code, not the selection.

**Frozen code** (sha256, unchanged since T360 at e539bf63/977f147a):
- `../rare_k/rare_k.py` `89d073825fa21433355aaad63e95e7bbba63676d9c6f4cdd421af9298f55196f`
- `../rare_check/rare.py` `e7db88752c8c1139793eb23994017a8685a36003dc03553103d54d8a9bc06b97`
- `../r1w_repro/r1w_indep.py` `f737102b57775a7c28729ad066fa1170e7128cbd97e3fbf5629465d41507df3d`
- `../rare_check/cont_e.py` `50dd1c9b85512e23f037171d19de03527f8f5989758cfb74d93b441de48fda2b`

**Run.** `rare_k.est(model, K=3, t0s=…, ts=())`, i.e. E1(t0) = {argmax = p3, g23 > t0}, on the 10 pairs:
- dev3_s526 at t0 = 3.25 and 3.5;
- dev3_s528–s531 at t0 = 2.35 and 2.75.

Weights come from T192's t364/ and t366/weights.npz. **Score:** est/truth within 2x; also reported against 1.3x,
with C.

## Results
To reproduce: `python3 run.py` (`results/deep.json`, `logs/run.log`). The hashes are asserted at run time. About 4
CPU-min.

| pair | truth p | my K3 est/truth | C (FPE, both thresholds together) |
|---|---|---|---|
| dev3_s526, t0 = 3.25 | 1.01e-9 | 1.20 | 1.20M |
| dev3_s526, t0 = 3.5 | 1.82e-10 | 1.12 | 1.20M |
| dev3_s528, t0 = 2.35 | 3.68e-8 | **1.39** | 1.29M |
| dev3_s528, t0 = 2.75 | 2.72e-9 | **1.31** | 1.29M |
| dev3_s529, t0 = 2.35 | 1.43e-7 | 1.09 | 1.39M |
| dev3_s529, t0 = 2.75 | 1.16e-8 | .88 | 1.39M |
| dev3_s530, t0 = 2.35 | 7.16e-7 | 1.23 | 1.04M |
| dev3_s530, t0 = 2.75 | 7.02e-8 | 1.23 | 1.04M |
| dev3_s531, t0 = 2.35 | 2.43e-7 | 1.13 | 1.09M |
| dev3_s531, t0 = 2.75 | 1.91e-8 | 1.23 | 1.09M |

- **Within 2x on 10/10** (ratios .88–1.39). This reproduces the deep-tail claim with independent code.
- **Within 1.3x on 8/10.** dev3_s528 is at 1.39 and 1.31, so the "1.3x on 10/10" strength does not quite reproduce.
  T366's own plain K3 range was .88–1.39, and my numbers fall in that same range.
- The estimates are one-signed high on 9/10, as in T366 (fix 1.10–1.30).
- **Cost.** My walk is charged with T312's cell convention and runs to t = 6, at about 1.0–1.4M FPE per model for
  both thresholds. T366 reports 0.36–0.47M for its plain K3. My implementation is costlier. The accuracy claim is
  not affected.
- Plain MC at the reported costs expects about 1e-4 to 1 events, so these are genuinely rare-event estimates.

## Claims
- [measured] My independently written K3 estimator, frozen since T360, is within 2x of the conditional-law truths
  on 10/10 fresh deep-tail pairs (p = 1.8e-10 to 7.2e-7), and within 1.3x on 8/10.
- [inferred] The H-R1 deep-tail extension holds at the within-2x level independently of the T218 child's code. The
  tighter "1.3x" figure should read "1.4x" for an independent implementation.

## Alternatives not ruled out
- These models are not triple-tie-dominated (T366's own caveat). A pass here says nothing about the target model's
  deep tail, where K3 is known to fail.

## Status
**Done.** 10/10 within 2x; 8/10 within 1.3x. Unverified beyond this check.
