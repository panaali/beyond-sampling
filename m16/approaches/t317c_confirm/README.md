# T320: independent confirmation of T317c (first deductive lenient candidate)

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T312](../r1w_repro/README.md) · original: claude/t218-tie-tail-labels @1c742430 `m16/approaches/r1_cont/` (estimator_c.py, cont_c.py; README T317/T317b/T317c)

## Question (DRI)
T317c scores HP2 F 6.72 [4.84, 8.73] at an official C of 5,455 FPE. Its estimates equal R1W V1 at G = 20.
- (a) Recount the executed FLOPs independently under Directive 8's written convention.
- (b) Audit that every step is deductive or a declared approximation, and that the model-free tables are
  model-independent, precomputed once and listed.
- (c) Reproduce the estimates on 10 DEV2 models within .012 pp.

## Pre-registration (2026-10-18, before running anything)
**Method for (a).** My own instrumented implementation, built on my T312 re-implementation (`../r1w_repro`), not
on cont_c.py. The four savings named in the T317 README are implemented and **executed**:
1. **Continuity restart:** at a flip (s*, j*), the values follow by affine continuation. Tangents are recomputed only
   for steps > s* (grouped by s*), and per-step root minima only for steps >= s* (cached below).
2. **B2 prefix tree:** the first cell (value, tangent, per-step root minima, logits) is computed once per distinct
   (input, tangent) prefix across the 720 lines of a line type.
3. **Merging touching correct pieces** before the Φ profile.
4. **Model-free Φ tables** for endpoints exactly 0 and tmax, and for the blind A12.

**FLOP counting method (independent):**
- every executed numpy operation on weight-dependent data is tallied by its element count: multiply-add 2,
  every other scalar op (add, compare, select, min/max, divide) 1, Φ/ndtr 20;
- indexing, np.unique keys and Python bookkeeping are uncharged, as in the scorer;
- model-free tables are uncharged and listed.

**Panel:** the 10 T297 DEV2 models (indices 0, 8, …, 72). G = 20 (T317b's frozen value).

**Comparison:** import `estimator_c.py` (frozen, claude/t218-tie-tail-labels @1c742430) only for the final
side-by-side: per-model estimate and cost_flops. The official HP2 C is 5,455 FPE; the official DEV2 count is
re-derived by running the frozen file on the same 10 models.

**Pass (DRI, pre-registered):** my C within **5%** of the frozen code's C (mean over the 10 models) **and** no
undeclared step in the audit **and** all 10 estimates within .012 pp. Any doubt is reported plainly.

## Status
Pre-registered.
