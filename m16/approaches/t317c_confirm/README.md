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
**Done.**
- (c) estimates: PASS.
- (a) cost: **pre-registered recount FAIL (+8.9%), fully explained.** A post hoc v2 with the leaner root rule agrees
  to +0.1%.
- (b) audit: no undeclared modelling step, but **three documentation gaps** against Directive 8's written
  convention (below).

Plain conclusion: the official C is confirmed as an honest executed count of *that* implementation. A different
correct implementation can legitimately cost ~9% more. All claims unverified.

## Method (as run)
- `cont_indep.py`: my estimator (from my T312 code) with the four savings **executed**: continuity restarts
  grouped by restart step, the B2 prefix tree over (input, tangent) prefixes via np.unique keys, merging of touching
  pieces, and model-free Φ tables. Every executed array op on weight-dependent data is tallied by element count.
- `compare.py [direction|ahead]`: runs it and the frozen `estimator_c.py` on the 10 T297 DEV2 models.
  - Frozen file hashes: estimator_c.py b3c88771…, cont_c.py a0dc5f4a…, at 1c742430.

**Model-free tables in my implementation** (precomputed once at import, identical for every model, uncharged; per
region, G = 20):
- dw8 = w · 720φ(x)/8, Φ(x)^8, Φ(x − tmax)^8, Φ(x), Φ(x + tmax), 1 − Φ(x): 6 × 20 floats per region, 240 in total;
- GL nodes and weights (2 × 20 per region);
- transverse points (v2*, g12*, g23*) and the v4 node per region (8 scalars, exact numerical integration).

## Results (10 DEV2 models; `results/compare.json`, `results/compare_v2.json`, `logs/compare*.log`)
| model | est mine | est frozen | diff (pp) | C mine (pre-reg) | C mine (v2) | C frozen (FPE) |
|---|---|---|---|---|---|---|
| fused_s0 | .886193 | .886221 | −.0029 | 5,978 | 5,486 | 5,470 |
| fused_s8 | .925671 | .925748 | −.0077 | 6,253 | 5,759 | 5,751 |
| fused_s16 | .908244 | .908292 | −.0049 | 5,160 | 4,770 | 4,765 |
| fused_s24 | .938246 | .938285 | −.0039 | 5,864 | 5,377 | 5,355 |
| fused_s32 | .951601 | .951573 | +.0029 | 4,856 | 4,490 | 4,486 |
| torch_s0 | .956754 | .956782 | −.0028 | 5,193 | 4,797 | 4,783 |
| torch_s8 | .935601 | .935635 | −.0034 | 6,751 | 6,183 | 6,175 |
| torch_s16 | .909674 | .909720 | −.0046 | 5,870 | 5,402 | 5,397 |
| torch_s24 | .883828 | .883818 | +.0010 | 5,508 | 5,062 | 5,045 |
| torch_s32 | .924813 | .924847 | −.0034 | 5,970 | 5,482 | 5,482 |
| **mean** | | | max \|d\| .0077 | **5,740 (+8.9%)** | **5,276 (+0.1%)** | **5,271** |

(Official HP2 C is 5,455; these are the 10 DEV2 models.)

**(c) Estimates: PASS.** Max |diff| .0077 pp (<= .012). The residual comes from my exact-integration transverse
points and my GL range (±6.5) versus the frozen MC-derived points and ranges.

**(a) Cost: the pre-registered run FAILS the 5% band (+8.9%). The cause is identified.**
- My counter's breakdown (mean FPE): tangent recompute 1,920; B2 value + tangent 1,545; **restart roots 772 + B2
  roots 250**; continuation 334; logit tangents 315; B2 logits 232; correct_interval 195; Φ 64; rest < 70.
- The whole gap is the **root search per step**. Mine also tests the crossing direction and charges 11H. The frozen
  code only tests "root ahead and finite" (5H), which is equivalent, since any root ahead is a sign change.
- Executing the leaner rule myself (v2, declared post hoc; charged 6H, counting the select) gives **5,276 vs 5,271
  FPE (+0.1%)**, with identical estimates.
- Other components differ in both directions and nearly cancel:
  - the frozen code computes first-cell logits per row (640 per row), mine per distinct prefix;
  - its correct_interval is charged 216 here (229 official), mine 145;
  - its B2 node constant is 1,218 FLOPs, mine 1,312 (pre-registered) or 1,216 (v2).
- So the official count is an **honest executed count of its own code**. The 9% spread between two correct
  implementations is the implementation latitude in "charge what is executed".

**(b) Audit.** Every step is deductive or declared: the R1W V1 approximations (closure c, K = 1 v4 node, v4 moving
with v3, μ linearised, blind g12 assumed correct, the gap beyond tmax assumed correct, product of line sets) plus
four exact cost-only savings. **No undeclared modelling step.** Three gaps against the written convention:
1. **The model-free tables are recomputed inside `est` on every call** (phi0 = Φ(nodes) and Φ(nodes)^8, 90φ(nodes),
   GL nodes via `_gl`, the v4 node via `gauss_v4`), **not precomputed once**. They are identical for every model, so
   this is harmless in substance, but it does not match the convention's "precomputed once".
2. **The tables are named but not listed with sizes** in the r1_cont README, as the convention requires.
   - Per region at G = 20: Φ(nodes), Φ(nodes)^8, density 90φ(nodes), GL nodes and weights.
   - Plus REG's transverse points and the v4 node.
3. **The transverse points come from a seeded model-free Monte Carlo** (r1w2's regions), while the T302 README says
   "exact law" (as found in T312). This is model-independent and deterministic, but undeclared as an approximation.
   It moves estimates by <= .008 pp.

Smaller notes:
- The profile executes ndtr at model-free arguments (endpoints 0, tmax) inside np.where and discards them. Under the
  convention these are uncharged, since their inputs are weight-independent. The DRI's "every executed ndtr"
  variant (F 7.04) is the conservative bound.
- Choosing the table entry (a == 0, a == tmax) is weight-dependent bookkeeping, charged in the merge comparisons.

## Claims
- V1 [measured] T317c's estimates reproduce from independent code within .0077 pp on 10 DEV2 models.
- V2 [measured] An independent executed FLOP count is 8.9% above the official count with a direction-testing root
  rule, and +0.1% with the frozen code's equivalent "ahead and finite" rule. The official C is honest for its
  implementation. Two correct implementations differ by ~9%, so a candidate statement should say "C = 5.3–5.7k FPE
  depending on implementation".
  - At C × 1.089 the HP2 point F would be ≈ 6.72 × 1.089 ≈ 7.3 and the upper CI ≈ 9.5, still a lenient pass.
- V3 [inspected] There is no undeclared modelling step. The three documentation gaps (tables recomputed per call,
  no sizes listed, MC transverse points described as exact) should be fixed before the result goes to ARC.

## Compute accounting
Both estimators on 10 models, twice: about 2 CPU-min.
