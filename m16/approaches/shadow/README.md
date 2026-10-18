# shadow — T290: algebraically coarsened shadow network screen (Astra r2 C2), DEV2 80 retrains

↑ [TASKS](../../../TASKS.md) · review/codex_astro_review_r2 (C2)

**All claims unverified. Screen only (DRI-bounded to ~1 CPU-h); sampling is diagnostic.**

## Question
Do these models have exploitable algebraic slack? Normalise, round the weights, and merge units that become identical. Is
there a precision at which the network shrinks while its accuracy barely moves?

## Pre-registration (written 2026-10-18, before any T290 number was computed)
**Shadow construction (deterministic, no fitting).**
1. **Normalise.**
   - Readout shift removal: Wo <- Wo - mean_rows(Wo). argmax is unchanged because every logit gains the same v·h.
   - Positive per-unit rescaling to unit-norm incoming weights [W_ih[j], W_hh[j,:]]. Scaling row j of W_ih and W_hh by c_j and
     column j of W_hh and W_o by 1/c_j preserves the function exactly. The scaling is iterated 200 times (Sinkhorn-style,
     fixed); the residual is reported.
2. **Round at precision 2^-k, k = 1..8**, two pre-registered variants:
   - (R) relative mantissa: each weight keeps k fractional mantissa bits, relative error <= 2^-(k+1);
   - (G) grid: step 2^-k · max|W| per matrix (W_ih, W_hh, W_o separately); small weights become 0.
3. **Merge.**
   - Algebraically identical units are found as the coarsest equitable partition of the recurrence. Refine from one block by
     the signature (W_ih[j], sum over block C of W_hh[j, C]), compared exactly (values rounded to 12 decimals), until stable.
     Units in one block have identical h at every step (h_0 = 0).
   - A block with all-zero signature is dead (h ≡ 0) and is removed. Blocks that feed neither W_o nor any live block are
     unobservable and are removed.
   - The merged shadow has one unit per remaining block, with summed outgoing weights. It is checked equal to the rounded
     model on 1e4 inputs.

**Measurements per (variant, k) and model.**
- Distinct units (original: 16, unless the original is already reducible; that is also reported).
- Accuracy change, shadow minus original: paired MC, the same 1e6 N(0, I) inputs (rng 290) for both, diagnostic.
- Prefix regions at k_prefix = 3 (width16_probe/prefix.py `run`, unchanged) for the original and every shadow, all 80 models.
  Also k_prefix = 4 for 4 models (fused_s0, fused_s1, torch_s0, torch_s1) at k in {2, 4, 6}, both variants.

**Kill rule (DRI).** Kill unless some (variant, k) has >= 90% of the 80 models satisfying BOTH:
- shrink: fewer units than the original, or >= 4x fewer k_prefix=3 regions;
- abs(accuracy change) <= 0.4 pp.

Also reported per (variant, k): the fraction meeting each condition alone.
**Alternatives written first.**
- (a) The trained nets have no algebraic slack. T232 found generic region counts, so rounding either leaves 16 distinct units
  or destroys accuracy.
- (b) Coarse grid rounding could kill units and still keep accuracy, if those units only matter on low-mass inputs.
- (c) Region counts could fall only because rounding creates degenerate (parallel) gates, without fewer units.

## Deviation (recorded 2026-10-18, after the first run crashed, before any summary was computed)
The 200-iteration unit-norm rescaling does not converge. Its scales drift: on fused_s23, W_ih underflowed to ~1e-44, so grid
rounding and the 12-decimal merge comparison became meaningless, and the merge check assertion fired. The function was preserved
in exact arithmetic, not in floating point.
- **Fix:** a single pass of the same rule, c_j = 1/||[W_ih[j], W_hh[j,:]]|| applied once. This is deterministic, with no drift.
- The 23 models already done were discarded and everything was rerun. The first-run per-model lines (shadow_run1.log) are kept
  for the record; they showed the same qualitative picture.

## Status
Done. **KILL** under the pre-registered rule (unverified).

## Results (unverified; `python m16/approaches/shadow/shadow.py /tmp/claude-0/dev2/weights.npz` -> shadow.json, shadow.log; `python m16/approaches/shadow/summary.py /tmp/claude-0/dev2/weights.npz` -> summary.json)
- All 80 originals have 16 distinct units, so no original is reducible.
- Paired MC uses 1e6 inputs. "shrink" = fewer units or >= 4x fewer k_prefix=3 regions.

| setting | shrink | abs(Δacc) <= .4 pp | both | median Δacc (pp) | median units | median regions3 ratio |
|---|---|---|---|---|---|---|
| R1 … R5 | 0 | 0 | 0 | -69 … -7 | 16 | 1.00 |
| R6 | 0 | 0.01 | 0 | -1.76 | 16 | 1.00 |
| R7 | 0 | 0.33 | 0 | -0.55 | 16 | 1.00 |
| R8 | 0 | 0.82 | 0 | -0.10 | 16 | 1.00 |
| G1 | 1.00 | 0 | 0 | -82.6 | 1 | 0.01 |
| G2 | 0.97 | 0 | 0 | -82.8 | 3 | 0.02 |
| G3 | 0.85 | 0 | 0 | -82.6 | 11 | 0.13 |
| G4 | 0.49 | 0 | 0 | -82.0 | 16 | 0.51 |
| G6 | 0.12 | 0 | 0 | -79.3 | 16 | 0.86 |
| G8 | 0 | 0 | 0 | -64.5 | 16 | 0.95 |

**Prefix regions at k_prefix = 4** (4 models):

| model | orig | R2 / R4 / R6 | G2 / G4 / G6 |
|---|---|---|---|
| fused_s0 | 18,696 | 17,108 / 18,342 / 18,556 | 415 / 13,805 / 18,348 |
| fused_s1 | 19,345 | 19,986 / 19,434 / 18,488 | 29 / 9,167 / 14,378 |
| torch_s0 | 17,263 | 20,630 / 15,741 / 17,255 | 34 / 4,620 / 10,186 |
| torch_s1 | 19,724 | 18,179 / 20,778 / 19,322 | 54 / 9,168 / 14,805 |

## Claims
- [measured] **No precision both shrinks the network and keeps accuracy** (frac_both = 0 for all 16 settings). The rule kills C2
  in this configuration.
- [measured] **Relative (mantissa) rounding never merges a single unit**, even at 1 bit, and leaves region counts unchanged
  (ratio 1.00). Accuracy recovers only at 8 bits, and there 82% of models are within .4 pp. So the units are algebraically
  distinct and no identical units are hidden.
- [measured] **Grid rounding shrinks the network only by zeroing small weights.** That destroys accuracy: -64 pp even at 8 bits.
  On fused_s0, rounding only W_hh at G8 drops accuracy 87% -> 40% (W_ih alone: no change; W_o alone: 72%). After normalisation
  W_hh spans 5e-4 … 7.3. [measured, diagnostic]
- [inferred] **The function depends on small recurrent weights**, about 2^-8 of the largest and below. There is no algebraic
  coarsening slack: dropping small weights is not a safe simplification. This matches T232: the trained nets have generic
  region counts, with no exploitable "training made it simple".

## Alternatives not ruled out
- The normalisation rule is a single pass (see Deviation). A different gauge (e.g. per-unit scaling chosen to equalise the W_hh
  dynamic range) could make grid rounding less destructive. Not tried; it would be a fitting-free alternative.
- Rounding only some matrices, or rounding to non-uniform grids, was not tried.
- Merging up to a tolerance (approximately identical units) is a different, non-algebraic method.
- A coupled discrepancy certificate (the rest of C2) was not attempted, because no shadow is close enough to certify.

## Failure modes
Accuracy is extremely sensitive to W_hh: a 2^-8 grid or a 6-bit mantissa costs ~2–60 pp.

## Compute accounting
- Main run: 80 models x 16 settings x (1e6 paired forward passes + k=3 enumeration), about 40 CPU-min.
- k=4 enumerations: 28 runs, about 12 CPU-min.
- Aborted first run: 23 models, ~12 CPU-min.
- Diagnostic sampling only; no estimator was run.
