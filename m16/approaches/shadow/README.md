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
