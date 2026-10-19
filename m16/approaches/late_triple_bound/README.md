# late_triple_bound: certified late-triple bound with a rigorous box-mass charge (T455)

↑ [m16 approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · predecessor:
[late_triple_verify (T418)](../late_triple_verify/README.md) · T414: branch `claude/t192-gpu-cost-law`, `input_bab/`

**Question.** T418 found the late-triple obstruction in T414's input-space BaB is the margin-steered split rule.
With alpha-CROWN, a widest-dimension split and mass order, the near-exact (float) box mass projected 1.6e6 / 2.4e7 boxes
to 1e-10 on (3,7,8) / (9,8,2). Does a RIGOROUS box-mass bound track the near-exact mass well enough to CERTIFY
P(E1(t0 = 2.5), triple) <= 1e-10 within 1e9 boxes? What would all late triples cost?

**Status.** Pre-registered 2026-10-19 (this section was pushed before the long runs). Owner: Claude
session_011FA8RfY3APG3rkDG2GxScR (own container). DRI session_01CD3g9Cvc3qk3pM11eyXXzj. CPU only, ~150 CPU-h cap.

## Pre-registration (fixed before the long runs)

**Rigorous box mass (`rmass.py`).** For a u-box (T414 coordinates) the slab mass is exactly a 2-D integral over
(u_r, g >= 2.5). The u_a and u_l integrals are closed-form Phi differences.
- The bound is an upper Riemann sum on a 48 x 96 grid: r uniform over the box, g = 2.5 + 17.5 s^2.
- In each cell, each nonnegative factor is bounded by its sup, using cell corners and monotonicity. Each Phi difference
  takes min(corner Phi difference, g_hi x width x max phi).
- The part g > 20 is bounded by the radial formula with the box's min |y|^2 (<= ~1e-30 per box).
- Float64, with the final value inflated by (1 + 1e-9) to cover rounding. The same sum is also implemented in
  python-flint arb (outward-rounded ball arithmetic) for validation.

Trial (not scored), on D3's open boxes:
- bound / near-exact mass: 1.45x median at 48 x 96 (2.0x at 24 x 48), never below 1;
- float bound >= arb bound on 5 boxes at 8 x 16, relative gap 1e-9 (the inflation).

**Validation (scored, recorded pass/fail).**
- (V1) float >= arb upper endpoint, same grid (48 x 96), on 200 random open boxes of the runs.
- (V2) bound >= an independent MC of the box mass, u uniform in the box with the exact radial integral in g, on 200
  boxes: bound >= MC - 3 s.e.
- (V3) Soundness: uniform points in the reservoir of pruned boxes (up to 20,000 boxes, 500 points each). Margins come
  from the reference forward on raw x at random g > 0; no point may be an event, and no sampled margin may fall below
  the alpha-CROWN lower bound recomputed on the box.

**Search (`cbab.py`).**
- T418's D3: auto_LiRPA CROWN on every child, alpha-CROWN (20 iterations) on CROWN's survivors.
- Pruned iff some margin lower bound is > 0.
- Split the widest dimension at its midpoint.
- Best-first by the rigorous charge, popping K = 256 boxes per round.
- Certified remaining mass = sum of the open boxes' rigorous bounds + the root-domain tail (6.2e-12).
- Float only in the bounds: no outward rounding in CROWN, as in T414 / T418.
- Checkpoint every 30 CPU-min; `--resume`.

**Runs (scored).** (3,7,8) and (9,8,2), 20 CPU-h each (10x T418's D3), in parallel, 1 thread each,
`ulimit -v 2500000`: `cbab.py 3,7,8 c378 72000` and `cbab.py 9,8,2 c982 72000` (resumed from checkpoints if the
container restarts).

**Reported.**
- Certified remaining mass vs boxes, and the near-exact mass alongside.
- Pruned fraction.
- Projection: log-log fit over the last decade of boxes, to 1e-10 and to 1e-12.
- (V1-V3).

**Kill line (T414's, fixed).** KILL if the certified projection to 1e-10 exceeds 1e9 boxes per triple, or the certified
mass stalls above 1e-9 (less than a 2x decrease over the last decade of boxes). GO otherwise. "Certified at 1e-10" only
if the certified mass actually reaches <= 1e-10 in the run.

**Cost estimate for all late triples (scored, secondary).**
- Late = the second-smallest of the three positions is >= 7, T414's sense: 22 position sets x 6 orders = 132 triples.
- Run T418's dyadic-leaf estimator (`late_triple_verify/dyadic.py`, alpha-CROWN, n = 500) on 12 late triples: 2 per
  (second-smallest position in {7, 8}) x (p3 position: smallest / middle / largest of the three), first in a fixed
  shuffle (seed 455).
- Calibrate leaves -> boxes on (3,7,8) and (9,8,2): boxes needed by the run (projected) / D2 leaves.
- Total = calibrated mean x 132, in boxes and CPU-h at the measured rate. Early triples are out of scope (T398).

**Alternatives written before results.**
- The certified curve may lag the near-exact curve by a constant (~1.5x), which only shifts the projection.
  Alternatively, it may flatten when boxes near the event boundary, which are never prunable, dominate.
  Check: the share of the certified mass in boxes that hold a sampled event point at the end.
- The one-decade projection may mislead either way; the run is 10x longer than T418's to test that.
- If the per-box cost of the charge matters, the 48 x 96 grid can be coarsened (2.0x looser at 24 x 48).

## Method
- `rmass.py` rigorous box mass (float + arb); `cbab.py` certified BaB. Uses `../late_triple_verify/net.py` (model,
  coordinates, near-exact mass) and `lbab.py` (auto_LiRPA bounders).
