# T267 second rung: one exact H4T3 capture and boundary reduction

Renewed claim6c62a50c follows completed independently audited H4T4 handoff5c30eebf.
This is explicitly a NEW exact geometry capture: no terminal H4T3 complex was found
in the bounded inventory. It is not a hidden rerun of an old artifact. One fixed
archive `arc_m43_repro/m43_weights.npz`, originally labeled H4T3 seed0, supplies
Wi(4),Wh(4,4),Wo(3,4). Read it once with allow_pickle=False, require finite floating
arrays and exact shapes, convert each stored scalar exactly to a dyadic Fraction.
Archive bytes are authoritative here; historical checkpoint identity is inherited
from the archive's source claim, not revalidated through a fresh checkpoint load.

## Order, fixed controls and reuse

Verify runtime/source/raw/reviewer pins before imports/calls. Reuse the prior eight
exact chain controls and six guards through independent saved7fe3ac0b and all its
input hashes. Do not call those controls again. Six NEW closed-angle controls precede
model decoding: positive coordinate octant, negative octant, permutation, positive
rescaling all have mass1/8; its two symmetric midpoint subdivisions have1/16 each.
Save all raw scalar returns before the fixed1e-14 exact-rational absolute check.
No RNG, source model forward, sampled inputs or accuracy-reference replay.

Run the unchanged pinned `rational_geometry/exact.py` generic implementation with
H4,T3: one gates(model), then each gate's exact certificate, orthant split and
pulling triangulation; each gate's outcomes(model) once with the original first-index
tie logic, exact certificates, orthants and triangulations. Preserve gate affine
maps, all cone constraints/tight incidences/rays, model dyadics, gate/correct simplices
and original work ledger. Gate triangulation is validation/discovery overhead and
is charged, not silently excluded from the end-to-end total. Require full gate
normalized determinant sum8 and positive correctness sum below8. These checks are
not an independent completeness/disjointness proof: exact splitting/triangulation
semantics of the inherited source remain prerequisites. Containment certificates
alone do not establish those global properties.

Apply the UNCHANGED reviewed parent primitive.py once to all correct simplices with
d=3. Identical-facet cancellation, signed anchor orientation, zero-anchor omission,
dyadic totals and incomplete geometric matching have the same conventions and
limitations as the first rung. Each orthant's anchor is its sign vector/3. Preserve
all reduced facets, coefficients and terms; no sorting/anchor/integration tuning.

## Closed-form Gaussian mass

For positive determinant magnitude D of L1-normalized rays a,b,c, define

    z = ||a|| ||b|| ||c|| + (a.b)||c|| + (b.c)||a|| + (c.a)||b||.
    mass = atan2(D,z)/(2*pi).

This is the standard positive trihedral solid angle divided by4*pi, valid for the
pointed same-orthant cones here. All pairwise dots are nonnegative, norms and z are
positive. D is converted from an exact normalized determinant, not recomputed in
floating arithmetic. `atan2` uses the positive numerator/denominator branch; no
wrapping or branch rescue. Degenerate anchor cones were removed exactly by the
chain reducer. The kernel rejects nonpositive/nonfinite arguments rather than
silently using zero. The signed row coefficient multiplies the positive mass.

Every original and every surviving nonzero anchor cone is evaluated once. Save the
normalized vertices, squared norms, pair dots, denominator, determinant, atan2,
unsigned and signed contribution as binary64 hex before later cap checks. fsum
both signed totals; report their difference, sum of absolute contributions, term
ratio and cost. This evaluates an exact real solid-angle identity numerically; it
is NOT a directed floating-point certificate. No new quadrature error bound is
claimed or borrowed from the d4 rule. No adaptive precision/rerun or dropped rows.

## Cost and resources

Charge the new exact gate/classification/orthant/triangulation discovery, every
certificate, capture/matching/archival operation and all integration, including the
baseline comparison and six analytic controls. Keep the inherited verification
hashing cost and full process CPU visible. The partial analytic kernel convention
is45 named arithmetic categories per call: six3D dots30, three square roots3,
denominator products5/adds3, atan2 1,2*pi multiplication1, division1, signed-weight
multiplication1. Nine exact rational coordinate conversions per call are separate.
This is not a price for libm, large integers, memory or native instructions.
All discovery integer-operation counters and observed bit maxima are retained;
bit maxima are instrumentation, not an independent complexity bound.

One review-lock process, single threads,17CPU internal,20/20CPU soft/hard,2GiB,
25wall seconds AFTER lock. The inherited exact engine also has a fixed12CPU-second
deadline from its creation,4096-bit intermediates and2000-ray cap. No automatic
resource extension or resume. Existing output directory/log are refused. Use
atomic artifact writes, durable per-gate attempts/results, and false complete/pass
flags plus current partial capture on failure. A returned gate list is saved before
later checks; each returned correct-cone list is saved before its certification.
All completed numerical returns remain. Main timers omit some imports/final writes;
the whole-process OS limit and final CPU cover that distinction.

Gate must have completed independent source/theorem/resource review, direct absolute
pins for angle.py/run.py/PROTOCOL.md/protocol.json/launch.sh, plus all inherited pins
checked separately. Publish immutable gate before the sole launch. Independent
saved review follows; no self-confirmation of geometry or full-cost superiority.

Command, after published gate only: `bash review/facet_cancellation/h4t3/launch.sh`.

Alternatives: few surviving facets may still cost more after exact discovery;
compare full construction and integral work. Incompatible subdivisions or fixed
orthant anchors may hide further cancellation; retain those unknowns rather than
declare all boundary schemes impossible. H4T3/H4T4 differences can be dimensional
or checkpoint-specific; neither transfers to noisy H16T10 or sealed HP1 performance.
