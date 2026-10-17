# T67: direct reachable target-first transition

## Question
Does removing stage1 interpolation for target-first predictions reduce the
remaining h9 full accuracy bias, or reveal cancellation with later stages?

## Method
For label0, target u is inserted into zero state, so h1=relu(Wi*u) exactly.
Compute the same later-grid continuation V2 as the baseline. Apply the
existing analytic scalar Gaussian transition from this single actual h1
into the unchanged stage2 bilinear grid. The remaining unique above-target
position is uniform among T-1 slots: output is (T-2)/(T-1) times below-input
need1 plus1/(T-1) times above-input need0. This removes only the interpolation
of V1 at h1. Other labels, all later grids and readout/tie rules are unchanged.
Initial bounded implementation requires H2T4 with target-last label excluded.

On each sign of u, h1 is linear, and next preactivation is b_s*u+Wi*x.
Every nonzero Wi coordinate yields x(u)=(grid-b_s*u)/Wi at its stage2 grid
crossings (zero is the ReLU boundary). Include x=+-6, x=u, u=0, every pair
line crossing within the sign domain, and all existing later-grid/terminal
knots. Between these, scalar interval ordering and bilinear formulas are
fixed. Integrating degree<=2 polynomials in x against Gaussian density gives
analytic functions of u; no additional exact-arithmetic topology changes.
T55 independently checked the measure and this completeness argument.
Floating branch discovery, moment fallback switches and roundoff remain
uncertified, as does the spatial error of later grids.

## Frozen alternatives and falsifiers
1. Stage1 interpolation materially causes remaining bias: direct removal
   improves full absolute error and error-squared CPU at the same h9.
2. Errors cancel: removing stage1 error worsens final bias. Retain sign and
   per-label changes; this does not imply interpolation removal is invalid.
3. Moving-row formula/topology is wrong: independent scalar quadrature,
   old-node aligned parity, constants/duplicates, and positive scaling must
   pass before trained integration. Charge all moving crossings and moments.
4. Outer integration confounds changes: complete same-panel GL2/4 changes
   must be below1e-7 and10% of any claimed improvement. No h/q sweep.

Controls first; enumerate combined complete panels, abort above2000. Then
one trained H2T4 base, baseline/hybrid at GL2/4. Both use the identical union
panel rule, unchanged h9. No noise expansion unless stable error and cost
improve; a new bounded protocol would be required. No new Monte Carlo.

## Status
Protocol frozen before code or measurements. Scope claimdf9b38a pushed.

## Results
Pending; comparison reference .30773074302392744 is never used by estimator.

## Compute accounting
Full grid/cache setup, all pair intersections/sorting, outer nodes, scalar
moving-row grid crossings, Gaussian moments/fallbacks and later sparse
operators. Clear the shared12-point fallback-rule cache per estimate and
charge its construction explicitly; prior source-mesh timing qualification
is preserved. One thread,2GB,<=110CPU cycles; standard locks. No total portable
FLOP or sampling-win claim based on CPU alone.

## Claims
[inferred] The label0 transition removes one interpolation relative to the
same later-grid continuation, not all discretization or integration error.
[speculative] This produces a useful full accuracy/cost improvement.

## Alternatives not ruled out
Other labels, later spatial grids, conditional geometry, and favorable error
cancellation. No family-wide conclusion from one model.

## Failure modes
Wrong remaining-rank mixture, omitted line crossings, zero/tie ownership,
float narrow intervals, and treating topology completeness as a numerical
error certificate. A failed expansion gate ends this fixed bounded variant.

## Primitive and topology checkpoint
`python m16/approaches/rank_conditioned_transport/direct_first_cycle/controls.py`
under the review lock passes in .428CPU. Independent scalar quadrature of
bilinear continuation agrees within4.44e-16 on five signed target values;
nine first-state node-aligned comparisons against the old full recurrence
agree within1.11e-16. Positive scaling1e-16 gives1.67e-16; constant/duplicate
retained mass errors<=5.56e-17. Moving topology has44 panels; its union with
1190 fixed panels has1227, below the frozen cap. Two interior probes per
non-tiny moving panel confirm fixed pairwise line ordering. This is a
finite floating test, not a robust-predicate proof. Full counts in controls.json.

## Frozen pilot: removal exposes a larger negative bias
`python m16/approaches/rank_conditioned_transport/direct_first_cycle/pilot.py`
completes all four same-panel integrations in47.964CPU seconds.

| Variant | GL4 estimate | Error | GL2/4 change | GL4 CPU |
| --- | ---: | ---: | ---: | ---: |
| baseline | .307723248833985 | -.000007494190 | 8.65e-12 | 15.459 |
| direct label0 | .307547885880563 | -.000182857143 | 1.54e-10 | 16.835 |

Both frozen precision checks pass, but the direct variant loses the accuracy
and cost gate. Its only changed contribution is label0. This is consistent
with cancellation between the removed interpolation and later error; it is
not evidence that the direct transition is algebraically wrong. Constant,
aligned-node and independent scalar quadrature controls remain satisfied.
No ensemble, h refinement or rule search follows this fixed recipe.

Before handoff, freeze one distinct diagnostic: independently partition the
already validated H2T4 R4 cone reference by correct target label, using the
existing floating geometry and positive order20 simplex rule. Compare all
three baseline and direct contributions against those values and require
their sum to reproduce the existing .307730743024 reference. This will tell
whether label0 moves toward its own truth while worsening total error, or
away from it. <=20CPU, no new inputs and no change to the estimator; charge
all geometry/volume work and preserve quadrature-only/floating limitations.

## Independent labelwise reference: large cancelling errors
The separate R4 cone calculation returns correct-label contributions
[.08335201493410087,.13328554811717747,.09109317997264917,0]. Their sum agrees
with the previous full reference within5.56e-17. It uses117 gate cells,
569 correct simplices and4,552,000 positive order20 nodes in .260CPU;
integration-only error bound sums1.79e-8, with no bounded omissions triggered.
Floating geometry/roundoff remains uncertified; exact source hashes recorded.

Baseline per-label errors are[-.003060092074,-.000002521140,+.003055119024,0].
Direct label0 removal changes the first to-.003235455027, leaving the others
identical. Thus removal worsens label0 itself; later discretization can
dominate. The baseline's small total error is cancellation between two
large component errors, and does not establish component convergence.
Missing truncated-input/target mass is below1.4e-8, immaterial at this scale.

T55 source review3624456 independently checks chronology, mixture, cache
independence and topology. It notes omitted fine-grained event counters for
h1 arithmetic and repeated CDF setup: full CPU includes them, event counts
remain partial/unaudited and cannot support portable total cost claims.

Freeze one read-only follow-up before examining earlier contribution rows:
compare this same labelwise truth with every saved h9 GL4 terminal-curvature
and terminal-gate estimate from panel_mesh_results.json, alongside baseline.
Question: did the prior mesh's larger total error conceal an improvement in
individual components? Report every saved variant, per-label errors and
component L1/L2 as diagnostics; do not substitute these for challenge accuracy
or select a winner. No new numerical integration or parameter search.

## Saved terminal-grid comparison resolves an earlier ambiguity
`saved_components.py` uses every prior GL4 curvature/gate row, without new
integration. Terminal-curvature and gate grids reduce component L2 error
from baseline .00432411 to .000339889 and .000431178, respectively (12.7x
and10.0x). Their per-label errors are:
- curvature[-.0000579573,+.0001214123,-.0003121286,0];
- gate[-.0000517998,+.0001622886,-.0003960984,0].

Their total errors still worsen to-.000248674/-.000285610. Thus the prior
terminal-grid negative on total accuracy concealed a substantial improvement
in individual contributions; the old total was fortuitous cancellation.
This is diagnostic evidence supporting continued representation work, NOT a
replacement challenge metric, general estimator win, or selected recipe.
Direct label0 on the old terminal grid instead increases component L2 to
.00444994. Exact-boundary terminal representation and interactions with earlier
interpolation remain open. T55 independently reviews this interpretation.

## Bounded handoff and remaining discriminator
The frozen direct-label0 recipe fails its expansion gate; it is complete,
with independent transition/source review3624456 and separate cone-based
component diagnosis. The latter uses a different geometric formulation but
its new per-label adaptation has author-run numerical status pending DRI
review; no self-confirmation is claimed. No larger grid/noise sweep follows.
A concrete next discriminator is the penultimate-target insertion: replace
interpolation of final-stage terminal values by an original-halfspace cache
at the preceding fixed state, preserving other labels. This directly targets
one measured large contribution error; it requires its own claim/protocol,
strict target-boundary ownership and full topology/cost controls.
