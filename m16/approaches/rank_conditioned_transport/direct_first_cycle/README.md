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
