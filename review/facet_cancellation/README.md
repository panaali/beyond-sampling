# T267 exact internal-facet cancellation

## Question
Can exact oriented internal-facet cancellation lower weighted Gaussian integration
work on an existing H4T4 correctness complex?

## Method
`primitive.py` constructs an exact signed boundary chain and cones surviving facets
to fixed orthant anchors. `fixtures.py` freezes eight analytic chains and six guards.
`run.py` runs those controls followed by the full archived5058-simplex H4T4 complex
and fixed GL20 baseline/boundary integrals. See `PROTOCOL.md` for the chain theorem,
known incomplete geometric matching, cost convention and resource contract.

## Status
H4T4 bounded numerical phase complete, independently7fe3ac0b. Claim6a164605;
combined scope9a292cb2. H4T3 requires a separately frozen exact capture and remains
outstanding. No completed numerical calls will be repeated.

## Results
See `RESULTS.md`:5,058 to3,056 integration terms,39.58% fewer fixed-rule nodes,
42.02% greater absolute determinant weight. Difference3.33e-16 is diagnostic.
Construction/recording2.518CPU exceeds0.469CPU saved integration in this run.
Raw0059948f; independent saved7fe3ac0b.

## Compute accounting
Full process CPU plus separate exact construction, facet matching, rule setup,
integration counts/CPU. Inherited enumeration remains charged separately. Fixed
GL20 costs8000 nodes per term, with partial39-arithmetic-category node accounting.
Command: `bash review/facet_cancellation/launch.sh` only after published gate.

## Claims
[inferred, independently reviewed593cc500] Oriented chain cancellation preserves
weighted integration under the stated exact chain/anchor conventions.
[measured, independently saved-audited7fe3ac0b] The fixed H4T4 reduction saves
integration nodes but raises signed error weight and does not demonstrate a cold
end-to-end speedup. Other boundary schemes remain untested.

## Alternatives not ruled out
Nonconforming triangulations and orthant-specific anchors can hide further exact
cancellation. Coarser boundary integration could differ. Small-rung behavior does
not establish large-model efficiency or a final estimator.

## Failure modes
Incorrect inherited complex, incompatible subdivisions, signed cancellation,
increased absolute weights, setup cost, floating kernel error or resource cap.
All failures remain; no automatic repair, numerical threshold tuning or rerun.
