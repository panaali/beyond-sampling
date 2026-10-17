# T68/T72 certification handoff to the DRI

The original trained H2T4 and H4T4 real-arithmetic networks now have deterministic
accuracy enclosures, conditional on exact-geometry correctness and the documented
IEEE754/NumPy assumptions. These are reference computations, not a scalable
solution or an equal-compute win for the original H16T10 challenge.

| Model | Total probability interval | Width | Mass computation CPU |
|---|---|---|---|
| H2T4 seed0 | [.3077307251624383,.3077307608854148] | 3.57e-8 | 2.7423s |
| H4T4 seed0 | [.4770130989463459,.4770131530418916] | 5.41e-8 | 28.2451s |

H4 composed cold algorithm cost is38.6053CPU seconds: full successful historical
exact reference10.0096, compression.3505, and current enclosure28.2451. Rule
construction is included. H2 corresponding successful cost is1.3668+2.7423 =
4.1091CPU seconds. These composed costs include redundant controls and exclude
historical failed/review runs, process imports and final report serialization.
No audited FPE or end-to-end wall-clock comparison is claimed.

## Interface and provenance

`check_t4.py --hidden {2,4}` creates exact geometry from the checkpoint's exact
binary-rational weights. This phase has already run; do not rerun it merely to
consume the reference. `check_enclosed.py --hidden {2,4}` checks the stored
integer determinants and evaluates mass. H4 geometry is gzip-compressed JSON.
Each simplex has four primitive integer ray **rows** and an exact Fraction for
the absolute determinant after positive L1 ray normalization.

The reusable primitive is:

```python
mass = EnclosedMass(order=20, seconds=budget_cpu_seconds)
raw_interval, error_upper = mass.mass(integer_ray_rows, normalized_det_fraction)
result = mass.total(list_of_raw_interval_and_error_pairs)
```

The caller must supply nondegenerate same-orthant four-dimensional cones and
their correct determinants. The present validated caller supplies a disjoint
correct-class triangulation, modulo null boundaries. The result distinguishes
the finite-rule rounding interval, an analytic truncation upper bound, and the
total probability interval. Authoritative binary endpoints are also stored as
hex strings. `rule20_certificate.json` records rational root/weight/pi brackets.
There are no random model inputs and no estimator fitting. This API is not yet
an audited `est(model,budget,rng)` production wrapper.

## What is and is not certified

1. Gate maps, projective rays, signs, incidences and determinant partition use
   exact integers/Fractions. This interprets checkpoint weights as exact reals;
   it does not certify float32 runtime decisions after rounded forward steps.
2. Positive Gauss-Legendre cubature uses exact rational root/weight brackets and
   a rational Machin pi enclosure. Every floating arithmetic step/reduction is
   bounded outward, with nonfinite endpoints rejected. A Chebyshev remainder
   bounds quadrature approximation separately from evaluation rounding.
3. Required platform assumptions: correctly rounded binary64 elementary NumPy
   operations, gradual underflow, adjacent `nextafter`, no fast-math reassociation.
   Fixed scalar/vector/reduction controls can falsify these assumptions, not
   formally prove compiler/hardware compliance.
4. Root independently checked checkpoint equality, all stored integer
   determinants and network correctness on rays/interiors of both artifacts.
   T55 independently derived the truncation formula and checked three H2/orthant
   finite-rule values with a separate70-digit rule. These checks support the
   implementation but do not independently prove triangulation completeness
   or disjointness. That still rests on the documented exact-algorithm proof.
5. Only the original H2/H4 T4 checkpoints are covered. Model changes require
   fresh geometry and full charged setup. No higher-T, noisy-copy or scaling
   result follows from these enclosures.

## Next distinct review opportunity after reading T63/R2

T63/R2 establishes a useful mechanism-derived estimate for M4T3 seed0 and tested
noise copies. Its unresolved approximation is freezing gates that do not cross
the true ordering boundary: such gates can still cross the finite predicted
error wedge. R2's slope-near-one cases expose this, while T75/T73 already own
the algorithm repair/scaling and T76 owns H2 transfer. Do not duplicate those
engines or launch further exhaustive reference enumeration.

Proposed distinct task for manager allocation: independently review the first
T76 boundary-family prototype's **gate-validity domain and omitted mass**, using
the existing H2T4 exact artifact only as a read-only diagnostic oracle. First
freeze a deterministic small list of families where a frozen gate's sign on
the boundary differs from its sign inside the predicted error wedge. For an
already linear gate on a proposed convex wedge, exact signs at extreme rays
give a sufficient whole-wedge validity check. For earlier recurrent gates,
the induction must establish upstream signs before using the downstream map;
checking final-layer signs alone is circular. Report unresolved families
explicitly rather than declaring them valid from a boundary midpoint.

Two alternatives and cheap falsifiers: errors may come from missing gates
(look for a sign violation inside the proposed wedge), or from mass evaluation
(compare the exact same supplied cone with this enclosed mass primitive).
Separate diagnostic cost from any production certificate. Neither test proves
useful scaling: building the wedge and its rays might itself grow like region
enumeration. Count family/gate/ray growth and require the production path not
to import or reuse enumerated artifacts. Coordinate this review with T55's
activation-certificate theory and T76's author; no implementation is started
without a separate claimed task.
