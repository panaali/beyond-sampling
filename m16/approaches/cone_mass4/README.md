# cone_mass4 (T109): cheap exact Gaussian masses of simplicial cones in R^4

↑ [Approaches index](../README.md) · [INDEX](../../../INDEX.md) · [TASKS](../../../TASKS.md) · users: [understand_m43/estT.py](../understand_m43/estT.py) (T87), [understand_ladder](../understand_ladder/)

## Question
The Gaussian mass of a simplicial cone in R^4 equals the 4-variate orthant probability of the facet normals'
correlation matrix, which equals the volume of a spherical tetrahedron divided by 2 pi^2. In T87 it is about 90% of
the ops: 12 Gauss nodes (or 48 when any |rho| > 0.95) of the all-correlation Plackett integral, with a 4x4 inverse
at every node. The cheap rule was also biased on thin simplices.

Goal: a function `mass4(normals)` costing far fewer ops (target < 300) with absolute error < 1e-7, including on
thin and near-degenerate cones.

## Pre-registered alternatives (before testing)
1. **No closed form beats a short 1-D integral.** S^3 is odd-dimensional, so the volume of a spherical tetrahedron
   is not elementary (Schläfli). The exact closed forms (Murakami-Yano / Ushijima) need 16 complex dilogarithms,
   i.e. at least about 1-2k ops at 1e-8. *Test:* count the dilogarithm cost; build only if a 1-D rule cannot reach
   the target.
2. **Differentiating along ONE edge instead of all six.**
   - This needs 1 asin per node instead of 6 asin plus a 4x4 inverse.
   - The path ends where det C = 0. There the cone degenerates to (3-D cone with 4 facets) x line, whose mass is a
     spherical polygon (Girard: closed form).
   - The integrand singularity at that end is a square root, removed by phi = phi* + Delta s^2. The 1/sqrt(1-rho^2)
     factor is absorbed by phi = asin(rho).
   - *Risk:* the next singularity (the other root of det) may be close.
   - *Test:* error vs node count on adversarial families against a 30-digit reference that uses an independent path
     (the all-correlation Plackett integral, mpmath tanh-sinh), and 1e8-sample MC on a few cases.
3. **Thin cones are hard only because of rounding.** Their mass is tiny, but the float correlations carry errors of
   about 1e-16 / det. *Test:* a reference built from the normals in 30 digits, not from the float correlations.

## Status
Running (2026-10-17).
