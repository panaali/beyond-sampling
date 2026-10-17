# T31: tiny spherical decision-facet transport

## Question
Does retaining only decision facets substantially compress a tiny RNN's spherical
ReLU complex, and does exact moving-facet flux recover its accuracy derivative?

## Method: frozen before model runs
Use three input coordinates and uniform measure on S² (equivalent to Gaussian
inputs for bias-free homogeneous networks). Start with eight spherical octants.
Split convex spherical polygons by each successive ReLU preactivation; carry the
exact linear hidden map within each cell. Refine only by the three input-ordering
planes to give each polygon a fixed second-largest target. Clip the two output
margins to integrate correctness by spherical polygon area.

Freeze three cases: a synthetic H3,T3 RNN whose all weights are standard normals
from RNG31001; zoo H4,T3 training seed0 with DEV copy0 relative weight noise .0025
and .02 using seed1. The unperturbed trained model is only a base for DEV copies.
No M16 complex, no harmonic projection, no ARC-estimator reproduction.

For each case, perturb only readout W_o(lambda)=W_o+lambda*Delta, with Delta a
fixed RNG31002 standard-normal matrix times0.1*RMS(W_o). Gates and rank regions
therefore stay fixed. On a correct-region boundary m(lambda).x=0 against one
competitor, retaining the other margin, coarea gives

    A'(0) = 1/(4pi) sum_arcs [m'.int_arc x ds]/||m||,
    int_arc x ds = tan(theta/2)*(u+v), theta=acos(u.v).

Compare this analytic sum with centered differences of independently clipped
spherical areas at lambda=±1e-3,±5e-4,±2.5e-4. No random integration is involved.
Report full gate cells, gate/rank cells, nonzero decision arcs, unique normalized
unoriented facet normals, and merged arc components on each plane. Arc merging
uses fixed geometric tolerance1e-9 and is a compression diagnostic only; exact
flux is accumulated before merging. Keep all failed cases.

### Alternatives and cheapest falsifiers
1. Gate-interface cancellation is algebraically valid but surviving decision
   patches remain numerous: compare all-gate-cell and retained-patch counts,
   including splitting caused by artificial octant/rank faces. A >=4x reduction
   in merged patches relative to full gate cells is the provisional go criterion;
   not a scaling claim. If neither DEV copy meets it, stop before larger models.
2. Boundary degeneracy or clipping errors corrupt flux: check total gate and
   rank-refined area4pi, affine hidden maps at cell interior representatives,
   exact octant/hemisphere area geometry, and finite-difference step halving.
   Require absolute flux discrepancy<=1e-6 at the finest step, or clear decrease
   under halving; otherwise flag failure and do not claim exact numerical flux.
3. A large reduction may be a three-dimensional or readout-only artifact: no
   inference to recurrent-weight perturbations or T10 is allowed from this probe.

## Status
Protocol frozen. Implementation and model runs pending.

## Results
Pending.

## Compute accounting
Count every polygon clipping operation, generated cells, affine propagation
matrix products, and interior model validation pass. Record CPU/wall separately.
Readout finite differences reuse the fixed gate/rank complex but reclip accuracy.
One thread, at most300 CPU seconds; stop at20,000 gate cells or50,000 rank cells.
Model download is public-checkpoint access, not external solution material.

## Claims
[inferred] The flux identity is exact away from degenerate facets in real arithmetic.
[speculative] Decision-facet retention compresses the complex enough to be useful.

## Alternatives not ruled out
Trained T10 may have far more facet normals; recurrent perturbations can alter
the gate complex; retained patches alone do not make their construction cheap.

## Failure modes
Floating-point topology, zero margins/ties, patch count growing with gate cells,
construction cost overwhelming retained representation, and inferring general
tractability from one tiny trained checkpoint. These are numerical geometric
calculations, not directed-rounding formal certificates.
