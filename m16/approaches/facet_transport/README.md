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
Done. The spherical flux primitive passes its independent geometric calibration
and all three finite-difference checks. Neither DEV copy reaches the frozen4x
compression gate, even using optimistic geometric-only merges. Stop before larger
models. No global estimator, harmonic projection or solution claim.

## Results

Protocol commit `fcdde11` preceded model evaluation. Reproduce from repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/facet_transport/probe.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/facet_transport/check_geometry.py
```

The first command downloads the public H4,T3 seed0 checkpoint through AlgZoo:
`gs://arc-ml-public/alg/zoo/2nd_argmax_4_3_1073741824_0.pth`.
No unperturbed-model accuracy is evaluated. `results.json` contains all three
cases, counters, validation errors and every finite-difference value.

| Case | Accuracy | Gate cells | Gate/rank cells | Raw decision arcs | Distinct planes | Geometric merged arcs | Optimistic compression |
|---|---:|---:|---:|---:|---:|---:|---:|
| Synthetic H3,T3 | .3232749683 | 29 | 59 | 17 | 9 | 9 | 3.22x |
| DEV H4,T3 sigma.0025 | .9804844930 | 100 | 164 | 114 | 66 | 70 | 1.43x |
| DEV H4,T3 sigma.02 | .9352406923 | 100 | 164 | 86 | 52 | 53 | 1.89x |

Geometric merging groups unoriented plane normals at the frozen1e-9 tolerance
and unions contiguous arcs. **It is not a transport-preserving merge.** Even on
one plane, velocity P_plane*m'/||m|| may differ across source affine cells.
One must preserve orientation and velocity annotations to merge flux. The count
of such patches is bounded optimistically by9–17,70–114,53–86 for the three
cases (geometric merged count <= transport-compatible count <= retaining every
raw annotated arc). Exact transport-compatible merging was NOT implemented or
measured. The lower bounds already fail the4x screening threshold, so no rerun
was needed merely to refine the counts. All flux sums use original raw arcs.

Gate cells also include artificial octant splits; merging identical gate cells
across octants could reduce the denominator and make compression LESS favorable.
These counts measure a representation, not unique abstract activation histories.

| Case | Analytic flux A'(0) | Absolute finite-difference error at1e-3 | At5e-4 | At2.5e-4 |
|---|---:|---:|---:|---:|
| Synthetic | -.01126935677 | 1.04e-10 | 2.60e-11 | 6.67e-12 |
| DEV sigma.0025 | -.01846958919 | 1.51e-7 | 3.77e-8 | 9.42e-9 |
| DEV sigma.02 | -.01674487675 | 4.16e-8 | 1.04e-8 | 2.60e-9 |

Every finest-step discrepancy is below1e-6 and decreases approximately fourfold
per step halving, as expected for centered differences away from topology changes.
No degenerate zero-margin/moving-readout cases were encountered. Total gate/rank
area differs from4pi by at most1.78e-15; affine hidden maps match independently
executed recurrences at all229 interior representatives within6.67e-15.

After protocol freeze, the parent supplied a separate closed-form calibration,
implemented in `check_geometry.py`: compare sign(x0+lambda*x1) with sign(x0) at
lambda=.3. Its exact accuracy is1-atan(lambda)/pi and derivative is
-1/[pi*(1+lambda²)]. Spherical clipping gives area error1.11e-16 and flux error0
in float64. Four octant-split arcs avoid antipodal endpoint singularities.
`geometry_check.json` records the result. This is an independent geometry check,
not an additional candidate model or a change to model selection.

These results establish a useful tiny-complex numerical primitive. They do not
show an efficient method for building the complex, nor that the same compression
would hold at T10 or under recurrent-weight perturbations.

## Compute accounting
Count every polygon clipping operation, generated cells, affine propagation
matrix products, and interior model validation pass. Record CPU/wall separately.
Readout finite differences reuse the fixed gate/rank complex but reclip accuracy.
One thread, at most300 CPU seconds; stop at20,000 gate cells or50,000 rank cells.
Model download is public-checkpoint access, not external solution material.
Measured main run:8480 polygon-clip calls,110 region-wise recurrent affine-map
products,229 interior hidden-recurrence validation forwards, plus spherical-area,
root, sorting and arc-merging arithmetic. The independent calibration adds eight
clip calls and no model evaluation. Total recorded main CPU is1.4252seconds,
wall15.8266seconds including checkpoint loading. Candidate geometry construction
and checks take .0408/.1943/.1722 CPU seconds respectively. No hypothetical
cost-saving from merging is credited. Matrix/clip counters are not a full FLOP
ledger; no MSE-vs-compute or leaderboard comparison is claimed. Raw source/arrays
are small, and no sampled truth or LP solve is used.

## Claims
[inferred] The flux identity is exact away from degenerate facets in real arithmetic.
[measured] Flux agrees with independent closed-form geometry and three decreasing-step finite differences.
[measured] Optimistic geometric retention compresses the two DEV gate complexes by only1.43x and1.89x, below the predeclared4x gate.
[inferred] This explicit-complex transport route does not justify scaling up from the tested tiny models; successful probability differentiation does not remove construction complexity.

## Alternatives not ruled out
Trained T10 may have far more facet normals; recurrent perturbations can alter
the gate complex; retained patches alone do not make their construction cheap.

## Failure modes
Floating-point topology, zero margins/ties, patch count growing with gate cells,
construction cost overwhelming retained representation, and inferring general
tractability from one tiny trained checkpoint. These are numerical geometric
calculations, not directed-rounding formal certificates.

## Sign and geometry derivation

For correct halfspace m(lambda).x>=0, differentiating its indicator gives
 delta(m.x)*(m'.x). On the unit sphere at m.x=0, the tangential gradient has
norm||m||, hence the positive flux sign shown above. For minor-arc endpoints
u,v and angle theta, direct integration of the great-circle parameterization
gives int x ds=tan(theta/2)*(u+v). The fixed gates and rank planes have zero
velocity under readout-only perturbation. Artificial internal interfaces need
not be integrated: only the true output-margin boundaries appear, although
those boundaries still have many cell-specific normals and velocities.

## DRI handoff

T31 was reserved and pushed before work; Claude remains DRI. Recommend register
the narrow negative compression result and retain the exact-flux primitive for
future independent checks. Do not promote it to a mechanistic estimator or a
no-compression theorem. No larger-model experiment is recommended by this probe.
