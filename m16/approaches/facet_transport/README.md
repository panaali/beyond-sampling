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
rounds normalized plane components to nine decimals and uses angular interval
tolerance1e-8. It is a compression diagnostic only; analytic flux is accumulated
before merging. Candidate margin-zero arcs require the post-review two-sided
correctness audit below before they can be counted as genuine boundaries. Keep all failed cases.

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
models. The corrected synthetic case reaches5.8x optimistic geometric compression; this
does not change the frozen decision rule based on the two DEV cases. No global
estimator, harmonic projection or solution claim.

## Results

Protocol commit `fcdde11` preceded model evaluation. Reproduce from repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/facet_transport/probe.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/facet_transport/check_geometry.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/facet_transport/boundary_audit.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/facet_transport/check_invariance.py
```

The first command downloads the public H4,T3 seed0 checkpoint through AlgZoo:
`gs://arc-ml-public/alg/zoo/2nd_argmax_4_3_1073741824_0.pth`.
No unperturbed-model accuracy is evaluated. `results.json` contains all three
cases, counters, validation errors and every finite-difference value.

The two audit commands read the same public checkpoint from
`/workspace/work/checkpoints/H4_T3_seed0.pth`, downloaded from
`https://storage.googleapis.com/arc-ml-public/alg/zoo/2nd_argmax_4_3_1073741824_0.pth`.

| Case | Accuracy | Gate cells | Gate/rank cells | Candidate arcs | Genuine arcs | Genuine planes | Geometric merged arcs | Optimistic compression |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Synthetic H3,T3 | .3232749683 | 29 | 59 | 17 | 9 | 5 | 5 | 5.80x |
| DEV H4,T3 sigma.0025 | .9804844930 | 100 | 164 | 114 | 114 | 66 | 70 | 1.43x |
| DEV H4,T3 sigma.02 | .9352406923 | 100 | 164 | 86 | 86 | 52 | 53 | 1.89x |

**Post-review correction:** `results.json` preserves the original provisional
candidate counts, including its invalid synthetic transport bounds9–17. A
margin-zero arc may lie on a gate/rank interface with the same correctness on
both sides. Unoriented merging does not cancel these internal interfaces, and
the finite-difference flux check cannot validate their count. The corrected
output-margin boundary counts above are from `boundary_audit.json`, which
supersedes those provisional counts for interpreting compression. Fixed input
rank interfaces may also change correctness; their zero-velocity patches are
not included here. Thus these are moving-output-boundary counts, not a full
count of every correctness interface. Omitting fixed interfaces makes this
compression screen more optimistic.

`boundary_audit.py` refines candidate arcs at gate/rank vertices and other arc
endpoints lying on their plane. For every resulting piece it evaluates the
full model and the actual second-largest-input target on both sides at three
interior arc locations (fractions1/3,1/2,2/3) and four angular offsets starting
at min(1e-4,arc length/10), decreasing tenfold. A label requires agreement across
all three locations and the three finest offsets. Pieces with finest offset
below1e-10, unstable labels, or nonzero-hidden logits with top gap at most
1e-12*(1+max|logit|) are uncertain. Exact all-zero hidden states use argmax's
defined index tie rule. A genuine piece must have opposite correctness on its
two sides; equal labels make it internal. This is a deterministic numerical
audit, not an exact adjacency proof or directed-rounding certificate.

No extra splits or uncertain pieces occurred in these cases. The synthetic
model has eight internal pieces: six correct-correct and two wrong-wrong.
Their total signed flux is -1.73e-18, explaining why the original flux check
passed despite the false boundary count. All114/86 DEV pieces are genuine
under this audit. The synthetic3.22x provisional result is withdrawn and
replaced by5.80x; neither DEV result changes.

Geometric merging groups unoriented plane normals and unions contiguous arcs.
**It is not a transport-preserving merge.** Even on one plane, velocity
P_plane*m'/||m|| may differ across source affine cells. Orientation and velocity
annotations must be retained for a flux-preserving merge. The corrected
optimistic bounds are5–9,70–114,53–86 patches respectively: geometric components
<= transport-compatible patches <= retaining every genuine annotated arc.
Transport-compatible merging was NOT implemented or measured. Both DEV lower
bounds still fail the4x screening threshold. Flux sums in the original run use
candidate arcs; the audit separately reports the genuine and internal sums.

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

The review also found an absolute-weight-scale bug: the original code treated
small normal norms (1e-15/1e-14) as identically zero. Positive scaling of Wi or Wo
preserves argmax, so that rule was invalid. `probe.py` now checks exact floating
zero, normalizes planes using max-component rescaling, and applies geometric
tolerances only after normalization. `check_invariance.py` checks the same three
cases at nominal scale and with Wi or Wo multiplied by1e-16 (also scaling the
readout perturbation with Wo). Every original candidate/gate/rank count is
unchanged, accuracy differs by at most4.45e-16, and flux by at most3.28e-15.
`invariance_check.json` records these checks. This fixes the absolute-threshold
bug, not all possible underflow, topology, or roundoff failures; no universal
scale-invariance certificate is claimed.

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

The post-review boundary audit adds3642 clips,110 affine propagation products,
and5208 full-model side-validation forwards (408/2736/2064 by case), taking
.2320 CPU/.2320 wall seconds. Its stored results precede the scale-normalization
fix; the fixed code leaves all nominal candidate geometry/counts unchanged in
the subsequent regression. The invariance regression adds12996 clips and330
region affine products, no full-model validation forwards, and .7510 CPU/.7511
wall seconds. These are model-dependent computations and are charged even
without sampled inputs. Across main, calibration, audit and scale regression:
25126 clips,550 region affine products,5437 validation forwards, plus all
polygon/arc arithmetic. Total recorded CPU is2.4082 seconds plus the standalone
closed-form calibration. No further models or candidate variants were explored.

## Claims
[inferred] The flux identity is exact away from degenerate facets in real arithmetic.
[measured] Flux agrees with independent closed-form geometry and three decreasing-step finite differences.
[measured] After removal of internal correctness interfaces, optimistic geometric retention compresses the two DEV gate complexes by only1.43x and1.89x, below the predeclared4x gate.
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
velocity under readout-only perturbation. Internal correct-correct and
wrong-wrong interfaces are not true correctness boundaries and must be removed
for retention counts. The audit explicitly detects them; their zero numerical
flux in the synthetic example does not make their original counting valid.

## DRI handoff

T31 was reserved and pushed before work; Claude remains DRI. Recommend register
the narrow negative compression result and retain the exact-flux primitive for
future independent checks. Do not promote it to a mechanistic estimator or a
no-compression theorem. No larger-model experiment is recommended by this probe.
