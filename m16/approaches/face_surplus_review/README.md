# T82: complete-stencil review of face smoothness

## Question
Did endpoint contamination or known face-window switching cause the negative T34 finite-level surplus screen? This reviews a screen, not a sparse-quadrature estimator or a theorem of impossibility.

## Method
T82 was claimed and pushed TASKS-only at57696fd before this folder was created. The original T34 source, results and independent documentary critique were read. Reimplement the stencil summary without importing `face_surplus.probe.summaries`; reuse the separately reviewed corrected face walker as a pinned common oracle, with bounded compiled-versus-Python checks rather than calling it an independently derived integration engine.

## Frozen protocol and alternatives
1. Endpoint/inverse-CDF contamination may explain the interior statistic. Evaluate both midpoint-selected and strictly whole-stencil-selected interior subsets: all three nodes must lie in[.05,.95]. Fit levels4..8 exactly as T34, plus a declared levels5..8 sensitivity. Retain counts and per-level RMS/L1, not only slopes.
2. Known geometric max/min switches may generate kinks independent of network boundaries. Before examining any mass, derive the face-window switch locations from fixed anchors and split there. Compare width-weighted within-panel surpluses with an unsplit rule over the same strict interior interval. This is a plausible boundary-aware rescue; it does not identify or fit network-output roots.
3. A correct screen may be driven by rare lines or shared-oracle error. Retain individual line energy, top5% share and mass values. Check a fixed small set with the Python walker and independent synthetic quadratic/hinge controls. A failed rescue does not exclude boundary-aware splitting at actual network switches, mixed-derivative structure or other coordinates.

Freeze four cells: (T5,noise.0025,R23,axis2 upper input), (T5,noise.02,R12,axis2 lower input), (T10,noise.0025,R23,axis2 upper input), (T10,noise.02,R12,axis7 lower input). These cover both lengths/families and representative previously important axes; they cannot establish the original leading-three criterion by themselves. Take32 independent anchors per cell using RNG820001+100*T+int(family)+axis, fixed configuration on each line. Reproduce original relative-RMS perturbation copy0 with RNG1, sequential Wi/Wh/Wo noise. Never read accuracy truths.

Each unsplit line uses257 nested dyadic nodes on[1e-5,1-1e-5]. Each geometry-split interior panel uses257 nodes on its interval; combine squared/L1 surpluses using panel widths, and report its extra evaluation cost. Compare to257-node unsplit[.05,.95] at the same levels; this is a representation diagnostic with unequal work explicitly reported, not an efficiency win. No data-dependent extra knots or level changes.

For R23 upper coordinate v, let s be the fixed tie value, o2 the largest lower input. The two window branches exchange at o1=4*s-3*o2, hence v=[Phi(4*s-3*o2)-Phi(s)]/[1-Phi(s)]. Include the cap8.5 crossing o1=s+25.5/sqrt2 if within the interval. For R12 lower coordinate v, max-lower identity changes at v=max(other lower quantiles); cap8.5 crosses at v=Phi(s-25.5/sqrt2)/Phi(s). Keep only strict interior knots and remove exact duplicates. These formulas derive solely from the face geometry. They were not selected using mass outputs.

Before trained runs, verify quadratic surplus exponent2 and a known hinge with/without its exact split; check strict selection excludes both endpoints of every selected stencil. Check geometry knots by evaluating the two window expressions on each side. Check Python versus compiled walker on the first anchor's17 coarsest nested nodes per cell. Save raw grids/values locally underdata for independent recomputation; source/results JSON are committed.

## Status
Bounded review complete with independent raw-statistics verification. The frozen complete-stencil and geometry-knot rescues do not clear the screen on the four selected coordinates; a same-count control limits attribution. Actual network-switch and mixed-regularity alternatives remain open. No sparse-quadrature family is retired.

## Results
Pending. T34 used200 anchors per coordinate; this is a smaller independent diagnostic with new anchors, not a same-stream numerical replication.

## Compute accounting
One thread, <=2GB RSS, ulimit-v2500000; numerical cycles use shared compute/review locks and<=110CPU (review lane<=20CPU). Count every face evaluation, walker-returned cost, geometry/table/setup and validation CPU separately. Walker FPE alone excludes construction/statistics and cannot establish end-to-end efficiency. No model fitting, random model-input Monte Carlo, or full-input region enumeration. Random anchors are diagnostic face locations, not a production deductive estimator.

## Claims
[inferred] Midpoint selection does not ensure the complete stencil is interior. [inferred] Explicit face-window max/min switches are computable before evaluating the network. No measured smoothness claim yet.

## Alternatives not ruled out
Actual network boundary splitting; coordinate changes beyond known face geometry; adaptive weighted rules; mixed-dimensional sparse grids; other models/axes; rare-line effects beyond32 anchors.

## Failure modes
Finite decay slopes are not convergence theorems. A shared face oracle may retain common bugs despite backend comparison. Panel splitting changes scale and evaluation budget. Floating window walking and tabulated inverse CDF are not interval certificates. Screen success does not establish an efficient accuracy estimator, and screen failure does not retire sparse quadrature.

## Independent pre-run review and control refinement
T55 independently checked every proposed knot formula. Some cap candidates are inactive because another min/max branch dominates; such extra cuts are harmless but must not be called observed kinks. The reviewer also identified a confound: splitting into panels increases evaluations and shrinks physical steps, even for a smooth quadratic. Before any trained run, add a same-cost control: on each anchor, use the same number of panels as the geometry split, but divide[.05,.95] into equal widths. Compare that equal-count control to the geometry-aligned split, and retain the unsplit comparison only as an unequal-cost diagnostic. The synthetic hinge gets the same control. This is a preregistered review-driven refinement, not a response to trained outputs.

The two exact checkpoint hashes match the original T34 record: T5 c026d6be1a44e4004d009443b768411e046e75ef42090e525b5ac8a3719e7040; T10 498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb. The corrected walker is compiled unchanged into this folder's ignoreddata directory using g++-std=c++17-O3-ffp-contract=off-fPIC-shared; no other agent's source or artifact is edited. Compilation used .443 wall seconds. URLs are the same public weight artifacts cited by T34; no new ARC prose was read.

## First bounded result
The four frozen coordinates all remain below the1.5 RMS-exponent heuristic when complete stencils are strictly interior. This independently checks only the selected coordinates/new anchors, not all original leading-three sets.

| Case | Midpoint interior | Complete-stencil interior | Unsplit [.05,.95] | Geometry split | Same-count uniform panels |
|---|---:|---:|---:|---:|---:|
| T5,.0025,R23,axis2 | 1.4321 | 1.0792 | .9413 | 1.0890 | 1.1184 |
| T5,.02,R12,axis2 | 1.0649 | .8270 | .8290 | .9986 | .8504 |
| T10,.0025,R23,axis2 | 1.2490 | .9517 | 1.1913 | 1.1802 | 1.2314 |
| T10,.02,R12,axis7 | 1.1847 | .9201 | 1.1311 | 1.0531 | 1.2341 |

These are finite levels4..8 fits. Complete-stencil levels5..8 sensitivity is1.1399/.8470/.9119/1.1135. At level8 the geometry-split RMS divided by same-count-uniform RMS is .9908/.5224/1.5771/2.0903: one substantive improvement, one near tie, two losses. Thus known window switches explain some variation but their removal does not consistently repair the screen. Do not infer a sparse-grid error bound or a family-wide impossibility.

Independent stencil controls give exact quadratic exponent2 and remove the known single-hinge surplus when its breakpoint is split. Whole-stencil selection keeps6 rather than8 level4 stencils. The68 fixed Python-versus-compiled face values agree within7.78e-16. All98 retained geometry knots are active switches in these particular cases, and their branch-equality residual is at most1.00e-15. This contrasts with the general possibility of inactive cap candidates, which remains correctly documented.

T55 also derived the pure resolution factor for a quadratic: with relative panel widths p_j, width-weighted RMS split/unsplit=sqrt(sum_j p_j^5); K equal panels give K^-2 without changing exponent2. `quadratic_control.py` checks the actual saved panel geometries against this independent expression; maximum factor discrepancy9.84e-14. This demonstrates why an amplitude reduction alone is not proof of useful boundary alignment. All trained mass values and panel specifications are retained in ignoreddata/*.npz for independent recomputation; all per-line energies and summaries are committed in `results.json`.

## Actual compute and reproduction
Four cases require181,956 compiled face evaluations, returning2,117,363.534 walker-FPE. The68 Python backend checks are additional validation and separately counted in JSON. Whole cycle2.830CPU seconds, peak261,816KB RSS, including torch/checkpoint preparation; compiled build previously .443 wall seconds. Geometry and walker CPU are separate per case. This is diagnostic overhead, not an end-to-end estimator comparison. Restore the two recorded public checkpoint files into this folder'sdata directory and build the unchanged walker.cpp there, then run:

```sh
ulimit -v 2500000
ulimit -t 20
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/face_surplus_review/review.py
```

The model-free `quadratic_control.py` uses the saved panel files and makes no network evaluation. T55 independently recomputed the first raw NPZ with separate summary code in commit421d359: RMS/L1 differ at most5.422e-20, the three corresponding slopes match, and the quadratic panel factor agrees within1.108e-12. This used .023CPU seconds and no new face calls.

## Finite alternatives checklist
- Tested midpoint versus whole-stencil interior selection, declared level-window sensitivity, known geometry-knot splitting, same-count uniform panels, synthetic smooth/hinge controls, source/compiled backend agreement, both lengths/families and two existing noise levels. All retained, including the one geometry-alignment improvement.
- A different table interpolation may affect the tie-value axis. This subset holds tie value fixed and varies upper/lower inputs, so that explanation cannot account for its observed line irregularity; no claim is made about untested tie-axis results.
- Actual network gate/output boundary switches along the face coordinates were not isolated. A boundary-aware rule using them remains open, requires a new costed algorithm, and is not excluded by splitting only the coarea-window switches.
- Larger anchor ensembles, other coordinates/seeds, mixed derivatives, weighted/adaptive sparse rules and alternative coordinate maps remain untested. No blind wider sweep was run because it would not isolate a mechanism.
- Oracle agreement is a different backend of the same reviewed formulation, not an independent derivation or interval certificate. Independent raw-summary review passed at421d359; any remaining common-oracle defect would require a separate discrepancy fixture.
