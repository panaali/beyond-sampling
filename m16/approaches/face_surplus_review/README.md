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
Protocol frozen; model files absent at their old paths after environment recovery. Download only the exact public checkpoints already identified by T34, verify their recorded hashes, and retain this provenance. No substitute model is permitted.

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
