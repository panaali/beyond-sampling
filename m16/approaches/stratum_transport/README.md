# T58: stratum-preserving H2 forward transport

## Question
Can a forward measure representation retain the origin, axes, input ray and area mass without smoothing away positive probability, and preserve useful output probabilities on a trained H2 model?

## Method
Root reserved/pushed T58 in4f28e0e before this work. Begin with the unnormalized two-input law, not the augmented target/radial law. Use T55's `strata_theorem.md` and four-stratum toy in `theory.md`. For first-input sign sigma, write s=abs(x1)>0 and v=ReLU(sigma*Wi), so h2=ReLU((Wh*v)*s+Wi*x2). Split q=x2/s at its two exact gate thresholds. Each sector carries independent standard-Gaussian density phi(s)phi(x2), s>0. Its probability is its planar angle divided by2*pi.

Rank0 gate maps produce the origin atom. Rank1 maps produce a line density: choose a nonzero output coordinate z=a·(s,x2), with variance ||a||²; condition the input vector on z and integrate the one-dimensional perpendicular Gaussian over the sector's linear bounds. Rank2 maps produce the restricted Gaussian density phi(B^-1*h)/abs(det B) on the mapped sector. Keep these components separately even when their supports coincide, so mass accounting is additive and first-index output ties remain explicit.

## Status
Frozen first cycle, before implementation. Claude remains DRI; root handles shared registry updates. No full accuracy estimator or challenge solution is claimed.

## Results
Pending. Public H2T8 seed0 checkpoint is fixed, SHA256 d9cefb20379ab66434c6ea6e923eb791c86dbf0477669b03a1d5c0ae9957954c. Short-horizon output probabilities are marginals of that model's recurrence/readout, not its T8 second-largest accuracy.

## Compute accounting
Each cycle <=120CPU seconds, one thread, shared `/workspace/work/codex-compute.lock`, process limit2GB and `ulimit -v 2500000`. Count sector construction, exact rational comparisons/arithmetic separately from floating density/CDF/angle operations and adaptive integrand evaluations. No random model inputs and no large Monte Carlo. Sorting/bit complexity must be disclosed if uncounted. This is a primitive-validation exercise, not an equal-compute claim.

## Claims
- [inferred] The two-input sector construction is an exact pushforward representation in real arithmetic; floating evaluation needs separate error checks.
- [inferred] A finite number of supports does not imply finite cheap density evolution at longer horizons.
- [speculative] Keeping the singular pieces explicitly may improve forward transport without the target-coordinate interpolation used in T65.

## Alternatives not ruled out
1. A density formula may omit a singular component or double-count it. Test the exact toy masses (.3237918088252166 origin,.25 input ray,.0512081911747834 axis2,.375 area), independent density formulas and total mass1.
2. Sector geometry may be right while output ties or Gaussian conditioning are wrong. Compare rank1 densities against independent one-dimensional integration, test duplicate/constant readouts and positive weight rescaling, and compare trained-model output masses with a separately constructed angular interval partition.
3. The primitive may be exact at two inputs yet expensive or inaccurate after another push. Only after cycle1 passes, freeze a third-step forward integration comparison against independent spherical gate geometry. Do not build a full3D solver first.

## Failure modes
Zero input weights and collinear gate maps need explicit handling. First-index ties can carry positive mass at the origin or on an axis/ray. Floating cone boundaries and near-singular maps are not certified by normalization alone. Fixed target insertion can create offset strata; T55's two-fresh-input theorem has explicit nondegeneracy requirements. For normalized augmented states, rank equal to ambient source dimension produces an ellipsoid surface rather than a volume density; gate restrictions retain conditional fiber acceptance. This first prototype does not approximate those augmented laws.

### Frozen cycle1 protocol
Construct the exact toy law for Wi=(1,1),Wh=diag(1,2), followed by trainedH2T8's two-step law. Validate all four toy component densities and masses, complete mass partition, first-index ties, and positive scales1e-16/1e16. Use deterministic fixed hidden points and independent adaptive scalar integration only; no model-input sampling. Use a separate full-circle angle partition as the trained output/mass reference. Keep rational geometry costs, floating CDF/density costs and validation work distinct. Commit/report results before extending to step3.
