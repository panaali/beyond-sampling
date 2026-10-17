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

### Cycle1 result
The two-input measure primitive passed. The exact toy masses are recovered, independent toy density formulas agree within2.78e-17, and line-density quadrature recovers each component mass. Nine geometry checks cover the toy, public trainedH2T8 and duplicate readouts, each at scales1,1e-16,1e16. Independent full-circle output probabilities agree within1.12e-16; all stratum/output masses sum to1. Runtime1.038CPU seconds including torch import; deterministic reference direction evaluations and exact coefficient checks are separate in `two_input_results.json`. No random model inputs were used. Reproduce using the shared one-thread/memory wrapper with the short `codex-review.lock` lane and `check_two_input.py`.

Two initial validation commands failed before results were saved: the scaled toy's independent floating reference split a positive-mass logit tie on the input ray. Algebraically identical transformed readouts acquired different rounded dot products, incorrectly reassigning part of the ray's.25 mass. The production law already used exact coefficients and first-index ties. The independent reference now resolves equal transformed coefficient pairs exactly; no arbitrary near-tie tolerance is used. The failed commands' CPU was not separately instrumented, so this cycle carries no efficiency claim. Rational arithmetic/comparison counts exclude bit/hash costs; reference exact-tie arithmetic is counted by coefficient checks rather than FLOPs.

### Frozen cycle2 protocol: exact-form third-step pushforward
Preserve each two-input component together with its source sector rather than inventing a smooth density for singular mass. In polar source coordinates h=r B(cos(theta),sin(theta)), the radius r is Rayleigh and its angular weight is1/(2*pi). Last-input gate and readout intervals have Y/r bounds c(theta)=a*cos(theta)+b*sin(theta). Integrating Y and r gives F(c)=1/2+c/(2*sqrt(1+c*c)). Partition theta where any gate/readout threshold order changes, including output differences independent of Y. On each resulting band the destination stratum and first-index output label are fixed. The antiderivative of F(c(theta)) is theta/2+asin((a*sin(theta)-b*cos(theta))/sqrt(1+a*a+b*b))/2. Infinite endpoints use F=0 or1. This yields an exact-form forward mass calculation, with explicitly floating angle arithmetic.

First verify that antiderivative against independent scalar quadrature at fixed coefficient/interval cases. Then run the toy and trainedH2T8 through three inputs; compare stratum and output masses against an independently constructed3D spherical gate partition. Retain all singular destinations by exact rank of D[Wh*B,Wi], and handle algebraic readout ties explicitly. No target rank or T8 accuracy claim follows from these marginals. Alternative1: missing threshold-order changes assign a wrong label; the independent spherical partition tests this. Alternative2: the formula or cone-boundary arithmetic loses mass; quadrature checks, mass1 and per-stratum comparisons test it. Limit cycle to110CPU seconds under the long shared compute lock, same one-thread/2GB bounds. Count source sectors, threshold forms, angular subdivisions, destination bands, exact rational work, and independent reference polygon operations separately. No model-input sampling or full augmented solver.

### Cycle2 result
The third-step pushforward passes the independent spherical geometry check. Maximum stratum/output discrepancy is1.12e-16 on the toy and6.25e-17 on trainedH2T8. The elementary forward calculation took0.00241/0.01600CPU seconds respectively; spherical reference took0.00553/0.01146CPU seconds. Total validation CPU0.96821 seconds includes torch import and15 independent antiderivative checks. `third_step_results.json` retains all masses, operation counts, small-band omissions and reference work. T55 independently checked the primitive and the independence of Rayleigh radius from a homogeneous source sector before this run.

This verifies an exact-form three-input marginal push, with floating angle arithmetic and explicitly retained singular components. It does not establish fixed-dimensional density closure after further steps, conditional target transport, full T8 rank accuracy or a competitive challenge estimator. The gate/readout ordering is handled geometrically; no random model inputs are used. Reproduce with the existing lock/thread/memory wrapper and `check_third_step.py`.
