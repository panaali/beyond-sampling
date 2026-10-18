# T264 retained cost/error frontier contract

## Fixed question and source

Can the published T206 error-sorted skip result support its claimed upper bound on savings when leaf mass costs differ and the allowed sampling error changes with remaining cost? Use one predetermined panel: H16,T4,zoo0,chamber at owner45e1f6c, pinned fully in manifest.json. Initial work reads collector/summary/estimator/cost source and Git artifact metadata only. No retained array or result payload has been decoded in this phase.

`anatomy.py` wraps every mass call with a separate Ops counter, then records one terminal `_cell_error` result and its mass-call cost increment per row. It separately computes each leaf's total Gaussian mass with a throwaway counter; that diagnostic is not estimator cost. `kind` is inferred from mass-call absence or a floating error-versus-leaf-mass comparison, not a rigorous label/geometry certificate. The saved rows are activation-leaf contributions, including mixed leaves, not one row per homogeneous correctness region.

`anat_summary.py` sorts `err` ascending and takes the largest cumulative prefix below a fixed SE computed at original total ops. It then subtracts the selected `mass_ops`. This is one feasible oracle heuristic if its nonnegative error/cost assumptions and source numerical values are accepted. It is not a general cost-optimal oracle upper bound. Counterexample with error budget0.2: two rows have(error,cost)=(0.1,1),(0.2,100). Error-first takes only the first; taking only the second saves100. This counterexample concerns the fixed-budget optimization claim, not any measured target result.

## Conditional model and units

For n retained rows, let e_i>=0 be their stored numerical error contributions, c_i>=0 their stored named mass-operation costs, O the original total named ops, M=sum_i c_i, and G=O−M>0. G remains charged for every selection: outside-mass geometry, clipping/readout and other counted work are not removed. The wrapped mass() includes its own pulling-triangulation face scans/rank tests, so c_i includes that internal geometry; it is not purely quadrature arithmetic. Stored costs are source-category proxies, not a complete native-operation or runtime ledger. No extrapolation to H16T10 follows.

Let p in(0,1) be the stored original accuracy; v=p(1−p). The original sampler convention is c_s=flops(H,T)+TH+3T−2, with flops=T(2H²+2H)+2HT. For H16,T4 this is2378 named FLOPs/input. The estimator convention is two FLOPs per stored Ops unit. For subset S, saved ops s(S)=sum c_i, removed error E(S)=sum e_i, and remaining nominal FLOPs C(S)=2(O−s(S)). The descriptive one-SE condition is

    E(S)^2 C(S) <= v c_s.

This continuous-equivalent sampler convention matches the existing report. It is not the exact finite-floor sample count, a confidence interval, or a certificate of sampling performance. Integer sample rounding would be a separately named alternative. p is held fixed at the original target accuracy, never replaced by the pruned estimate. G>0 avoids the zero-cost singularity. v=0 or invalid input will be reported outside this protocol's domain, not silently repaired.

If true disjoint error contributions equal e_i, replacing each selected contribution by zero overestimates accuracy by exactly E(S) in real arithmetic. Here e_i are numerical quadrature outputs; there are no per-row certified intervals in this source contract. Even exact arithmetic on their binary values proves only the retained-value diagnostic. Geometry coverage, first-index degeneracies, mass integration error, floating summation and cost completeness remain inherited limitations. Oracle knowledge of all e_i and selection work cannot be free deployable advice.

## Fractional optimistic bound

Allow fractional removal x_i in[0,1]. For each saved cost s in[0,M], define E_min(s)=min(sum e_i x_i : sum c_i x_i=s). Ignore zero-cost rows in this optimization (choosing them cannot improve saved cost and can only increase error). Take all positive-cost zero-error rows first; then order positive-cost rows by increasing exact ratio e_i/c_i, using source row index for deterministic ties. Fractional knapsack exchange proves that this order attains E_min(s): moving equal cost from a larger error/cost ratio to a smaller one never increases error. Hence E_min is nonnegative, continuous, piecewise affine with nondecreasing slope.

Every feasible integer subset with saved cost s has E_min(s)<=E(S), so it satisfies

    2 E_min(s)^2 (O−s) <= v c_s.

Conversely the fractional prefix at s realizes this retained-value inequality. The largest feasible fractional s therefore gives an optimistic upper bound on integer saved cost. It is not a deployable fractional leaf rule, nor a lower bound on achievable estimator cost across other algorithms.

Crucially, feasibility need not be monotone in s: larger removal increases bias while lowering remaining cost and raising the allowed SE. On one segment E_min(s)=a s+b>=0 with a>=0,

    d[E_min(s)^2(O−s)]/ds = E_min(s)[2a(O−s)−E_min(s)].

For a>0 the bracket is affine with slope−3a. Thus this function has at most one interior maximum and its minimum over a segment occurs at an endpoint; a=0 gives a nonincreasing function (constant if E_min=0). Consequently, a segment with both endpoints infeasible has no feasible interior. Evaluate every breakpoint, including0 andM, rather than stopping at the first violation.

If M is feasible it is the fractional optimum. Otherwise take the largest feasible breakpoint s_j. Its next endpoint is infeasible, and every later segment has both endpoints infeasible. On(s_j,s_(j+1)) there is exactly one transition from feasible to infeasible, except a feasible boundary equality may make that transition the left endpoint itself. Exact rational bisection with the predicate above brackets the largest feasible saved cost; retain both lower and upper endpoints and finite resolution. The upper endpoint is a valid optimistic savings upper. No floating root solver or uniqueness assumption beyond this segment proof is needed.

The resulting cost lower bound is2(O−s_upper), always at least2G. The best independently verified integer subset gives a feasible savings lower s_feasible. Together these bracket the integer optimum; a small gap would settle the retained diagnostic more strongly than the previous heuristic. A large gap stays inconclusive. Fractions describe exact stored binary e_i/p values, not exact true masses.

## Frozen next protocol proposal, not execution clearance

1. Independently validate schema/finite values, exact nonnegative costs/errors, sum of mass costs, strictly positive fixed G (stop otherwise), row count, stored total-error/accuracy relations and source provenance. Preserve negative/cancellation rows; do not clamp, drop or reinterpret them to force eligibility. If strict assumptions fail, stop and write a separate revised proposal.
2. Retain original ascending-error fixed-threshold prefix, a self-consistent ascending-error prefix scan, and exact error/cost-ratio prefix scan. Evaluate all prefixes under their own remaining cost, not first crossing only. Use index tie-breaks. These are concrete feasible integer selections, not proofs of integer optimality.
3. Compute the fractional bound above with exact comparisons and a predetermined bisection precision. Retain selected row identities, fractional segment endpoints, all breakpoint decisions and construction/selection work separately. No new masses, regions or model calls.
4. Before application, freeze synthetic controls covering unequal-cost counterexample, zero-error costs, ratio ties, endpoint equality, nonmonotone feasibility, no-positive-mass-cost, invalid negative/NaN inputs and resource/failure durability. Source and saved-control review must precede the one fixed retained panel.
5. Report original and remaining category costs, bias and SE, integer feasible lower/fractional upper savings, and all qualifications. Do not call a conditional oracle bound a full algorithm ceiling or transfer a one-model result to noise/width scaling. Broad T212 numerical campaigns remain frozen.

No array payload, optimizer contraction, scientific numerical test, source replay or model execution is authorized by this written phase.
