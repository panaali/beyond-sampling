# T264 independent retained-frontier contract review

## Question
Does the proposed retained-value fractional relaxation bound integer leaf-subset savings under its stated self-consistent cost/error condition?

## Method
Written proof/source review under assignmentd9b83886. Read final CONTRACT/README atdd5afb28 plus four source snapshots at owner45e1f6c. Independently compare text bytes/SHA/Git blobs and three retained artifact tree IDs/sizes without reading their payloads. No optimizer, mass, model or array execution.

## Status
Pass for written source/theorem/protocol design only; execution_allowed=false. No target frontier, synthetic implementation or experiment is approved by this artifact. Explicit source/resource gates and fixed numerical conventions remain prerequisites for later controls/application.

## Results
[inferred] The collector wraps mass with its own Ops counter, charges that to caller and cumulative MASS, and stores each terminal _cell_error contribution and mass-cost increment. Diagnostic leaf_mass uses the original mass function with a throwaway counter and restores simplex statistics. Its cost is not in the retained estimator Ops. The kind field is inferred from no mass work or numerical error/leaf-mass closeness; it is not a rigorous label certificate. Leaves may contain correct and incorrect portions.

[inferred] The original summary sorts numerical error and chooses a prefix at the original-cost SE. This is one conditional feasible subset heuristic, not maximal cost savings: with allowed error0.2, the stated two-row(0.1,1)/(0.2,100) example separates the objectives. At reduced cost the allowed continuous-equivalent SE increases, so a fixed-original-threshold feasible subset remains feasible, conditional on nonnegative contributions and the declared convention. Floating source summation is not an exact stored-value optimality proof.

[inferred] G=O-sum(c_i)>0 denotes counted work OUTSIDE wrapped mass calls. The final correction preserves this distinction: c_i includes triangulation face scans and rank tests INSIDE mass(), so not all geometry stays fixed. G, positive numerical errors/costs, fixed p in(0,1), and no free oracle construction are explicit premises. The source sampler expression gives2378 named FLOPs/input atH16,T4; two FLOPs/Ops is a convention, not a native runtime bound. Continuous-equivalent sampling is not finite-floor sampling, a CI or a mass certificate.

[inferred proof] Remove zero-cost rows from the optimizer (retain them in accounting); an optimal relaxation never needs positive-error zero-cost rows. At fixed saved cost s, exchanging equal saved cost from a higher e/c item to a lower one cannot increase removed error. Thus ratio-sorted fractional prefixes attain the nonnegative continuous piecewise-affine E_min(s), with nondecreasing slopes. Positive-cost zero-error items come first. Every integer feasible subset has E_min(s)<=E(S), and O-s>=G>0 preserves the squared-inequality direction. Hence the largest fractional feasible savings is an upper bound for the fixed integer-subset model, not for all pruning algorithms.

[inferred proof] On each segment put E=as+b>=0,a>=0. For F=E²(O-s), F'=E[2a(O-s)-E]. If a>0, the bracket decreases with slope-3a; E cannot be negative on the segment, so an interior strict minimum is impossible. With a=0 the function is nonincreasing. Consequently both endpoints infeasible implies no feasible interior. Global feasibility may still be nonmonotone as slopes change: every breakpoint must be checked.

If M is feasible, all positive-cost rows realize fractional savings M; this is not an instruction to remove positive-error zero-cost rows. If M=0 the optimum savings is0. Otherwise, after the largest feasible breakpoint, its next breakpoint and all later breakpoints are infeasible; later segments cannot contain feasibility. On that following segment there is exactly one feasible-to-infeasible boundary, possibly its left endpoint. Exact bisection can therefore maintain feasible lower/infeasible upper endpoints without assuming global monotonicity. The retained upper endpoint bounds maximal integer savings;2(O-upper) is a corresponding remaining-cost lower bound and is at least2G. A separately verified feasible integer subset provides the opposing savings lower bound. Duplicate breakpoints and zero-cost cases require explicit handling in future controls.

[measured metadata] All four source snapshot bindings and three artifact IDs/sizes pass. Data payloads remain unread. Final docsdd5afb28 incorporate reviewer-requested strictG>0 and outside-mass geometry wording; no mathematical formula or source changes were required.

## Compute accounting
Only text/Git metadata reads, hashes and syntax-free source inspection. Zero author imports, scientific optimizer runs, array/JSON-result payload decoding, mass/geometry/model/RNG calls. Command: `python m16/approaches/feynman_fundamentals/t264_frontier_review/check_metadata.py` (immutable output).

## Claims
[inferred] The fractional proof is valid under the fixed retained-value assumptions. [measured metadata] Source/artifact bindings pass. No actual target feasibility, useful relaxation gap, certified bias, full-cost ceiling, noise-family orH16T10 inference.

## Alternatives not ruled out
Numerical errors/costs may violate eligibility; fractional gap or oracle discovery cost may be large. Joint pruning, different integration rules or cheaper positive certificates may outperform this fixed zeroing model. A feasible heuristic with a loose upper bound remains inconclusive.

## Failure modes
Later numerical scope must freeze tolerance/rounding treatment before reading outcomes, preserve negative/nonfinite rows, require strictG>0 andp in(0,1), and keep original-error-first heuristic distinct from self-consistent scans. Binary-exact retained values are not exact true masses. All construction/selection work and inherited source coverage/tie errors remain outside any unqualified deployment claim.
