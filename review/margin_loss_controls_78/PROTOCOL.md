# T259 fixed numerical controls

Claim b2825a8c (shared b35a7c16) authorizes preparation of the already registered numerical discriminator. This protocol freezes 17 cases before any candidate execution. The preceding written theorem and source limits are in margin_loss_protocol/PROOF.md (3bc9e2c3), independently reviewed in 593589de. Neither prior written review nor this document alone authorizes execution.

## Cases and decision

fixtures.py contains all literal inputs and analytic expected outputs. The base X is the 2-by-2 identity, hidden targets [[2,1],[1,3]], diagonal penalty [1,2]. Its hidden ridge coefficients are [[1,1/2],[1/3,1]]. A rectangular and a rank-deficient readout check commutation, including first-index zero ties. Common-mode perturbations and an unpenalized shared intercept are tested separately. Three fixed routing choices use cell 0, cell 1, and the pooled fit; routing never depends on target columns.

Eight cases cross unordered sum, ordered sum, unordered mean, ordered mean with sample sum/mean. The pair solver constructs the expanded pairwise design directly, imposing a zero-sum coefficient gauge, and solves that system. The comparator solves ordinary ridge and is centered only afterward. Thus the two systems differ in dimension (4 versus 2) and construction. Matched penalty scaling is alpha times the sample factor; the ordinary comparator is unchanged because scaling its entire objective does not change its minimizer. For K=3 the respective alpha values are 3,6,1,1. Exact residual pair sums, centered squared norms and penalties are saved. Expected centered coefficients are [[2/3,1/6,-5/6],[-1/3,1/3,0]].

Four separating cases are fixed in advance:

- Unmatched regularization: X=I, Y=[[2,0],[0,1]], diagonal penalty [1,3], query [1,7/2]. The matched pair solution predicts [1/16,-1/16], winner 0; retaining an unscaled penalty predicts [-1/30,1/30], winner 1.
- Output-specific penalties 0 and 1 yield 1 and 1/2 from identical scalar targets. Their sum 3/2 differs from fitting the summed target with zero penalty, which yields 2.
- Singular X=[1,0] has no unique generic ridge solution. The expected singular failure is retained; explicit fixed and alternative nullspace maps agree on training and differ on the query. Arbitrary summed-target nullspace choice yields query 7 instead of 0.
- Nonnegative training targets [2,1], query -1, identity readout: linear prediction [-2,-1] chooses 1; clipping hidden predictions to [0,0] chooses first index 0.

Success requires every exact assertion, 17 passed rows, 36 solve attempts, 35 returns, precisely one expected singular failure and 36 progress events. No tolerance is used. A wrong fixture, failed invariant or exhausted cap remains a failure; no automatic retry or fixture substitution. The singular case is a deliberately expected exception, not an unreported failed campaign. No extra malformed-domain sweep is included.

## Independent verification and hypotheses

Two explanations for historical replacement error remain: loss weighting is inadequate, or the frozen feature map misses dependence. These controls only discriminate whether changing to the specific unweighted all-pair quadratic loss is algebraically new. Weighted/top-rival/nonquadratic losses and richer features remain open.

An implementation that accidentally centers all results would make positive cases look plausible; the deliberately unmatched penalty and ReLU controls must instead change labels. A pair-design normalization bug might preserve a single sum convention; the eight fixed normalizations and saved objectives make that falsifiable. A post-run independent reviewer reconstructs each saved system and coefficients with different arithmetic code, all fixture outputs and objective identities, without importing candidate functions or refitting trained data. Numerical controls remain finite examples of the separately reviewed theorem, not a new universal proof or 99% performance result.

## Resources, provenance and durability

One process only, using the exact launch.sh: codex-review.lock, one thread, 15 internal CPU seconds, 16 soft /20 hard CPU seconds, 2 GiB address-space soft/hard, 25 wall seconds after lock. Python invoked/resolved path and binary digest are frozen in protocol.json. Stdlib exact Fraction arithmetic only; no NumPy, model, checkpoint, RNG, Gaussian, DEV2, HP1/HP2 or source producer imports/calls. Matrix dimensions are at most 4. Calls and rational systems are retained; these are source call counts, not a native-FLOP or deployable fit cost.

Before launch, publish the independent completed source/resource artifact, then a separate immutable gate.json binding every source/protocol file and that review. Verify the successful push before a separate launch tool call. Protocol and gate pin maps are checked independently, and every review pin is checked. The gate must name the published scope and source review. Source-stage metadata checks/AST parsing are allowed; no candidate/reference execution before this sequence.

The runner refuses any existing output directory; the launcher refuses any prior run1.log. Every case and every linear solve saves its started record before calculation; each result is saved before assertions. Atomic replacement preserves the previous complete JSON on interruption. A directory-ownership flag prevents an attempted repeat from overwriting earlier evidence in the exception handler. All exceptions and CPU failures are retained without automatic recovery. Internal CPU starts inside main, excludes interpreter/top-level stdlib imports and final serialization overhead; external Bash time and OS CPU caps cover the full process. Resource/CPU metadata is not itself a performance comparison.

Reproduction command after the separately published gate: `bash launch.sh`. This command must never be run again on existing output. Reviewer execution has its own frozen source, resource gate, immutable input pins, log and attempt.
