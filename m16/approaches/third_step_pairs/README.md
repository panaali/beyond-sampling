# Exact third-step pair-moment prerequisite

## Question
Can a small exact cone-moment primitive supply the covariance information
missing from the independently reviewed exact third-step means?

## Method
First scope is only a pointed3D cone primitive, using exact feasible rays and
facets, spherical triangle masses and Gaussian integration by parts. See
PROOF.md, primitive.py and the fixed24 controls in controls.py. It does not
assemble trained neuron pairs or run a network.

## Status
Claim91e79d3b pushed. Source138a0e91 independently gated by78/T55; all24 controls now pass, independent saved audit84e0040c passes. Subsequent frozen six-network pair
assembly also completes all340 cells plus8 reference controls; independent saved
audit pending. See NETWORK_HANDOFF.md.

## Results
All24 pass in0.187677CPU; max analytic difference5.998e-118. See HANDOFF.md.
16 full-dimensional analytic controls,2 zero-volume controls,
3 explicitly unsupported lineality cases and3 invalid input controls.
Reproduction after source gate: one thread2GB under secondary20CPU lock,
`/workspace/clean-neurons/.venv/bin/python m16/approaches/third_step_pairs/controls.py`.
Existing outputs/progress refused and all failures retained.

## Compute accounting
Exact named Fraction operations and reduced operand bits are recorded. MP
candidate/reference and geometry CPU are separate. Exact comparisons,
search/sort/GCD, transient bits, validation/reference construction and complete
native/control/memory costs remain unpriced. No model or random input calls.

## Claims
[inferred] The real-arithmetic identity is exact under the stated facet and
pointedness premises. [speculative] This can support efficient h3 pair moments;
no network covariance, later closure or final accuracy result yet exists.

## Alternatives not ruled out
A geometric implementation error could mimic success on orthants: independent
four-facet/wedge/rotated/redundant controls challenge it. Finite MP cancellation
could hide a correct formula: two fixed precisions and independent160-digit
references discriminate a bounded set, not all inputs. General lineality,
more facets, robust floating kernels and later conditioning remain untried.

## Failure modes
Ray/facet topology, duplicate/opposite constraints, narrow/wide angular
conditioning and signed moment cancellation. Unsupported rank<3 is an explicit
scope restriction, not a zero-mass conclusion. No broad method retirement.
