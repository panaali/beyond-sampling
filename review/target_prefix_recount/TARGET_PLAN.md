# Next-phase plan only: target provenance and planar check

This document proposes the next bounded phase after the frozen18 analytic controls and their saved review. It authorizes no target calls and contains no target enumeration result. The published T219 claim requires a separate provenance/source/resource gate before target work; k3/k4 and optional k5 remain later scopes.

## Existing bytes, not a new fit

The T201 pilot (source f568e61c, raw5c742488, independent saved75da4bad) stores `review/mechanism_claims_review/rebuild/pilot/weights.npz`, SHA256 `fea77e7cb6e6e23ba2dcba71466b1cd69ac31063f76a3a949fb5f7ea8ab63466`, with Wi16, Wh16x16, Wo10x16 float64 arrays. Pilot results bind the original432 float32 coefficients and conversion bytes to checkpoint SHA256 `498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb` at `/workspace/work/t39_sources/example_checkpoint.pth`. That saved review did not independently deserialize the checkpoint. No historical model call is being inferred from an archive hash alone.

The proposed independent extraction will deserialize that checkpoint once with weights_only=True, CPU, no model construction, check exact keys/shapes/dtype/finiteness and original byte digests, and compare the float64 conversion byte-for-byte to the existing archive. Common denominator D for Wi/Wh will be an exact power of two determined from the stored floats' exact ratios. Every integer coefficient/D will be checked against its original ratio. Wo is provenance-only and is not needed for gate-pattern counting. No biases or initial-state parameters exist in this contract; architecture source and saved metadata must retain that qualification.

## Independent planar partition

For x0=s*u, u>0, s in {-1,+1}, the first hidden state is h1_i=max(s*Wi_i,0)*u. Write the next preactivation as alpha_j*u+Wi_j*x1, with alpha_j=sum_i Wh_ji*max(s*Wi_i,0). All values are exact fractions. Neurons with Wi_j!=0 contribute a possible slope x1/u=-alpha_j/Wi_j; sort/deduplicate these exact slopes. Consecutive slopes (including two infinite tails) give exactly one open interval each. Choose rational interval representatives and determine every second-step sign directly from the scalar affine forms. A zero Wi row contributes a constant sign or coefficient-zero OFF, and does not add a root. Distinct roots must change at least one gate, so intervals have distinct complete histories within each halfplane.

This independently derives the planar history count and boundary directions using scalar Fraction arithmetic, without core.clip/lift/rank. Compare all gate histories and exact primitive rays to the cone implementation, not just the total count. Duplicate roots, permanent-zero gates and both unbounded tails must be retained explicitly. Both x0 signs have different initial masks because Wi is not all zero. The x0=0 line is a null boundary and contributes no separate full-dimensional region.

## Later resource decisions

Only after provenance and planar agreement are independently saved-reviewed should a separate k3 scope be frozen. Depth4 must use the measured cheaper-phase cost to choose durable parent chunks under110CPU each and a finite overall budget. Every parent is handled once; failed/unmatched attempts remain failures, not skipped work. Exact history certificates, counts, operation/bit-size inventories, CPU and partial work must be retained. No count at k2..4 proves future factor30 or a T10 lower bound, and no result excludes aggregation across regions.
