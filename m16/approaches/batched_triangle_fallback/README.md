# T116 (triangle fallback): grouped positive quadrature

## Question
Can grouping positive triangle fallbacks by their unchanged selected quadrature orders reduce dispatch overhead in the fixed six-input pilot? Claim783ebb0 was published before implementation. This is distinct from the duplicate bulk-track task ID, pending DRI renumbering.

## Method
Freeze the original scalar order-selection arithmetic, tolerances, binomial bound and positive Duffy rule exactly. Factor selection into a local method; preserve every selected order and per-triangle metadata. Group equal-order triangles only for the existing vertices.T @ bary, squared norm and positive weighted reciprocal-cube sum. Cap each node batch at65536 nodes total. Every rule/node/bound call remains counted with the same conservative arithmetic allowances, plus grouping/scatter work. This is a possible CPU improvement, not an operation-count saving from reducing the mathematical work.

Freeze before results: mixed-order controls using all15 original thin triangles, zero-remainder/coincident endpoints, and existing16 T78 triangles under fixed GL4/8 outer nodes. Compare scalar versus grouped fallback values, selected orders, bounds, generated-rule totals and full counters; absolute/relative floating differences must be reported, not silently treated as identical. Compare exact stored nodes on representative mixed-order controls. If these pass, unchanged T10664-cone indices/orders/inner allocations, using the raw edge variant, one cold pass each; then ten alternating complete pilot calls for timing only. No full93M-node integration, new reference accuracy or noisy models.

Alternatives and discriminators: (1) order selection dominates, so batching nodes may not improve total CPU; measure selection versus node time and complete pilot, not only matrix throughput. (2) batched matmul/reduction changes rounding near a bound; preserve scalar selection and bounds, compare every fixed control and unchanged fallback indices. If dispatch gain is absent, a separately frozen vectorized selection with scalar handling near decision thresholds remains open; do not silently switch variants or declare all batching useless.

## Status
Protocol frozen; implementation next.

## Results
None yet.

## Compute accounting
One thread,2GB; short controls use review lock<=20CPU, longer cycles compute lock<=110CPU. Same conservative node/rule/selection charges as original, plus actual grouping overhead. Cold source/frame/rule setup and artifact overhead separated. No model forward evaluations or random inputs.

## Claims
[inferred] Grouping independent positive evaluations leaves the real-arithmetic integral unchanged when nodes and per-triangle rules are unchanged.
[speculative] It lowers complete pilot CPU without changing the selected orders or numerical guarantees.

## Alternatives not ruled out
Vectorized order selection with guarded scalar decisions, native compilation, different positive integration primitives and tighter independently audited counters. No broad impossibility follows from this finite comparison.

## Failure modes
Batched floating reduction may round differently; original bounds cover quadrature truncation, not all floating error. High-order groups must obey node-memory limits. Grouping changes cache order and may add overhead. An empty fallback set must preserve analytic-only behavior.
