# H4T4 numerical result: fewer integrals, higher signed weight and setup

Sole producer run0059948f (published ea1c1c70) completed under the frozen protocol,
after source gate593cc500 and immutable binding9d9d9937 (published a36429ea).
Eight exact synthetic chains and six rejection guards passed. Independent saved
audit is assigned to recover_78 and remains pending; these are producer results.

| Fixed H4T4 seed0 archive | Original simplex integration | Cancelled boundary, coned to orthant anchors |
|---|---:|---:|
| Nonzero integration terms | 5,058 | 3,056 |
| GL20 nodes | 40,464,000 | 24,448,000 |
| Partial floating node categories (39/node) | 1,578,096,000 | 953,472,000 |
| Absolute normalized determinant sum/upper | 7.47630309540539 | 10.617752711788295 |
| Ideal real-rule truncation upper | 2.9661195223e-8 | 4.2124460713e-8 |
| Numerical Gaussian sum | 0.47701312599411866 | 0.47701312599411900 |
| Measured integration/recording CPU seconds | 1.233678163 | 0.764507846 |

The difference is3.3306690739e-16. This is a floating diagnostic, not a directed
error certificate. The signed method uses60.42% of the fixed-rule nodes but its
absolute determinant weight rises42.02%. Its ideal bound is therefore worse at
the same quadrature order. No changed order was tried after observing the result.
The inherited source evaluation, true-network coverage and disjointness remain
qualified as in the protocol and prior exact-artifact review.

## What cancels

20,232 oriented facet occurrences group into12,535 exact keys.7,697 keys occur twice
with opposite signs and cancel;4,838 occur once and survive. Of the latter,1,782
produce zero determinant at the fixed orthant anchor, leaving3,056 integrals:
2,329 positive and727 negative. All5,058 exact per-simplex determinant expansions
passed. Dyadic signed-total brackets overlap; individual rounding at2^-128 makes
the absolute totals rigorous upper sums for the stored exact determinants.

Only identical subdivisions within an orthant were matched. The count of surviving
keys is not the count of genuine correctness interfaces. Incompatible subdivisions,
cross-orthant cancellation, alternate anchors and different boundary-integral
formulations remain untried; this result is not a lower bound for those methods.

## Charged construction, comparison and integration

New archive decoding/conversion took0.080631254CPU; chain construction, matching,
identity checks, progress recording and compressed archival took2.518363247CPU;
dyadic accounting took0.075982265CPU. The separately reported boundary component
sum (including rule setup and boundary integration) is3.444271777CPU. Full process
CPU, including controls, prerequisites, baseline comparison and other serialization,
is4.858434603; peak RSS147,140KiB. The observed integration saving of0.469170317CPU
does not repay the2.518363247CPU construction/recording phase in this one run.
This includes research durability overhead and is not a benchmark of an optimized
deployed matcher. It does not establish a hardware or native-operation lower bound.

Construction additionally records17,593 exact4x4 determinants,1,688,928 integer
multiplications,422,232 integer accumulations,80,928 GCD calls,80,928 ray divisions,
20,232 dictionary lookups/coefficient updates and20,232 rational identity terms.
These arbitrary-precision operations cannot simply be added to the floating
39/node categories as equal-cost native work. Sorting, control and memory remain
outside that partial arithmetic ledger but inside measured process time.

Full original enumeration is not free. Pinned historical results report
0.042940864CPU for gate enumeration and5.465172562CPU for geometry/exact audit,
plus their original integer-work ledger. Those inherited figures and this run's
new timings are not a matched end-to-end performance benchmark. Neither route
has earned a final-estimator cost or MSE claim from this experiment.

## Decision and next rung

[measured, pending independent saved audit] Exact same-facet cancellation is real
and cuts fixed-rule integration work by39.58%, but not by orders of magnitude;
signed error amplification and construction cost matter. [inferred] This fixed
implementation does not demonstrate a cold end-to-end speedup. It does not rule
out a more efficient matcher, coarser facets, alternative anchors or lower-dimensional
boundary integration.

H4T3 remains outstanding: no exact terminal simplex archive was found in the bounded
inventory. The available M4,3 source saves weights and aggregate results, not the
terminal complex. A separately frozen small exact capture is required before its
numerical rung; no such capture occurred here. C1 H4T4 residual-band numerical work
is separately owned by recover_910 as T278.

Exact command was `bash review/facet_cancellation/launch.sh`, once. Raw source,
progress, all control chains, full target chain, tensor rule, per-term numerical
records and final summaries are retained under `output/`; no failed numerical
attempt or completed scientific rerun occurred.
