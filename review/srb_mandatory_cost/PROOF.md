# Conditional mandatory-work lower bound for stratified Rao–Blackwell cost

## Frozen question and source

The question is whether the current returned charge can cover work the unchanged source must perform, under explicit accounting premises. This is not a full cost census, runtime claim, estimate of accuracy, or leaderboard reclassification. T214 was claimed in fecab78e; pre-artifact refinement/delegation27eb2b01 chooses a conservative dense-only floor rather than all visible arithmetic.

Sources are the already reviewed immutable T204 snapshot: rb9.Q/h9, cv.make/draw_all, estimator plain allocation and leaderboard policy/override. All paths/hashes and the entire source-declared CURVE_BUDGETS=(1000,4000,16000,64000,256000,1000000) are frozen in protocol.json before any new arithmetic contraction. No budget is selected for a favorable result. The owner has seen34's existing deterministic budget1000 composition message; this is not a blinded historical experiment, and those values are not inputs to this proof or the future calculation.

## Premises

1. The H16,T10 stratified no-CV entry follows the frozen source and returns successfully with a nonempty kept prefix batch of M rows. No model/source replacement is made.
2. At least one retained row has finite prefix values with distinct largest and second-largest values. For a clean simple sufficient condition, all retained prefixes may be assumed finite with distinct top two. Exact ties in finite RNG output are not declared impossible. This premise must be verified or explicitly conditioned on when attaching a historical conclusion.
3. Dense matrix multiplication is charged in the declared canonical algorithmic convention, two FLOPs per multiply-add, including the dense zero-state recurrence as implemented. This is not a lower bound on hardware instructions, optimized sparse algorithms, or a different implementation.
4. For this leaderboard comparison one FPE is6000 FLOPs at H16,T10, as in its source. The classic scalar dot-expression convention is a separate diagnostic, never silently mixed.
5. Conditional rule5 interpretation: one retained prefix evaluation receives one free generated input/label allowance. All D−M additional rejection/stratum-search candidate inputs incur the stated0.04FPE each. This interpretation is explicit; it needs DRI adoption to become an unconditional policy verdict. Model-free preprocessing reuse remains governed separately by rule4.

## Lemma: at least three active interval positions

For a retained row in premise2, the sorted Q breakpoints contain two distinct finite numbers m2<m1 from its input ranking. Additional neural breakpoints may coincide, be infinite, or be NaN; the source replaces NaNs by +infinity before sorting. The edge list also contains -infinity and +infinity.

Any sorted extended-real list containing two distinct finite values has at least three strict adjacent increases: from -infinity to its first finite distinct value, across at least one finite gap between m2 and m1, and from its last finite distinct value to +infinity. Repeated values add zero-width positions but do not remove those increases. Thus this one row has hi>lo at at least three distinct loop positions.

The batch skips a position only if no row has hi>lo there. Hence the union of active positions J is at least3. No claim about typical J, the number of a different row's intervals, or historical input law is needed.

## Dense mandatory source work

h9 executes nine dense (M,16)@(16,16) products; Q then executes one more to form its final-step offset. Every active interval position executes two dense (M,16)@(16,10) products for A and B, including rows where a position is invalid. No per-row pruning or rank-one update appears in this source path.

Canonical dense count is

20*M*16^2 + 4*J*M*16*10 = (5120+640J)*M >= 7040*M FLOPs.

The separately named classic dot-expression count is

10*M*16*(2*16-1) + 2*J*M*10*(2*16-1) = (4960+620J)*M >= 6820*M scalar units.

Both count vectors derive from the independently reviewed T204 call shapes; the second is not a substitute canonical/native charge. Sorting, labels, elementwise arrays, comparisons, reductions, CDFs, allocation/control, memory and setup are omitted. Their omission makes a conditional source-category floor, not a complete total.

The earlier reviewed visible category totals7699M/7479M at J3 include9M input negations used only in argsort(-x). Rule5 exempts input-label generation, so those raw categories are not silently promoted to required charges. The PRIMARY bound here uses only dense7040M. Even the optional label-excluded visible counts7690M/7470M are not needed for the conclusion and will not be used as the primary floor.

## Candidate generation lower bound

The frozen draw_all draws exactly B=50000 candidate rows per loop iteration. Every candidate is assigned one stratum; kept subsets are disjoint and truncated to remaining quotas. A successful call returns M=sum(nh) kept rows, so D is a positive multiple of B and D>=M. Consequently

D >= D_min(M) = 50000*ceil(M/50000).

This uses only source flow, not an empirical rejection factor or the generated stream's probability law. It is usually loose: stratum deficits may force additional batches. The source discards actual D when make receives draw_all's result.

Under premise5, an exact rational policy floor in FPE is

L(M) = 7040*M/6000 + (D_min(M)-M)/25.

The classic-dot diagnostic substitutes6820 for7040. Dense operations and extra candidate allowance are disjoint under this stated interpretation. No claim is made that0.04 is a measured operation count or that omitted candidate processing can be added again without checking the policy.

## Returned charge and decision

The source forms binary64 c=(1.6+0.1), n=int(budget//c), and quotas nh=max(4,floor(alloc*n)) elementwise, with the frozen seven-element plain allocation. M=sum(nh). It returns C_raw=int(ceil(M*c)). The harness multiplies the WHOLE returned integer by the binary64 expression (3.3/1.6). Its rounded returned value C_report is distinct from replacing just COST_Q or from exact decimal multiplication.

A future bounded calculation must retain the exact source binary values/hex and use exact rational comparison of the resulting binary reported charge with L(M). L(M)>C_report establishes a conditional undercharge under premises1–5; L(M)<=C_report is INCONCLUSIVE, not proof the charge is sufficient. It must also report the dense-only floor without rule5, the classic-dot alternative and omitted categories. This is not a claim about a specific historical run's D, unobserved prefix ties, complete realized cost, estimator MSE or validity of the estimator's output.

## Alternatives and cheapest discriminators

- A large rejection ratio could explain a charge gap, but this bound needs only the unavoidable batch multiple. Actual D would tighten it;34's separately frozen deterministic APIstream controls test source composition without estimating a Gaussian rejection law.
- Different free-input interpretation could change the policy floor. Keep dense-only and extra-candidate terms separate so a DRI ruling changes only the explicit term rather than the source arithmetic.
- The3.3 hand factor might cover omitted Q work at large M. A floor below it remains inconclusive; no negative is extrapolated to all budgets or methods.
- Equal top inputs can reduce the minimum strict intervals. The J>=3 claim is conditional, not an assertion that the finite sampler never produces ties. Actual saved batch top2 can be checked if available, but no new input draw is authorized here.

## Accounting and phase boundary

This first artifact is source/theorem only. No estimator/Q/h9/draw_all calls, model loads, random inputs, geometry, CDF/reference evaluations, native trace, quota or cost contraction has been executed. Reading and hashing frozen source/advice are administrative provenance work, not a claim that algorithmic setup is free. A separately published bounded driver/source gate is required for the six arithmetic rows.
