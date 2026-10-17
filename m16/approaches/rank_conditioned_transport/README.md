# T67: fixed target rank and two-dimensional hidden transport

## Question
Does conditioning on the second-largest target eliminate the dominant target-coordinate interpolation error while preserving honest cost and boundary treatment?

## Method
For iid continuous inputs, each target index j is second largest with probability1/T. Conditional on that event, q=F(x_j) has Beta(T-1,2) density. The unique larger index is uniform among the T-1 other positions. Until it occurs, its probability at the next non-target position is1/m, where m is the number of remaining non-target positions. The other inputs are independent conditional draws below or above the fixed target. Target insertion does not decrease m.

We use the normal law truncated symmetrically to[-L,L], with F its normalized CDF. The rank statements remain exact. Original accuracy lies between Pbox*A_trunc and Pbox*A_trunc+1-Pbox, where Pbox=(1-2*Phi(-L))^T, conditional on exact evaluation of A_trunc. DefaultL6 gives omitted mass below2e-8 for T<=10. Numerical quadrature/interpolation errors are separate and uncertified.

Outer target integration uses positive Gauss-Jacobi Beta weights. Conditional scalar integration uses positive Gauss-Legendre quantile nodes. Backward transport uses only two hidden coordinates plus the binary remaining-above state. Each time has its own finite box B_(t+1)=Wh_positive*B_t+L*abs(Wi), startingB0=0, ensuring all reachable transitions are in the next box even if Wh_positive expands.

Final fresh scalar-input integration explicitly splits ReLU gate intervals and intersects affine output margins; identically equal margins obey first-index argmax ties. When the target is last, that fixed target step is included in the scalar interval construction. No discontinuous argmax labels are silently interpolated as terminal data.

## Status
First controls complete; a target-quadrature failure is retained and the structural fix is frozen below. Root pushed claim42f60e1. T55 supplies derivation and T50 read-only mathematical review; this folder owns the distinct engine.

## Results
First cycle (checks.py, checks_results.json): six constant-readout controls on T3,T4,T7 have maximum error2.78e-17; six scalar terminal partitions have maximum defect2.22e-16, with duplicate logits correctly assigned to the first index. CPU.380seconds. The deterministic toy Wi=(1,0),Wh=0 selects the last label iff final input is positive, otherwise label0. Exact accuracy is1/T+1/[2(T-1)]-(T+1)/[(T-1)*2^T], unchanged by symmetric truncation. Global Gauss-Jacobi fails this sharp-boundary control: T4 errors at outer orders8/16/32 are-.01334/-.00773/-.00420; T7 errors+.002583/+.000906/+.000274. T3 happens to be exact and would have hidden the problem alone.

Before any trained run, freeze this structural repair: split target quantiles at q=.5 (the deterministic zero-input boundary under the symmetric law), using positive Gauss-Legendre on each half with explicit Beta density. Report actual2*n target nodes. This split depends on the known zero/ReLU boundary, not observed accuracy. Repeat all controls; preserve the failed unsplit results. Then bounded H2T7 LOCALPILOT tests, initial grid9,outer8 per half,input8 and one-axis refinements grid17/outer16/input16, without fitting settings to heldout truth.

Split-boundary checks (checks.py --target-rule split_sign, checks_split_results.json) pass: all nine sign-toy cases have absolute error<=1.11e-16; constants<=8.33e-17; terminal partitions<=2.22e-16. CPU.909seconds. T55 independently reviewed the engine's chronology, remaining-slot count, conditional recurrence, scalar margins and normalization, finding no algebraic discrepancy. Trained convergence remains untested at this checkpoint.

First trained cycle (pilot.py --seconds18, pilot_results.json), H2T7 LOCALPILOT checkpoint SHA09ac1bc640e3b15641a12606d91a4c28e699aa47fd14aff202bac52bfa0444e0:

| hidden nodes per axis | target nodes per half | input nodes | estimate | CPU |
|---:|---:|---:|---:|---:|
|9|8|8|.21248113|.308|
|17|8|8|.22475549|.811|
|9|16|8|.21254897|.543|
|9|8|16|.21243939|.322|

Existing heldout estimate.23706055 has SE.00234936; it is not a precise reference. No new samples were generated. The input-tail bound is1.38123e-8, but hidden interpolation is plainly unresolved: doubling hidden resolution changes the estimate by.01227, whereas target/input refinements change it by less than.00007. This is a negative result for the coarse uniform hidden grid. Finite stage bounds grow to approximately(25.24,28.35), making those uniform cells large.

Next bounded cycle is frozen before measurements: preserve the proved boxes but use fixed quadratic hidden coordinates B_t*(i/(n-1))^2 to resolve near-origin states. Compare hidden9/17/33 at target8/input8, plus uniform33 as a control. This changes mesh spacing only and uses no sampled states or truth-fitted allocation. Report all four outcomes and convergence rather than selecting whichever is nearest heldout accuracy.

Quadratic mesh cycle (mesh_cycle.py, mesh_results.json): h9=.22578459(.304CPU),h17=.23215131(.908CPU),h33=.23427521(3.183CPU). Uniformh33=.23162007(2.788CPU). Concentrating near zero improves this convergence sequence, but h17-to-h33 still moves.002124 and the coarse heldout reference cannot resolve small bias. No out-of-domain query occurred. Next prespecified checks: at quadratich33 separately double target and input rules; compare the same engine against the known H2T4 cone reference at h17/h33; no extra Monte Carlo.

## Compute accounting
Charge all target nodes, terminal gate/margin interval operations, conditional quantile transforms, hidden-grid construction, interpolation and rule setup; report CPU separately from operation counts. No new Monte Carlo or training. Each cycle<=120CPU with one thread,2GB cap and shared compute lock; checkpoints after coherent results. Large query arrays chunk below65536 points.

## Claims
[inferred] Conditioning removes recurrent target-coordinate interpolation. [speculative] Remaining2D interpolation may be accurate enough at lower cost. This is deterministic numerical transport, not an exact accuracy certificate.

## Alternatives not ruled out
1. An incorrect1/m law or missing1/T factor could make plausible probabilities. Constant readout, constant continuation and toy analytic controls test this before trained runs.
2. The apparent gain could hide terminal label smoothing. Explicit scalar decision intervals and first-index tie controls isolate this source.
3. Hidden-grid interpolation or target quadrature may still dominate bias. Refine one axis at a time and retain all results, including nonmonotone or negative findings.
4. Finite hidden boxes might clip expanding dynamics. Use stage-specific proved boxes and fail on out-of-domain queries; account separately for discarded input-tail mass.

## Failure modes
No certified bound for grid interpolation or outer/conditional quadrature. Gate arithmetic uses floats. Rare conditional tails need stable CDF differences. Positive-mass output ties cannot be discarded. Truncation guarantees alone do not certify total accuracy. The original challenge remains open.
