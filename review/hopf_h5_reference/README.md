# T11594 fixed independent H5T4 reference precision

## Question
Do the unchanged original H5T4 Hopf estimates beat sampling under the reviewed
conditional source-cost convention when diagnostic truth is more precise?

## Method
Claim b55c0779 follows completed native-cost handoff8f001bcd, independently
79f9bef8/d0458af9. Freeze all25 existing model, output and cost identities,
fresh distinct CRC32/PCG64 streams and exactly100M draws/model before draws.
Use the exact original mc_acc body with only final return changed to integer
hits. First deterministic forward/chunk parity, then clean0 first1M throughput
pilot and independent saved gate before expansion. No estimator selection,
parameter changes, old10M combination or H4 truth transfer.

## Status
All25 fixed references complete and independently2695e545 verified.
All60group/50individual fresh comparisons independently282a2788 verified. No further random draws are authorized by this completed phase.

## Results
The prior60group/50individual original75-family10M intervals all contain1;
8group/5individual point losses are retained. This is not a method rejection.
New25-truth nominal95% simultaneous intervals will be separately reported.
Deterministic controls: all six original/integer/direct recurrence and chunk
checks pass,318 extra forwards,0 random draws,.010051108CPU.
Command: `python review/hopf_h5_reference/reference.py --controls`.

Pilot:640411/1M hits,.240898577 forwardCPU, projected602.2464425<=900.
Command: `python review/hopf_h5_reference/reference.py --pilot`.
The pilot is reused once; its accuracy does not determine expansion.

Fresh result:42/60 group and35/50 individual upperratios are below1;
18/15 remain inconclusive, none has lower bound above1. Eight/five point
ratios remain>=1, max1.774335641. Largest group upper2.895198010.
Optimized matched pooled upperratios byseed0..4: .404154243,.901702064,
.511686932,.637346630,1.669985866. Therefore no all-panel H5 pass or
method rejection. These are conditional source-cost comparisons, not
actual-runtime failure/success proofs or mechanistic classification.

## Compute accounting
One thread2GB; primary105/110CPU perprocess, shared fair lock. Fixed pilot
forward CPU x2500 must be<=900CPU before expansion; a scheduling projection,
not a total runtime ceiling. Source/import/controls/IO and actual block CPU
are separately retained. Completed2.5B diagnostic draws,318 extra deterministic
forward evaluations. Forward619.695751504CPU; recorded research630.019108551,
max continuation process26.589146916. Summary.237497972CPU and comparison
.175528142CPU are separate. Pilot/control imports excluded from recorded
research timer; continuation process imports included. No failed attempt,
repeat or production-estimator evaluation. No accuracy adjustment.

## Claims
[measured] Full fixed25-model coverage and all saved references pass independent
review2695e545. [measured, independently282a2788 checked]42/60 and35/50
conditional comparisons certify upper<1; the remainder are inconclusive.
[measured] The reported maximum point ratio decreases from3.157472189
to1.774335641 with the fresh references. Current precision still does not
establish the whole panel.

## Alternatives not ruled out
An apparent failure may reflect real quadrature bias or old-reference noise;
fresh independent data separates them. A source/model drift can imitate
accuracy change; pinned identities and original/integer forward parity pass.
Higher resolutions and corrected root-mask candidates remain distinct untested
variants here. Tighter valid confidence contractions or separately frozen fresh
reference data may resolve remaining uncertainty; no optional extension with
unchanged fixed-N confidence interpretation. Remaining native-cost closure
could improve source allowances; no existing trace multipliers may transfer.

## Failure modes
Source/pin/shape mismatch, forward-parity failure, resource-gate failure,
changed RNG/state chain, incomplete blocks, numerical confidence inversion,
floating-forward semantics and existing conditional native-cost exclusions.
The original Hopf root-mask defect remains unchanged; no generic correctness
or full-runtime/challenge result follows from this fixed panel.
