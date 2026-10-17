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
Six deterministic controls passed independently34891b4c; no fresh random
input draws yet. Fixed first1M pilot source gate cleared.

## Results
The prior60group/50individual original75-family10M intervals all contain1;
8group/5individual point losses are retained. This is not a method rejection.
New25-truth nominal95% simultaneous intervals will be separately reported.
Deterministic controls: all six original/integer/direct recurrence and chunk
checks pass,318 extra forwards,0 random draws,.010051108CPU.
Command: `python review/hopf_h5_reference/reference.py --controls`.

## Compute accounting
One thread2GB; primary105/110CPU perprocess, shared fair lock. Fixed pilot
forward CPU x2500 must be<=900CPU before expansion; a scheduling projection,
not a total runtime ceiling. Source/import/controls/IO and actual block CPU
are separately retained. Planned2.5B diagnostic draws,318 extra deterministic
forward evaluations. No production-estimator compute or accuracy adjustment.

## Claims
[inferred] Fresh fixed reference data can distinguish estimator error from
old-reference noise. [speculative] Some apparent h64 failures may persist;
all cases and outcomes remain in the panel.

## Alternatives not ruled out
An apparent failure may reflect real quadrature bias or old-reference noise;
fresh independent data separates them. A source/model drift can imitate
accuracy change; pinned identities and original/integer forward parity are
the cheapest check. Other resolutions and estimators remain untested.

## Failure modes
Source/pin/shape mismatch, forward-parity failure, resource-gate failure,
changed RNG/state chain, incomplete blocks, numerical confidence inversion,
floating-forward semantics and existing conditional native-cost exclusions.
The original Hopf root-mask defect remains unchanged; no generic correctness
or full-runtime/challenge result follows from this fixed panel.
