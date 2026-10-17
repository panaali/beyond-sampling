# T181 stable Dirichlet quadratic variance

## Question
Can equivalent positive-square algebra reduce original-moment cancellation while preserving the reviewed H8T4 moment shortcut and its cost benefit? Claimed TASKS-only20c8287b before implementation.

## Method
For symmetric4x4 GramG, r=G1,d=diagG,s=sumG,t=trG and mu=(s+t)/20, Var(yTG y)=[sum_i(r_i+d_i−5mu)^2+sum_i<=j(Gij−mu)^2]/210. T55 independently expanded the identity before this phase. Reuse the same cached floating s,r,d and Gram entries; do not assume their exact relationships survive rounding. Only moment variance arithmetic changes. Same12 cone controls, four exact nearconstant probes, same64 T168 rows/target/ordercap/kernel/exact fallback.

## Status
Frozen control/source preparation.

## Results
Pending; original T168 fixed64 known109484 versus147062 without moments.

## Compute accounting
All changed scalar arithmetic and failed guards counted. Reconstruct second moment asvariance+mu² for diagnostic compatibility and charge two operations. Exact256 monomial references are validation work, never fitted estimator advice. One thread2GB<=20CPU review cycles.

## Claims
[inferred, independently reviewed] Exact equivalent variance identity and real-arithmetic interval proof. [speculative] Better floating stability. Positivity does not establish an outward floating bound.

## Alternatives not ruled out
Gram/mean/centering can still lose significant bits, extreme scaling can underflow, and stable formula may not improve actual trained inputs. Exact/interval arithmetic remains separate. No whole-family negative or full-cost/noisy-rung claim.

## Failure modes
Retain all nonpositive variance/reversed interval or finitecap failures; no clamping, source-reference fitting or cap/target change.
