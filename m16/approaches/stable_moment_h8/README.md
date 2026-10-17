# T181 stable Dirichlet quadratic variance

## Question
Can equivalent positive-square algebra reduce original-moment cancellation while preserving the reviewed H8T4 moment shortcut and its cost benefit? Claimed TASKS-only20c8287b before implementation.

## Method
For symmetric4x4 GramG, r=G1,d=diagG,s=sumG,t=trG and mu=(s+t)/20, Var(yTG y)=[sum_i(r_i+d_i−5mu)^2+sum_i<=j(Gij−mu)^2]/210. T55 independently expanded the identity before this phase. Reuse the same cached floating s,r,d and Gram entries; do not assume their exact relationships survive rounding. Only moment variance arithmetic changes. Same12 cone controls, four exact nearconstant probes, same64 T168 rows/target/ordercap/kernel/exact fallback.

## Status
Fixed controls and64 pilot complete after independent source gate; saved-data review requested.

## Results
All64 frozen cases return with25 moment shortcuts; known subtotal109414 versus original-moment109484 and no-moment147062. Detailed errors, variance validation and limitations are recorded below.

## Compute accounting
All changed scalar arithmetic and failed guards counted. Reconstruct second moment asvariance+mu² for diagnostic compatibility and charge two operations. Exact256 monomial references are validation work, never fitted estimator advice. One thread2GB<=20CPU review cycles.

## Claims
[inferred, independently reviewed] Exact equivalent variance identity and real-arithmetic interval proof. [measured] Improved variance accuracy in34/35 frozen trained moment attempts and two formerly zero synthetic probes; one trained attempt slightly worsens and the smallest synthetic probe remainszero. Positivity does not establish an outward floating bound.

## Alternatives not ruled out
Gram/mean/centering can still lose significant bits, extreme scaling can underflow, and stable formula may not improve actual trained inputs. Exact/interval arithmetic remains separate. No whole-family negative or full-cost/noisy-rung claim.

## Failure modes
Retain all nonpositive variance/reversed interval or finitecap failures; no clamping, source-reference fitting or cap/target change.

Controls7b57fc65 preserve6 returned/6 finite unresolved. Nearconstant variances at eps2^-12 and2^-20 recover8.628018934e-17 and2.008867211e-26 instead of0; relative errors9.8e-18 and4.67e-9. eps2^-28 remains0 and disables the shortcut, consistent with prior Gram/mean rounding loss. No variance clamp or cap increase.

Fixed64 pilot retains29 center intervals,25 moment intervals,9 quadratures,1 exact rescue; all return and39 non-moment values/bounds stay bitwise baseline. Named subtotal109414 versus109484 original moments and147062 without moments. Savings70 additional units exactly equal2 arithmetic saved across35 attempts; total reduction25.60% versus no moments. Conditional bound1.80721668098e-11, reference-error L1 3.81342583240e-12; no fullcorpus extrapolation.

Separate exact256-monomial reference validation on35 attempted moments costs0.037536CPU and never enters estimator selection. Maximum relative variance error improves1.139710236e-7→5.087941447e-13; median1.17223e-10→2.26073e-14.34 improve; one worsens: index32868, original−9.64e-17 versus stable−2.94e-16 relative error. All raw exact fractions and both values are retained; positivity does not guarantee every individual floating value improves. Pilot CPU0.290773 includes validation/serialization, not a clean runtime benchmark.

Reproduction, only if needed: one-thread2GB review-lock Python `controls.py` then `pilot.py`. Existing saved records permit independent verification without rerunning integrations.
