# T107 native exact-zero gate branch

## Question
Can the H2T5 native DD lifecycle support identically zero non-leading gate
forms without changing ordinary trained topology or positive-volume argmax ties?
Claim b7acf51 was pushed before source copies/protocol. DRI owns shared code;
all modifications are local to this folder.

## Method and frozen controls
Pin the existing D5 native geometry/clip/wrapper/build/reference files in
_source with hashes. Add only a coefficient-exact zero test to each non-leading
gate form. If all coefficients equal zero, keep its gate OFF and keep the cone
unchanged, marking this identically tight constraint on every ray. This ensures
future apex allmask remains consistent; no division by a zero norm and no
geometric split. Preserve leading all-zero-Wi rejection and all near-zero,
rank, normalization and readout predicates. Record extra zero-test coefficients
and zero branches separately from existing counts.

Separate Python reference adds the same mathematical zero/off semantics before
its clip call. Compare ordered headers/incidence/rays/maps/simplices for
supported new cases and retain all original13 counters; new native counters
are extra. Exact fixed-map recurrence at representative points independently
checks the branch/gates. Incidence bits for a zero form are redundant on all
rays; they cannot introduce a facet in pulling triangulation, which skips every
constraint incident to its chosen vertex.

Freeze controls Wi=(1,0) and its permutation, Wh=0; readout winner4 for positive
last input, winner0 at zero has exact accuracy89/320. Zero/duplicate readouts
have accuracy1/5. Additional recurrent control Wh=[[0,0],[1,0]] tests gate forms
that are zero only on some prefix cells. Both allzero Wi remain explicitly
unsupported, malformed/nonfinite weights still rejected. Small nonzero Wi2
at1e-8 remains a distinct near-zero control and must match unmodified native
behavior; no threshold-based classification as exactzero. Positive hidden
coordinate scales1e-6/1e6 preserve topology where old rules permit it.

First build original and modified local binaries, charge both builds, test
synthetics only. After controls, original trained H2T5 seed0 and the existing
RNG1 relative-RMS.0025/.02 copies compare all saved geometry arrays with the
unmodified native implementation; no new mass algorithm or noise sweep.
Independent T55 source review before adoption. <=110CPU cycles,2GB, one thread,
primary lock; <=20CPU primitive checks may use review lock.

## Alternatives and falsifiers
1. Skipping a zero form without incidence propagation corrupts later adjacency:
   compare Python ordered incidence plus constant-model orthant volume checks.
2. Exactzero predicate might absorb tiny meaningful gates: freeze nonzero1e-8
   control and leave underflow/norm rejection behavior explicit.
3. Off branching could mishandle positive-mass output ties: zero/duplicate
   outputs use first-index winners and analytic1/5 controls.
4. Source parity alone can share a bug: directly evaluate every final gate-map
   interior against recurrence and use independent analytic accuracy controls.

## Status
Protocol frozen. No repair/accuracy claim until controls pass.

## Results
Pending; original rejection preserved as a required control.

## Compute accounting
Count compiler/setup, original/native/reference execution and all extra zero
checks. Existing native operation cost is opaque; do not inherit a portable
sampling-win counter from old Python. No estimator inputs sampled; deterministic
ray/interior evaluations are validation and fully counted.

## Claims
[speculative] An exactzero/off branch fixes the safe rejection without affecting
trained cells. Exactzero comparisons concern computed floating coefficients;
this is not a general exact-real arithmetic topology certificate.

## Alternatives not ruled out
Allzero Wi support, certified predicates/rank and portable native cost remain
separate extensions. No H4/H16 or dimension extension in this task.

## Failure modes
Nearzero norm underflow, inherited1e-11 clip and1e-9 rank thresholds, platform
BLAS differences, coincident facets, uncharged native operations and compiler
first-use overhead. Repeated parity is not universal correctness proof.
