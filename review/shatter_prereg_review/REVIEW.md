# Independent T239 preregistration reconciliation

This review freezes owner commit `1cd671cc2e19b59f6fafe7a9ada34a7a72f80923`
and its sole `m16/approaches/shatter_test/README.md`. It updates the earlier
T242 frozen absence of a dedicated T239 implementation. At this commit the
folder has a preregistration, not an implementation or result archive. No owner
code/data/arrays/checkpoints have been imported or evaluated here. The owner's
capture plan and numerical budget are unchanged; this is not execution clearance.
The commit establishes the written plan existed then; it cannot independently
prove the statement that no T239 number had previously been computed.

## 1. Useful advances over the earlier inventory

The plan acknowledges that old T206 archives lack ray coordinates and that
T192 does not supply the requested per-leaf H16T5 data. It therefore explicitly
limits its trained T comparison to3→4 and proposes new captures retaining rays.
It also specifies q=.9,.99,.999, two widths, seed0, numerical support/disproof
thresholds and alternatives rather than choosing thresholds after this review.
The planned random/control prefix comparison is separate from labeled final
leaves. These are concrete improvements in scope. A new capture can provide
new auditable evidence; it does not reconstruct the missing historical T232
random-weight realizations or prove their earlier runtime identity by itself.

## 2. What coverage by correct/error contribution measures

Let the declared disjoint final activation leaves be C_i, with m_i their true
probabilities, e_i=P(C_i∩error), c_i=m_i−e_i. The owner's N_e(q),N_c(q) are
meaningful when each total denominator is positive and all required leaves
are present. A mixed leaf can contribute to both lists. These are coverage
counts by activation-leaf contribution, not necessarily counts of uniformly
wrong/right decision regions. Calling the error mass or c=m−e 'exact' needs
qualification: exact-mode gate enumeration is not an exact Gaussian numerical
integral, and c subtraction can magnify errors in rare correct/error pieces.

Retain m,e,c, all negative/nonfinite/out-of-range values, normalization and
unresolved leaves, mass-kernel source/tolerance and any actual bounds. Existing
T242 interval/ordering arguments apply if certified bounds are available;
otherwise support/disproof thresholds refer to numerical diagnostics with
explicit integration uncertainty. No new mass evaluator is authorized here.

Equal q makes the coverage fraction class-conditional. It does not make the
absolute omitted masses equal: they are (1−q)P_E and (1−q)P_C. N itself is not
'cells per unit mass'. Reporting class totals and omitted masses alongside N
avoids treating the sentence '(1) normalises per unit mass' as an additional
mathematical normalization. Neither conditional N nor N/P is an intrinsic
complexity or randomness certificate; arbitrary subdivisions can change them.

The support rule can be written explicitly for each H as

    R_H = N_e(H,4,.99)/N_c(H,4,.99),
    G_H = [N_e(H,4,.99)/N_e(H,3,.99)]
          /[N_c(H,4,.99)/N_c(H,3,.99)].

The text requests R_H≥3 for both widths and growth-factor ratio≥1.5. It should
state whether that latter condition is also required for both widths. The
'disproves if N_e≤N_c or N_e grows no faster' sentence does not yet say whether
one width, both widths, or an aggregate triggers the global verdict. Preserve
per-width support/disproof/mixed results and resolve that quantifier before
combining them. Zero denominators remain undefined, not favorable ratios. The
q=.9/.999 outputs are descriptive unless their separate decision role is fixed.
These observations do not invent new thresholds or reinterpret owner results.

Different zoo T3/T4 checkpoints are different trained functions. A measured
ratio is a cross-checkpoint comparison, not the next-step growth of one fixed
network; the target-prefix controls are a separate design. One seed per width
cannot establish a training-distribution or architecture-wide law.

## 3. Center-assigned bins are a different statistic from gap-event mass

The proposed representative is a normalized mean of unit generator rays. This
removes positive scaling of an individual generator, but is not generally
invariant to its multiplicity or to adding a redundant interior generator.
For the same first-quadrant cone, lists (e1,e2) and (e1,e1,e2) produce distinct
normalized means proportional to(1,1) and(2,1). Even with duplicate removal,
adding a redundant interior ray changes the average without changing the cone.
Actual archived extreme-ray uniqueness and representation conventions can
reduce this ambiguity; they must be recorded, not presumed from the word 'rays'.

For unit u and a tie plane xi=xj, distance to the plane is
|u_i−u_j|/sqrt(2). Angular distance on the sphere is
arcsin(|u_i−u_j|/sqrt(2)). Thus the owner's d is a sine/angular-distance proxy,
not the angle itself. Keeping the original numeric d thresholds while labeling
the statistic precisely avoids silently changing the preregistered bins.

Assigning an entire leaf's mass according to this center defines a valid
REPRESENTATIVE-BIN partition statistic, if center/ray identities are fixed.
It does not compute P(C_i∩{true normalized gap in bin}). Gap values generally
vary inside a cone; a center can be far from a boundary that some directions
approach arbitrarily closely. The same concern applies to blind classification
unless audited chart/sign constraints establish whole-leaf ownership. This
is not an allegation that every actual blind label is wrong: the implementation
and archives have not been supplied or examined in this review.

Consequently 'cells per unit mass' based on center bins is the number of
assigned leaves divided by their assigned whole-leaf mass. It cannot be called
actual probability density near ties or a certified correct/error mass of that
event without membership/intersection evidence. A cheaper diagnostic may retain
the center rule under its explicit name; exact event intersections would be a
different, separately costed experiment. Neither is automatically authorized
by this review. Do not silently use a convenient representative to claim the
more expensive event-based estimand.

Define blind as either a separate marginal row with overlap disclosed, or an
exclusive category and report nonblind gap bins separately. 'Plus blind' alone
does not settle whether totals may be summed. Pin treatment of zero vector sum,
zero/invalid rays, threshold equality, empty far bins and zero mass denominators.
The original d<.05 versus d≥.1 decision can be reported unchanged once these
semantics are fixed. The gap interval[.05,.1) is intentionally outside that
contrast, not missing probability to be silently reassigned.

## 4. Random comparison and competing explanations

At prefix k4 the proposed comparison has no output labels. It can ask whether
the declared representative-bin fragmentation profile differs between a target
prefix and random/scrambled controls. It cannot directly establish that the
ERROR part of the target resembles the error part of a random network, because
neither error set is computed by that prefix statistic.

The verbal prediction ('trained near density approaches random, far density
lower') is stronger than its one ratio criterion. If D_F/D_N for a trained
model is at most one third the control median ratio, this alone does not show
closeness of D_N across models or a smaller absolute D_F. Multiplying both
trained densities by any positive factor preserves the ratio. Report the two
absolute assigned densities and their ratio; do not promote ratio success into
both verbal claims. This algebra does not predict any actual result.

Specify which of the12 controls enter the median, whether frob-matched and
row-matched draws are pooled, and how sparse/zero far/near cases are handled.
They are distinct interventions. Same source, start partition and thresholds
help control geometric artifacts but do not causally isolate training: hidden
weight distribution, realized scale, region shape, learned function and
label/accuracy law can change together. Pin actual new weight bytes, seed/
construction details and source/runtime identity rather than assuming the old
aggregate-only T232 records suffice. Retain every declared control and failure.

Two explanations remain live: (a) error locations are more fragmented under
this particular enumeration; (b) rarity, tie-plane geometry, center assignment,
start refinement and different trained functions create the observed ratios.
The fixed support/disproof rules test selected operational statistics. They do
not prove or disprove architectural capacity exhaustion, randomness of the
error function, impossibility of aggregation, or performance of all estimators.

## 5. Minimal later evidence contract

Before interpreting newly produced numbers, retain immutable model/source/
runtime identifiers; the counted unit/start charts; parent/history/leaf IDs;
actual ray matrices and multiplicities; original m/e returns and uncertainty;
whole-leaf versus intersection-bin semantics; complete/partial termination and
failed cases. Keep labels/masses for final leaves separate from unlabeled prefix
controls and partition count separate from representation-invariant complexity.
A numerical audit should verify archive binding and conservation first, then
any claimed coverage threshold/ordering, and only then interpret the preregistered
criteria. This source review supplies no program correctness or runtime evidence.

Owner T239 remains an informative bounded diagnostic if conclusions track the
chosen statistic. The principal actionable distinction is center-assigned mass
versus actual gap-event mass, together with exact-mode versus certified masses.
No experiment is declared failed or dead and no owner code/plan is changed here.
