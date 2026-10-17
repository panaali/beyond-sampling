# Bounded H4T4 bulk-mechanism handoff

Ownership claimf51b14a was pushed first. Ordered research commits:
7c60c2f,e60763b,88d6f90,408592d,f799183,f08cd45,2a99806,8f216e2,36780af,
2836d69 (the last also records T49 review). This handoff follows those commits.
All implementation is local; T68 artifacts and geometry/mass sources are read-only.

Fixed original H4T4 checkpoint only, knownaccuracy.477013125994119:
- Exact readout primal/dual witnesses show all4 classes possible in h>=0.
  Independent Fraction check passes all20 frozen rows. Actual recurrence
  witnesses subsequently establish all4 classes genuinely reachable.
- Exact full/correct simplex attribution: final/earlier all-off account for
  only4.3409% of error; no no-reset exact output-contrast forgotten column.
- Stronger rescue:166 additional gate interiors have constantoutput, covering
  .2746298312 probability and .1351190692 error. With resets this explains
  ~30.18% of error;267 decision-varying histories retain .3651656586 error.
- Whole-gate sufficient correctness rescue certifies87/450 histories but only
  .005750413 probability. The remaining width.994249587 is uninformative.
  This fails as a standalone shortcut, not a proof against stronger certificates.

Independent T55 source/Decimal reviews validate attribution; direct integer
recurrence verifies all450 output certificates on4252 uniqueperhistory rays,
with actualinput witnesses foreachclass. Whole-correctness logic/raw-sum review
is still queued at time of this handoff. DRI acceptance/classification pending.

Failures retained: initial exact determinant accumulator exceeded Python decimal
string limit after classification (5.2164CPU). Hex output repairs serialization
only. Generic readout solver has no infeasibility certificate when normalized
weak-later constraints are infeasible; it raises, a limitation absent on the
frozen cases. No floating threshold defines erasure, tie or class reachability.

Recorded completed phases use.1547+6.2399+1.9346+2.4180CPU; failedrun is extra.
Historical T68 cold reference cost10.0096CPU is separate. All mass quadrature
bounds remain conditional on floating arithmetic; this is a diagnostic, not
an estimator speed/cost win. No noisy evaluation or passed rung is claimed.

Exhaustion checklist: global missing-readout, exactalloff/contrast erasure,
constant-outputwholegate interiors and coarse possible-label certificates have
been tested; approximate attenuation, exactsimultaneous rank/readout feasibility,
rank-conditioned certificates and cross-gate unions remain untested/open.
T110 worker34 owns orderedtop-two conditioning; do not duplicate its engine.
The strongest next bulk recipe should condition on rank or merge common-output
regions with an explicit cost/accuracy discriminator. No automatic fullrung
sweep or larger model is justified by these partial diagnostics.
