# Pinned author preregistration and interpretation checklist

The sole tracked file under m16/approaches/random_control at author revision
1d6f4db95e90c1159ec0cb8d420344648dd0c7ca is README.md, copied byte-for-byte in
snapshot/AUTHOR_README.md; manifest.json retains Git blob/SHA/length. No source
implementation, draw archive or result is present IN THAT FOLDER AT THAT
REVISION. This does not establish absence of ongoing work elsewhere/later.
Original branch history is not imported by this snapshot and no historical
execution provenance is inferred. There are zero author calls or array decodes.

The preregistration fixes five global-Frobenius Gaussian draws (seeds100..104),
five row-norm Gaussian draws (200..204), within-row recurrent permutation300,
and entrywise recurrent sign flips301. It requests target comparisons k2..4,
conditional k5 if cheap, k4 mass coverage and1e6-input diagnostic accuracy.
These are author plans, not independently verified calls/results. This review
neither changes that plan nor authorizes its execution on this machine.

Before numerical evidence is independently accepted, retain:

- Exact input/recurrent/readout draw arrays or equivalent complete reproducible
  byte bindings, generator/version, target checkpoint, all positive/zero norm
  cases, source dependencies, branch/CPU cap outcomes and every failed draw.
- Distinct complete gate histories versus triangulated cells/fan pieces,
  boundary/zero-gate ownership, deduplication semantics and feasibility/
  completeness limitations. Existing target34/792/21032 verification does not
  automatically validate random-model floating geometry.
- Exact mass source/rule/error assumptions and normalization. A table of
  numerical masses supports table coverage, not automatically true N_delta.
- Accuracy hits/trials/stream IDs and independent per-model uncertainty,
  especially all-OFF first-index ties.1e6 is a diagnostic, not directive7 rung
  truth. No comparison with a trained readout can isolate recurrence alone when
  the random readout was also changed.
- Group-specific five-draw summaries as well as any preregistered pooled median
  across ten draws. The two normalization ensembles are different distributions.
  The stronger-decision N_1e-4 comparison should name its aggregation explicitly
  before seeing results; the text names a median for growth but is less explicit
  for mass coverage. Small groups/descriptive bands are not confidence claims.
- The phrases 'strong' and 'within30%' are frozen descriptive rules; 'no training
  effect' should not be elevated to a null-effect theorem. Random networks with
  lower growth indicate another direction of difference, not a universal result.
- Runtime-triggered k5 inclusion is a selected subset. Report failures/omissions
  and do not estimate the full random ensemble's typical k5 growth from only its
  cheap members. No finite-depth measurements prove a T10 multiplicative law.

Two competing explanations and discriminators are already visible without new
calls: (1) norm matching changes probabilities but not globally rescaled counts
(PROOF A), distinguish per-model counts from coverage rather than conflating
metrics; (2) a depth-two value34 is generic (PROOF B), so deeper comparisons are
needed before attributing that count to learned structure. Source/rule errors,
readout replacement and selected cheap-depth inclusion remain additional
alternatives. This is not an exhaustion claim or a rejection of the random
control, which addresses a useful previously unmeasured comparison.
