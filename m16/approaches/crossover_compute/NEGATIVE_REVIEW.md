# Negative-result review inventory (T19)

## Question

Which failed experiments justify rejecting their tested variant, and which still
need independent checks or plausible rescue tests before a method is retired?

## Method

Read the saved protocols, source, results, independent-review records and current
task ownership. This is a scope/evidence audit, not a rerun of all experiments.
Two alternatives guide the audit: an apparent scientific failure may be an
implementation or numerical defect; a correct negative for one representation
may leave a useful alternative representation untested. The table records the
cheapest discriminating follow-up rather than assuming either explanation.

## Status

Running. No entire method family is declared dead by this audit. Completed
bounded studies remain completed experiments; unresolved alternatives remain
in the backlog. Claude alone confirms claims or marks a scoped negative verified.

## Results and reproduction

Sources are linked below. No new numerical experiment is asserted in this file.
The audit distinguishes independent arithmetic/source review, independently
verified shared primitives, and an independently implemented end-to-end test.
These are different levels of evidence.

| Study | Evidence already present | What the negative actually supports | Missing check or plausible rescue | Next task |
|---|---|---|---|---|
| [T32 tensor cross](../tensor_cross/README.md) | Eight frozen rank2/4, order3/5 settings; author rerun after independently reviewed walker fix gives identical estimates; exact rank2 synthetic tests | That fixed cross implementation misses mass on two DEV models; 2.72–23.51 percentage-point errors exceed reference noise | Independent TT contraction/pivot implementation; separate exact-grid error from interpolation; variable order, local fibers and residual enrichment remain untested | T81 Ready |
| [T34 surplus screen](../face_surplus/README.md) | Frozen finite-level screen; independent source/summary review caught midpoint-selected interior stencils; sensitivity arithmetic retained | No tested case passes the chosen heuristic screen in those coordinates; this is not a sparse-grid estimator failure | Strict whole-stencil interior review and small independent mass subset; boundary-aware coordinates/local partitions still open | T82 Ready |
| [T37 reference residual](../reference_error/README.md) | Independent reference/pairing/SE/source checks; eight frozen empirical residual laws miss by22.56–48.68pp; an earlier particular normalization has an exact divergent-moment witness | Global and position-conditioned empirical OUTPUT-residual independence fails for that reference/alignment and two finite scales | Hidden-reference models, richer continuous conditioning and different anchors remain open. Gaussian fitting cannot cure the already-tested lost dependence, but this is not a theorem excluding all residual laws | Existing T37 Ready |
| [T38 harmonic propagation](../harmonic_sphere/README.md) | Author random-instance controls; multiple degree/branch variants; observed readout bias and counted costs | Tested isotropic truncations lose at equal operations on recorded models | T57 independent implementation underway; fresh zero/duplicate-readout controls reportedly expose tie handling and non-unit projected root mass, pending committed review. Anisotropic basis is an explicit untried rescue | T57 active; T56 active |
| [T61/T65 coordinates](../small_h2_78/README.md) | Fixed factorial grids, range/constant controls, independently checked terminal primitive; author estimates not converged | Angular alignment failed at tested grids; both target and hidden interpolation can matter | No independent full transport implementation. Fixed-target formulation and support-aware forward transport are genuine alternatives, not ruled out by these results | T67 Ready; T58/T77 active |
| [T67 fixed-target transport](../rank_conditioned_transport/README.md) | Analytic transitions and reuse agree; independent mathematical review; exact T4 controls expose quadrature/spatial cancellation | Tested spatial gains do not establish global accuracy; switching outer variable alone does not resolve oscillation | Interior grid-induced target kinks, split integration or analytical target integration remain plausible; further blind resolution sweeps are not a substitute for diagnosis | T67 Ready/inconclusive |

The historical words "negative" and "done" must be read at the experiment's
declared scope. They do not establish that all variants have been exhausted.
In particular, existing shared-primitive verification does not independently
verify every algorithm that uses the primitive.

### Current checks demonstrating the distinction

- T52/T48 T6: the two older converged estimates both assigned zero to73,853
  omitted positive simplices, with upper omitted mass5.00327e-6. A full DD
  estimate being2.16082e-6 higher is compatible with that accounting, not yet a
  contradiction between full integrals. An independent omitted-mass-only rule
  is running; do not treat either old midpoint as exact truth.
- T77: support-aware projection eliminates the toy's forbidden axis mass,
  independently justified by an exact reachable-cone recursion. Some endpoint
  errors improve while some stratum errors worsen. Higher source quadrature
  does not remove the later bias; positive within-cell density reconstruction
  is a specific open rescue, not grounds to retire forward transport.
- T76: initial two-input suffix results are positive on12 existing references,
  but broader perturbation/training-seed checks fail on some models. The
  approximation, operation counts and a cheaper one-input control require
  independent review. These observations are neither a general winner nor a
  general impossibility result.

## Compute accounting

This inventory uses existing text, source and results; no new model evaluations,
mass integration or fitted parameters. Worker checks are charged in their own
task records. Public H16T3 seed0/seed1 checkpoints were downloaded for T57;
hashes and actual numerical costs belong in that independent review.

## Claims

- [measured, documentary] The cited studies contain unresolved alternatives and
  different levels of independent verification.
- [inferred] Closing the finite study is justified where its frozen endpoint was
  reached; retiring its broader method family is not justified by these records.
- [speculative] The listed rescues may help. Their presence in the backlog is
  not evidence of success.

## Alternatives not ruled out

The checklist is finite and justified by current failure mechanisms. It is not
an enumeration of every conceivable algorithm. A broad impossibility statement
needs a theorem with explicit assumptions, not a longer parameter sweep.

## Failure modes

Documentation may lag a worker's latest result; follow task ownership and pinned
commits. A shared oracle can hide common bugs. Review can confirm a limited
negative without validating the proposed rescue. Neither sunk compute nor a
desire to keep workers busy justifies duplicating work or omitting resource caps.
