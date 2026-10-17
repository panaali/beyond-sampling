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
| [T32 tensor cross](../tensor_cross/README.md) | T81 independent contraction/TT-SVD/CUR controls and complete5472-face small grids; root source, scalar contraction and saved-error decomposition check | Fixed cross can miss rank1 support; tested ordering rescue helps4/8 and hurts4/8; full-grid bias is separate | Residual/local support enrichment and neural weight structure remain open. Arbitrary point-query rank1 worst-case obstruction is proved only for that class, not neural estimation | T81 distinct rescue Ready |
| [T34 surplus screen](../face_surplus/README.md) | T82 whole-stencil interior checks, independent face subset and same-count uniform-panel control; T55 independently recomputes summaries | Four finite slopes remain .827–1.079; geometric-knot split is mixed. A smaller physical stencil is not by itself an estimator improvement | Actual network-switch partitions and mixed regularity remain open; heuristic screening is not sparse-quadrature impossibility | T82 distinct rescue Ready |
| [T37 reference residual](../reference_error/README.md) | Independent reference/pairing/SE/source checks; eight frozen empirical residual laws miss by22.56–48.68pp; an earlier particular normalization has an exact divergent-moment witness | Global and position-conditioned empirical OUTPUT-residual independence fails for that reference/alignment and two finite scales | Hidden-reference models, richer continuous conditioning and different anchors remain open. Gaussian fitting cannot cure the already-tested lost dependence, but this is not a theorem excluding all residual laws | Existing T37 Ready |
| [T38 harmonic propagation](../harmonic_sphere/README.md) | T57 different projection reproduces six trained estimates<=5e-14; fixes ties/root normalization; missing dense-map cost documented | Tested isotropic variants remain poor; local corrections do not rescue them | T56 supported anisotropic basis improves H4 noisy RMSE .04009 to .01158 with lower CPU; one copy worsens. No sampling win/full portable cost yet | T56 cost/representation follow-up Ready |
| [T61/T65 coordinates](../small_h2_78/README.md) | Fixed factorials, range controls, checked primitives; T77 support and T83 reconstruction/jump diagnostics | Tested coordinate/mesh variants not converged; uniform cells lose shape and smooth reconstruction misses a true internal jump | Backward target splitting T67 and forward explicit interface T91 are distinct active rescues | T67/T91 active |
| [T67 fixed-target transport](../rank_conditioned_transport/README.md) | Analytic transitions/reuse agree; independent derivation; exact T4 controls expose quadrature/spatial cancellation | Tested mesh gains do not establish global accuracy; changing outer variable alone does not resolve oscillation | Interior grid-induced target kinks, split/analytical integration remain plausible; no blind resolution sweep | T67 self-claimed next phase |

The historical words "negative" and "done" must be read at the experiment's
declared scope. They do not establish that all variants have been exhausted.
In particular, existing shared-primitive verification does not independently
verify every algorithm that uses the primitive.

### Current checks demonstrating the distinction

- T52/T48 T6: old rules assigned zero to73,853 omitted positive simplices. Independent positive5D Jacobi omitted-mass estimate2.46270e-6 with1.07842e-7 conditional bound reconciles the full DD estimate within combined bounds. Both retained references and the failed apparent-discrepancy hypothesis are preserved.
- T76: extending last-two-input approximation to three inputs fails17/28 own-cost cases. Independent direct radial/S2 integration reproduces key failures; a nonzero history/affine formulation remains open. The cheap one-input rule's favorable tiny-budget ratio is not evidence of high absolute accuracy.
- T83: positive affine reconstruction improves early transport but later endpoints are mixed; log-linear is worse. Exact limits and one-sided numerical checks independently verify an internal density jump. T91 now tests explicit interface representation; its first negative is under investigation, not treated as proof the jump is irrelevant.
- T30: exact rational fixtures expose approximate-tie, lookahead and silent-cap bugs in specific legacy walkers. Local repairs and the corrected v2 Python backend pass. Tube's declared remainder is honest truncation, distinct from wrappers that discard it. Compiled backend remains untested.
- T56: the anisotropic rescue is positive relative to the isotropic baseline on fixed groups, but costs/full estimator independence and sampling comparison remain unverified. Reviewing negatives can reopen a direction without establishing a challenge solution.
- Root T81 source/summary check independently verifies scalar contraction (3.55e-15), low-rank reconstruction (1.78e-14), all four saved grid/cross bias decompositions and weighted error inequalities. This is not an independent neural walker or fresh full-model run;0.00182CPU, no model calls. Record: ROOT_T81_REVIEW.json.

### Recovery and subsequent scope checks

All394 previously unmerged worker/reviewer commits are now preserved on the shared branch; RECOVERY_MANIFEST.json verifies exact snapshots. Preservation is not a second scientific experiment. RECOVERY_REVIEW.md and the registry link the actual independent checks.

- T81 residual enrichment now includes global/local products and additive fibers. Independent Fraction contraction passes, while unseen support, off-fiber explosion and negative additive values remain explicit counterexamples. These reject particular guarantees, not neural-specific tensor methods.
- T82 actual network-switch slicing exposed floating orientation and inherited-halfspace defects; local repairs and controls are retained. Partition normalization helps3/4 cases but cannot certify correctness-dependent outer integration. Cost remains a partial proxy.
- T37 hidden-reference first order and final-step rescue fail both DEV models. A support-preserving rescue improves both, leaving one clearly biased and one inconclusive; no family-wide negative is justified.
- T20 cold proposal fitting is now charged. Its standard-deviation allocation is a heuristic for a mixture rule; independent review distinguishes it from the second-moment objective. Deterministic-node bias remains visible even with a corrected exact radial integrator. An iid objective theorem is not a deterministic quadrature error guarantee.
- T107 exact-zero native gate handling has independent Fraction and source review; repairs preserve trained arrays. Floating near-zero semantics, initial all-zero input rejection, and universal robustness remain separate limitations.
- T105 output forgetting can hold without hidden-state forgetting. Independent direct integration verifies the positive-event partial contribution. An explicit unresolved complement prevents any full-accuracy claim; signed-event extensions require disjointness and full rank coupling.
- T110 fewer initial class charts preserve exact model integrals on fixed small cases but change the approximation and can worsen it. Nearest-facet refinement is a distinct rescue. Root separately derives chart-sign masses from the binomial count of positive iid inputs using exact Fraction arithmetic: all T3..10 formulas and total partitions pass (check_class_sign_review.py, CLASS_SIGN_REVIEW.json). This verifies rank/sign measure only, not neural clipping or full cost. A quadratic initial partition does not make downstream ReLU splitting polynomial.

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
