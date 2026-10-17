# T26 controlled position patching

## Question
Do n3/n13 or the E29 fast-mode subspaces selectively mediate a position swap,
after matching intervention energy and immediate readout magnitude? Can the
partial hidden patches actually be reached by inputs? Claim051dd5a was pushed
before implementation. This is a mechanism audit, not an accuracy estimator.

## Method and frozen first phase
Public example checkpoint SHA256498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb.
Recreate only the first4096 examples of E29 seed82026, after its10000x10 discarded
pilot draws. Select examples whose global largest and second-largest inputs
both occur before step8 (zero-based positions0..7), maximum positive, and
recipient hidden state after8 steps has one of the fixed masks64975,63951,
65519,62927,65007. Preserve every selected pair; no correctness or outcome
selection. Form donor inputs by exchanging those two positions. Input suffix
positions8,9 remains identical. Full donor h8 followed by that suffix is a
genuine reachable intervention, with a known donor-input witness.

Compare recipient h8 + P(h8_donor−h8_recipient) for (a) coordinate pair3/13;
(b) the E29 orthogonal fast span, the two higher-frequency complex eigenpairs
with largest moduli above the dominant complex eigenangle in the recipient
mask. Use the same orthogonal SVD construction as E29; absent modes are an
explicit skip. These partial patches are NOT assumed on-manifold.

Controls are fixed before outcomes: all other coordinate pairs for(a), and64
orthonormal Gaussian rank-matched subspaces per active mask for(b), RNG26001.
Random controls lie in the same active-coordinate support as the fast span.
For each pair choose the control minimizing the maximum absolute log ratio of
(i) hidden patch L2 norm and(ii) immediate Wo-patch L2 norm, using no target
labels/logit signs or post-suffix outputs. A successful match requires both
ratios within exp(±log1.10). No rescaling, threshold tuning, dropped failures
or new controls after results. Report matched and unmatched fractions and all
target/control outcomes separately; insufficient matches mean inconclusive.

Resume the common two-input suffix for recipient, full donor, both partial
patches and selected controls. Outcomes: donor-target minus recipient-target
logit contrast, argmax switches to the donor target, recipient/donor correctness,
and paired mean contrasts/standard errors. No division by a small donor effect.
Report negative coordinates of every partial/control h8 patch, but nonnegativity
alone is not a reachability certificate.

Construct the exact recipient gate-cell affine map A:R^8→R^16 (zero bias).
For each desired partial displacement d, test least-squares residual of A dx=d
at fixed relative tolerance1e−10, and rank of (I−P)A. Full rank8 means no
nonzero same-cell input direction preserving all complementary coordinates.
Passing this linear test is necessary, not sufficient: check gate inequalities
by forwarding the least-norm witness prefix if residual passes. Failure proves
only local recipient-cell infeasibility, not global unreachability across all
gate cells. An independently reachable full donor state can still fail this
local test because its input lies in a different cell. Keep that control.

Numerical budget<=20CPU on review lock, single BLAS/OpenMP thread,2GB RSS.
Freeze before running; account all donor/recipient/suffix/witness forwards,
linear maps, decompositions, projection/control matching and setup. No third-
party code is executed or shared code edited. Independent algebra review
requested from T55; no self-confirmed causal specificity.

## Alternatives before the run
1. Larger state perturbation alone may drive apparent specificity: match patch
   energy, retaining match failures rather than comparing unequal interventions.
2. Larger direct readout gain may explain it: jointly match Wo-patch magnitude.
3. Partial patches may be outside the reachable manifold: test local linear
   feasibility and distinguish full donor positive controls from partial patches.
4. The label change may rely on nonlinear suffix gates: compare immediate
   readout magnitude used for matching with final outputs, without retuning.

## Status
Bounded partial-patch test and strongest reachable rescue complete;
**selective causal mediation remains inconclusive**. `RESULTS.md` records all
outcomes and limits. Independent T55 algebra review is recorded there; DRI
confirmation is not claimed. No neuron role or mechanism family is retired.

## Results
First frozen run:1014 selected pairs;626 coordinate-pair and279 fast-span
controls match both norms within10%. Partial target-minus-control donor-contrast
changes are−.5802 (SE.1401) and−10.7741 (SE1.9622). However every tested
partial direction fails the recipient-cell linear feasibility test, and274
fast patches have negative coordinates. Thus these are **not causal negatives**.
Full donor resumption agrees exactly. CPU1.13193sec including imports/loading,
RSS208772KiB; all results/costs retained in `results.json`.
Full-state swaps alone do not identify a neuron group's causal role.
Reachable minimum-leakage rescue has615/626 and279/279 positive-step witnesses,
all preserving gate histories. Median step sizes.033/.005 are small and require
substantial complementary changes; only306/7 positive-step norm matches survive.
No witness completes the donor input order. Detailed costs/protocol and all
initial/rescue artifacts are retained; total numerical CPU3.26037sec.

## Compute accounting
CPU/RSS and operation/forward/decomposition counts will be retained. Random
subspace generation is a diagnostic control, not random model sampling for an
estimator. Existing E29 cohort regeneration is explicitly charged.

## Claims
[inferred] Full donor h8 has a reachable input witness with the common suffix.
[inferred] Nonnegative partial states need not be reachable.
[speculative] Selective mediation could survive matched controls; no favorable
group outcome is assumed or used to choose controls.

## Alternatives not ruled out
Other position swaps, times, gate histories, global input witnesses, conditional
resampling, exact constrained reachable interventions and distinct trained
models. An inconclusive matching test is not a negative causal theorem.

## Failure modes
Natural matched controls may be scarce. SVD feasibility is numerical, not exact
certification. Local-cell impossibility does not imply global unreachability.
Matched immediate readout norm does not equalize nonlinear suffix sensitivity.
Partial patches can be off-manifold even if all coordinates are nonnegative.

## Frozen reachable-tangent rescue (before second execution)
Use the same1014 pairs and ONLY the first run's successfully matched controls;
do not regenerate the matching choice or select outcomes. Reconstruct those
fixed subspaces deterministically and count this setup. For each target/control
Q and desired group coordinates b=Q^T(h_donor−h_recipient), let U span the
recipient-prefix gate-map range. The minimum-norm reachable hidden displacement
preserving those group coordinates is d=U*(Q^T U)^+b, provided the equality
is feasible. It minimizes complementary leakage because ||Q^T d|| is fixed.
Solve A8 dx=d, then compute a safe COMMON target/control step alpha<=.9 from
all preactivation inequalities at every one of the10 original steps (including
the common suffix). Zero preactivation forms are treated explicitly; zero
safe-step pairs are retained as failures. No assertion of strict feasibility
on identically-zero forms. This produces actual perturbed input witnesses,
not arbitrary hidden vectors; check full forward reconstruction and gate signs.

Report unavoidable complement/target norm ratio, retained fraction alpha,
group-coordinate equality residual, witness errors and donor-order changes.
Evaluate target/control donor-logit contrast change using the same common alpha;
also report derivative changes divided by alpha only for alpha>0. These latter
are local sensitivities, not a completed donor-position swap or isolated-group
mediation. Do not rematch readout energies after this rescue; instead report
how many original10% matches survive. If they fail, state that the first matched
test and the reachable test answer different questions. <=20CPU, same lock,
no new model inputs/cohorts beyond deterministic reconstruction.
