# T251: actual T241 phase-A source review

Frozen author revision: `a24a544cf9a259884928bb08c41aaa85b01284bd` on
`claude/t241-neuron-account`. Claim `01c26f9a`; independent written reviewer
recover_fundamentals delegated in `3a63641c`. This reviews five text files,
not their numerical outputs. The T244 contract and independent review
`bd27e7a8` remain prerequisites, not validation of this new implementation.

## 1. What the code implements

`snapshot/nacommon.py::run` stores h_t after input x_(t-1), starting at zero.
`run_from(h,x,t0)` begins with x_t0, so `phaseA.py` replacing h_t and passing
t is the correct suffix indexing. Remaining inputs are unchanged, and gates
are recomputed. Predictions use first-index `argmax`. The input target uses
`argsort` for the second-largest coordinate; ties have probability zero in
the ideal continuous input law, but no explicit finite input-tie convention
or guard is provided. This is a float64 recurrence implementation, not an
exact-real or original-library parity certificate.

Single-coordinate zero, pilot-mean and resample interventions are applied
after the ReLU. The pilot has a distinct literal RNG seed from evaluation;
there is no fitted phase-B formula in this snapshot. Resampling uses a fixed
permutation of the evaluation bank. A permutation allows self-donors and
introduces shared-bank dependence: the phrase "another random input" is not
a guarantee of a different row. All-step interventions reuse each donor's
clean trajectory at every step, rather than rolling out an intervened donor.
Group final-only interventions patch the clean h_10 and perform readout only.

The 22 named groups comprise six named groups plus 16 individual neurons.
Two named singleton groups duplicate individual n2/n4; these are duplicate
interventions, not independent replications. The source defines a coherent
computational intervention, but its effects alone do not identify a unique
natural role, information content, or redundancy theorem. Off-distribution
states and disrupted correlated codes are alternative explanations.

## 2. Modal interventions and selection

`snapshot/phaseA_modes.py` selects five final gate patterns by frequency on
the same evaluation sample. It diagonalizes `Wh[act,act]`, groups conjugate
eigenvalues using fixed tolerances, and constructs
`P = real(R[:,ks] @ inv(R)[ks,:])`. In exact arithmetic, when the eigensystem
is complete and the selected group is real/conjugate-closed, this is an
oblique invariant-subspace projector. It is not generally orthogonal.

Within each selected original final pattern, one permutation supplies donors
for all modes. At h_9 it still conditions on the **original final gate**,
which depends on x_9 and the original final transition. It is a retrospective
conditional intervention, not a prefix-observable donor rule. The last step
is recomputed and its gates may change. Only the selected final-active
coordinates are patched; other coordinates retain the recipient's state.
The modal reconstruction may produce negative coordinates before continuation
or readout. No ReLU is applied to the patched state itself. This is a defined
signed-state perturbation, but need not stay on the original reachable ReLU
state set.

There is no explicit condition-number threshold, projector idempotence/
completeness check, complex-residual check, repeated-eigenspace convention or
graceful singular-eigenbasis path. The README acknowledges ill-conditioned
patterns; source alone cannot establish which saved patterns satisfy a
numerical tolerance. Eigenvector-based neuron loadings are descriptive and
can depend on basis choices in repeated eigenspaces. Pattern selection and
modal ranking on the evaluation set need to remain descriptive unless a
separate frozen selection/holdout analysis is provided.

## 3. Concrete algebraic correction: projected energy is not variance removed

The source comment describes `energy_share` as "fraction of centered
final-state variance removed by the projector", but the expression is

    E_P = ||H P^T||_F^2 / ||H||_F^2,

where rows of H are centered states. The relative reduction in centered sum
of squares after **removing** the component is instead

    D_P = (||H||_F^2 - ||H - H P^T||_F^2) / ||H||_F^2
        = (2 <H,H P^T>_F - ||H P^T||_F^2) / ||H||_F^2.

They agree for an orthogonal projector, but not for a general oblique one.
An exact counterexample is

    P = [[1,1],[0,0]], R = [[1,-1],[0,1]], R^-1 = [[1,1],[0,1]].

Here P is the first eigen-coordinate projector, P^2=P. For centered states
±(0,1), E_P=1, while residuals are ±(-1,1), giving D_P=-1. These centered
states can arise from nonnegative states (0,0) and (0,2), or from strictly
positive active-coordinate states (1,1) and (1,3). A recurrent matrix
R diag(1,2) R^-1 supplies a complete real eigensystem realizing this P.
More generally replacing the top-right 1 by k gives E_P=k^2, so values above
one are possible algebraically, not solely a floating-point failure.

Thus the saved expression may be reported as a squared projected-norm ratio,
but not an additive variance share or the variance removed. Its denominator
also needs a nonzero-variance condition. The resampling intervention is
different again from removing a component. This review does not establish
an error in any saved accuracy or the size of this discrepancy on the model.
Suggested author correction: relabel the existing quantity, retain it,
and only add removed-SSE or covariance/cross-term diagnostics under a new
frozen evidence scope. No author file is changed here.

## 4. Metrics and uncertainty

`metrics` returns means for accuracy, agreement and four case accuracies.
The three x9 rank cases partition the sample; the blind case overlaps them.
An empty case would yield NaN. Case shares are saved, but the implementation
does not save explicit integer denominators or the five-way paired outcome
table from T244. Single-site JSON matrices round means to five decimals;
the NPZ stores matrix values, not row predictions or paired transitions.

Accuracy and agreement alone cannot reconstruct the paired accuracy-change
variance: agreement counts also include both-wrong same/different-label
outcomes. The README's generic standard-error approximation is not an
independent uncertainty calculation for every case, donor intervention,
selected pattern or multiple comparison. Dependence from reusing a donor
bank needs its own sampling interpretation. A finite large ablation effect
is evidence of sensitivity to that intervention, not a proof that a neuron
is nonredundant under every replacement. Conversely a small effect is not
proof of dispensability. The stated "mostly off-manifold" explanation is
a hypothesis, not separated from competing explanations by these arms.

`reportA.py` formats rounded single-site means; it prints five group modes,
omitting zero_final_only from the display although that arm is generated.
No saved number has been independently contracted in this review.

## 5. Provenance, finite program and cost limits

`nacommon.py` loads `m16_weights.npz` at import, without source-level hash,
shape, dtype or finite-value validation. The comment names an original
checkpoint but does not prove these particular archive bytes came from it.
The tree inventory records the archive's Git blob and size without reading
or decoding its payload. Existing separately reviewed checkpoint bindings
cannot silently be transferred to this new archive. Seeds and algorithm
choices are visible in source, but historical runtime/library binding,
threading and source-to-results correspondence are not independently shown.

For default phaseA N=200000, the source has 480 single-site assessments and
132 group assessments (66 all-step, 66 final-only), excluding the baseline.
The corresponding recurrent sample-step work is

    3*16*N*sum(10-t for t=1..10) + 22*3*N*10 + N*10 + 50000*10
    = 2830*N + 500000 = 566500000 at the default N.

This includes clean evaluation/pilot recurrences; it excludes readouts,
sampling, copies, metrics, sorting, mode study and library work. It is a
source-derived workload count, **not** FLOPs, native cost, elapsed time, or
actual execution evidence. The README's approximate sequence-forward/FLOP
statement is not a measured complete ledger. Source scripts have no durable
partial-results protocol or enforced CPU/memory cap; this is a limitation
of reproducibility and failure recovery, not evidence of a historical
resource violation. We grant no execution clearance.

## 6. Bounded next evidence and alternatives

Before empirical validation, bind the exact archive/checkpoint and runtime,
then separately scope a saved-output contraction if adequate records exist.
Paired counts/denominators, selected-pattern supports and projection diagnostics
are missing from the visible output schema and may require a new explicitly
approved author phase rather than being reconstructible from aggregate means.
Conditional donors, independent selection/holdout and cumulative replacement
remain plausible stronger tests. This source review neither rules them out
nor authorizes them. It does not audit phases B/C/D, absent from this tree.
