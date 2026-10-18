# T228 source and claim audit

Pinned source: `manifest.json` binds 37 original Git blobs at author revision
`9ba95de75f5e3737668f8e87755512cc79011eb5`. No author module was imported;
no archive array, checkpoint, estimator, model, RNG, geometry or mass was evaluated.
This is a source interpretation, pending independent review, not validation of the
historical execution or every saved numerical value.

## T216: exactly which symmetry was tested

A pure hidden-coordinate permutation P leaves a fixed bias-free recurrence and
readout unchanged when Wi=P Wi, Wh=P Wh P^-1 and Wo=Wo P^-1. The pair scanner
compares incoming and outgoing vectors under the transposition of i and j. Its
reported outgoing maximum below one can exclude those tested transpositions,
conditional on faithful weights and arithmetic. It is not by itself a test of
all multi-neuron permutations: a three-cycle recurrent permutation matrix, equal
input weights and identical readout columns has a cyclic automorphism without
any transposition automorphism. A complete pure-permutation exclusion could
instead use the distinctness of all readout columns, or directly solve the full
constraints. That exact target check was not run in this phase.

The saved `symm.json` retains near pairs and top-five summaries, not all 120
pair records or checkpoint weights/hashes. It cannot independently reconstruct
the complete scan. `dead` tests small incoming/outgoing norms (<1e-6), not
reachability or inactivity of neurons. No zero norms is not a proof that no
neuron is functionally dead on the reachable state set.

The positive hidden gauge is Wi'=D Wi, Wh'=D Wh D^-1, Wo'=Wo D^-1 for positive
diagonal D. Exact proportionality predicates survive appropriate invertible
coordinate changes, but these Euclidean row cosines are not generally invariant.
For Wi=(1,1), Wh=I, the two augmented rows have cosine 1/2; under D=diag(2,1)
they become (2,1,0),(1,0,1), cosine 2/sqrt(10). Thus the source comment's
unqualified scale-invariance of near-collinearity scores is false. This does not
make the stored cosines in their chosen parameterization wrong.

`symm_model` projects selected augmented incoming rows onto an SVD direction.
It keeps Wo unchanged, but edits rows of Wh, hence also changes entries in the
recurrent outgoing columns. The README statement 'outgoing weights are kept'
needs this distinction. These interventions are not function-preserving
quotients. In exact arithmetic projected rows share a linear form (possibly
with opposite signs); binary64 projection does not establish exact collinearity
or exact complementarity. At the zero form both ReLUs are OFF, not complementary.

The 10M paired-MC script is retained with RNG seed11, the full-group intervention
and the correct paired difference/std calculation. Historical inputs, paired
outcomes and model hashes are not retained. Three 2M variants appear only in
`symm_variants.json` and prose; the inline driver, RNG/weight bytes and raw counts
are absent. Reproducing saved arithmetic does not reproduce these experiments.
Their reported large losses can reject the recorded interventions, not all
function-preserving merges, reachable-domain equivalences, signed-coordinate
representations or more general symmetries. Absence of parallel incoming rows
alone does not exclude all those possibilities.

## T217: finite conditional gate-word concentration

`prefix.run` starts with an artificial simplicial fan, runs floating `estT`
geometry in exact-splitting mode, and groups leaves by complete stored gate word.
The mode name does not turn `TOL=1e-10`, numerical ranks/inverses or clipping
into exact arithmetic. The target counts 34/792/21032 separately agree with the
independently verified T219 exact-history counts; this does not certify this
run's individual leaves, zoo counts, split-event labels or masses.

`eventword.py` splits every leaf by pairwise input-order planes. Assuming valid
full-dimensional ordered pieces, its R/S/L label is constant in each piece;
the first later input is R or S, never L. The event word forgets which old
non-top-two inputs are ordered relative to each other. The two models labelled
zoo16_4/zoo16_5 are seed0 only. Target/zoo are loaded from live alg_zoo entrypoints;
no per-run checkpoint hashes or environment versions are recorded.

The Gaussian masses at k2/k3 use floating inverse/correlation/asin formulas;
k4 uses determinant-selected GL10 Duffy or GL128 log/cofactor quadrature with
fixed truncation. These are numerical masses, not certified exact integrals.
The source clips correlations and accepts signed mass outputs without a global
positivity/error gate. Near-unit total mass can hide cancelling individual
errors. Existing T4 mass reviews elsewhere do not automatically cover these
new, order-clipped cones. The incidental `m43` import reads m43_weights.npz;
that blob is pinned here, but it is unrelated to target model provenance.

For nonnegative joint masses, modal_share=sum_e max_g p(e,g) is exactly the best
probability of recovering the ENTIRE gate history with one word chosen per
listed event label. Its failure of 99% is not an accuracy lower bound for a
network predictor, an error estimator, or an aggregate of many histories.
exp H(G|E) is an entropy effective count, not a certified covering cardinality
or computation count. `n99_mass_weighted` averages within-event 99% cover counts,
not a worst-event cover or full-model complexity guarantee. The step versus
cumulative 'branching >=8' ambiguity remains a DRI criterion interpretation.

Saved NPZ fields are only `event`, integer `gate` id and `mass`. The `SX` sign
array and id-to-gate-history mapping are NOT saved. Therefore the +sign(x0)
variant cannot be independently reconstructed from this archive without extra
data/replay. No attempt to infer the missing signs from numeric gate IDs is
justified. Original geometric rays, event inequalities and mass-call ledgers
are also absent. Saved-only contraction can audit unsigned formulas, event-law
marginals, negative/nonfinite masses and summaries, but cannot establish label
ownership, history equivalence or numerical integration accuracy.

## What a later saved-only check can establish

Freeze all nine NPZ/JSON pairs and six prefix summaries, the symmetry summaries,
and the three source formulas before decoding arrays. Independently group
exact binary64 masses by event and id; contract total/modal/cover counts and
entropy diagnostics; verify analytic event marginals from the 2!,3!,4! equally
likely strict input orders. Keep every mismatch/nonpositive mass. Compare the
reported intervention losses and k2/k3/k4 counts without rerunning models or
making new significance claims. Preserve signed variant/provenance gaps.
No T222 coverage curve or T225 report edits belong to this phase.

## Alternatives and cheapest discriminators

1. A source/statistics defect, rather than physical fragmentation, could explain
   the modal table. A fixed saved-array arithmetic and event-marginal check is
   cheap and independent; it does not resolve mass/geometry error.
2. The negative symmetry result may be representation-specific or intervention
   damage. The exact gauge counterexample above already distinguishes cosine
   invariance from exact algebraic symmetry. A future separately claimed exact
   readout-column/permutation check could address pure permutations only.
3. Correct final outputs may aggregate distinct histories. Modal recovery of
   the entire word cannot decide this; future-equivalent/domain certificates
   remain a distinct lane. No numerical expansion is proposed here.

These finite findings do not meet a universal method-family exhaustion claim.
