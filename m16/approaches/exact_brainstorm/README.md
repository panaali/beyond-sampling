# exact_brainstorm (T346): clever exact methods, and exact computation as a component

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [Owner Q&A](../../../OWNER_QA.md) §6.2

## Question
Plain region-by-region enumeration of M_16,10 is dead (H-E13, T270: prefix regions grow x23-33 per step, ~1e13 at
T=10; mass coverage grows >= 7.6-14x per step; bounds widen ~50x per step). The owner asked (2026-10-18) to explore
two things that verdict does not exclude: (1) cleverer exact methods, and (2) exact computation used as a component
of an estimator. Step 1 (this file): the DRI's seed ideas. Step 2: a brainstorm with Fable and Astra (TASKS T346).

## Method (seed ideas, all [speculative])
Each idea: what it is; why it might escape the region explosion; what we already know; the cheapest kill test.

**A. Enumerate only the error set's boundary, not all regions.** Accuracy needs only the set where the output margin
changes sign. Regions where every margin keeps its sign contribute their whole mass, so their internal gate
boundaries do not matter; only pieces of the decision boundary need resolving. *Known:* errors live near the tie
walls, where regions are ~40x finer (H-E15), so the boundary may carry most of the region count anyway; prior work
in [boundary/](../boundary/README.md) and [cone_enum/](../cone_enum/README.md). *Kill test:* from T270's k = 4-6 data,
count regions that intersect the decision boundary vs all regions; go if their growth is <= ~6x per step.

**B. Merge prefixes with identical futures (automaton minimisation).** Two prefix regions whose affine maps to the
hidden state agree, or differ only in directions the rest of the network never reads, behave identically from then
on and can be merged. *Known:* the state seems low-dimensional in function (running max, leave-one-out max, recency;
T241), but small weights matter (T290) and exact equality is rare. *Kill test:* on k = 4-5 prefixes, count distinct
(affine map projected on the readout-relevant subspace) classes, exactly and at tolerances; go if classes << regions.

**C. Backward (readout-side) decomposition.** Start from the error condition at step 10 and pull its preimage back
step by step, keeping only branches with non-negligible Gaussian mass. Because the error set is thin, backward
branching might be much smaller than the forward x23-33. *Known:* untested (T326's floor note named it as the only
kind of geometry change that reduces distinct inputs). *Kill test:* backward branch counts for 3-4 steps vs forward.

**D. Exact in some steps, declared glue in others.** Exact enumeration is cheap for the first few steps (k = 4:
~21k regions) and the decision is made late. Variants: exact prefixes plus exact 1-D walks for the suffix (T317's B2
prefix tree is a restricted version); or a declared low-dimensional state early and exact late steps. *Kill test:*
oracle split of the error mass into "determined by step k" vs later, for k = 3..6.

**E. Exact enumeration restricted to a small sub-cone (rare events).** Regions inside a small input sub-cone (e.g.
far-gap g23 > 1) may grow much more slowly than overall. *Status:* running as T344 option (b).

**F. Exact low-dimensional slices as components.** Generalise the exact 1-D walks: exact 2-D cells only where the
product approximation fails (T333's corner check), exact 3-D in (v2, g12, g23) under a declared lows closure
(T344 option (a)). The rule found so far (T324/T334, addendum A): the exact joint set is needed at >= 2 anchors per
region, and full J2 costs >= 13-19k FPE, so only localised versions can pay.

**G. Exact truncated models as anchors.** Exact accuracy is cheap for small T (and for fewer input numbers); a
declared mechanistic recurrence from T to T+1 would carry it to T = 10. *Known:* E25's length scan and the ladder
(E43-E48: the recipe is accurate along T but its cost explodes); an extrapolation in T is an inductive step unless
the recurrence is derived from W.

**H. Structure of the affine maps.** Within a region the map is a product of D_t W_h with rank-1 input injections
x_t W_i. Shared sub-products across regions (a trie over gate words, like T317's B2 tree) or low-rank structure
could cut the per-region cost even if the count stays. *Known:* Fable r5 found low-rank W_hh needs rank < 8 to save
anything (T290: small weights matter); this attacks cost per region, not count, so it cannot fix a 1e13 count.

**I. GPU brute force for trend data, not for the answer.** The T192 kernels can count k = 6-7 quickly; use them to
measure the growth rates of A-C, not to enumerate T = 10.

**J. Exact sensitivities for the long-term goal.** Along exact pieces, derivatives of an event's probability with
respect to the weights are exact; useful for "which weights make a rare failure more likely" (Owner Q&A §4.4), not
for the accuracy estimate.

Literature to bring to the brainstorm (not read for this file; log any post-January-2026 ARC source in
[lit/web_log.md](../../../lit/web_log.md)): linear-region counts in practice vs worst case (Hanin and Rolnick);
reverse-search hyperplane-arrangement enumeration (Avis-Fukuda); exact and probabilistic verification of ReLU
networks under Gaussian inputs; zonotope/DeepPoly and branch-and-bound verifiers; ARC's M_4,3 Girard-volume
estimator (E32) and its deduction-projection work (lit/web_arc S10).

## Status
Proposed, low priority (owner, 2026-10-18). No code, no runs.

## Results
None yet.

## Compute accounting
None used.

## Claims
All ideas above are [speculative]. The facts they rest on are cited with their task IDs.

## Alternatives not ruled out
The region explosion may be intrinsic to the near-tie geometry (H-E15), in which case A-D all inherit it; that is
exactly what their kill tests check.

## Failure modes
Every idea that keeps exactness must still resolve the thin, fragmented error set near the tie walls; an idea that
drops exactness becomes a declared approximation and joins the T317 family's 1:1 cost/error trade (Fable r5).
