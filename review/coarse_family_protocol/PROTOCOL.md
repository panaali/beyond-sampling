# T230 source-only protocol and execution boundaries

Claim1f396509 reserves documentation/source/theorem work only. Nothing below
licenses a model, walk, random, face, fit, array-contraction or mass call.
Independent design reviewer910 is assigned after higher-priorityT231. The
existing T227 evidence and original negative remain unchanged.

## Question and competing explanations

Does the existing coarse partition describe a regular full integrand, or merely
collect most already-selected tail weight in a few large outcome-defined bins?
Alternative1: high coverage is mainly occupancy; compare an occupancy-preserving
null. Alternative2: useful variation is between families but invisible in the
selected global tail; measure full-family residuals. Alternative3: an apparent
size effect is tail selection rather than quadrature stability; reconstruct
actual smaller rules and their own tails. Alternative4: labels help only after
expensive outcome computation; retain label/conditional-generation cost.
None of these alternatives is excluded by the existing fine-label failure.

## Data contract before a future application can be proposed

One immutable record per face: model-byte hash and exact source/runtime pins,
rule identity, kind, original index, quadrature weight, source X=Z*F, label and
label-status, walker completion/cap status, tie ownership and construction cost.
Preserve actual inputs or exact regeneration provenance; save ALL rows, not just
the top5%. Scalar label consistency with vector F is a diagnostic with a frozen
tolerance, never an exactness certificate. Unknown/capped/nonfinite records are
retained and prevent a success claim; no silent deletion or zero replacement.

The original label must be specified literally: (kind,ub,wt,min(nflips,3)),
where ub uses u<=0,<=.25,<=.5 boundaries, wt describes the dominant wrong
component, and flips count source cell endpoints inside that component. Resolve
exact equal dominant masses by original stable component order. Add an explicit
(kind,ZERO) label only when the completed source has F=0 and no error component;
this is a source-numerical zero, not exact network correctness. Missing labels
or disagreement are UNKNOWN, not ZERO. Retain observed source tolerance/cap
qualifications. A new independent exact labeling algorithm would be a new task.

T227 has full global summaries and selected-tail labels, but no full-face values
and labels. Those cannot be reconstructed by treating omitted rows as zero.
Original rules700/701 were already inspected; reuse would be descriptive, not
untouched validation. New data requires separate DRI priority and a frozen model,
rule IDs, sample count, runtime/resource cap and source gate before any call.
No target size or new rule ID is silently authorized here.

## Proposed future diagnostic (not yet an executable preregistration)

1. Freeze the original coarse labels and L=30; retain fine mech L=100 only as
   historical comparator. No extra label search. A primary full-family A-to-B
   squared-residual score uses every row as in DERIVATION.md, with unseen-label
   fallback and zero denominator reported undefined. Report occupancy, mean,
   second moment, within-family variance, zero fraction and family top5% share
   over each ENTIRE family, not its intersection with the global tail.
2. Recompute each smaller-rule tail from all faces of that smaller rule. If the
   source Sobol generator has a proved prefix relationship, use first n/2 rows
   of EACH kind from its full arrays, preserving original weights/indices. Verify
   byte-prefix identity, not assumed generator equivalence. Otherwise a distinct
   smaller-rule evaluation needs its own authorization. Rank all its X^2 values
   and choose ceil(.05*Nsmall), with ties broken by(kind,index). This is different
   from thinning the old selected tail. Pooling rules is secondary and gets a
   freshly selected pooled tail.
3. For the occupancy null, permute the complete multiset of family labels among
   face indices WITHIN each kind and rule. Preserve every family count, zero
   label count and original label names exactly. Independently permute A and B;
   this deliberately preserves occupancy differences between rules while breaking
   X-label association. Train the same selection/means on permuted A and score
   permuted B. A source-only implementation must freeze the shuffle algorithm,
   seeds and finite repetition count before generating assignments. Save every
   permutation or full assignment; no regenerated unarchived random controls.
   A separate tail-only permutation answers a narrower question and must not
   replace this full-array null.
4. Report raw original and null scores, both directions and all size variants;
   do not optimize over null draws or new labels. A finite permutation reference
   distribution is only conditional on the frozen arrays and the stated
   exchangeability null, not a Gaussian-rule CI or population error bound.
5. Any numerical go/no-go threshold, new model/rule identities and resource
   allocation must be fixed in a separate execution protocol before data access.
   This design intentionally makes no success decision on unavailable arrays.

## First cheap implementation step, if separately claimed

Synthetic tables only: constant integrand; allzero integrand; independent label
with95% occupancy; perfect outcome partition; unseen validation label; tied
rank cutoff; separate-kind unequal occupancy; and missing/nonfinite/capped rows.
Test exact analytic score identities, stable tie/zero handling, multiset
conservation and smaller-rule tail selection versus old-tail thinning. No
trained arrays, RNG assignments or functions have been evaluated in this phase.
Freeze exact fixtures and reviewer gate before such controls, with <=20CPU,
2GiB, one thread/review lock and durable outputs. This is an available next
prerequisite, not execution clearance.

## Accounting and stop conditions

Charge model-free setup distinctly, all full/scalar walks and labels, all
rejected proposals, unknown p_l estimation, conditional sampling/generation,
pilot fitting/selection, shuffling and validation. Review overhead is separate
from any candidate estimator. Reusing validation to fit allocation/means or
p_l is disallowed in a claimed independent comparison. A deterministic
mechanistic method still needs its own accuracy/error/cost argument; oracle
conditional Monte Carlo is only a baseline. Missing arrays/cap-status/provenance
block application. No new data work proceeds until DRI priorities and a separate
source/resource gate clear it. No broad family retirement follows a finite miss.
