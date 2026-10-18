# Coarse-family regularity: a falsifiable follow-up design

## Question
Does coarse grouping make the full tie-face integrand regular enough to exploit,
or does it merely cover a large share of the selected tail?

## Method
`manifest.json` pins source inventory only. `DERIVATION.md` separates tail
coverage, oracle conditional variance, implementable sampling and deterministic
quadrature diagnostics. `PROTOCOL.md` specifies full-array/label provenance,
actual smaller-rule tails and occupancy-preserving nulls, with explicit gates
before any future computation. The label is outcome-derived and expensive.

## Status
Source/protocol phase only, claimed1f396509. No numerical execution or array
contraction. Independent design review is delegated to910 afterT231. Full-target
application remains blocked on data/provenance, DRI priority and separate gates.

## Results
No new numerical result. Independently reviewed T227 retains coarse L30
coverage90.86–98.89% over saved selected tails while fine labels fail their
fixed criterion. Neither establishes full-family regularity or a cheap
estimator. The independent-label analytic counterexample shows why high
coverage alone cannot imply variance reduction.

## Compute accounting
Only text/source/hash inspection and documentation; no author imports, model,
face, RNG, fitting, geometry, integration or raw-array calculations. Prospective
setup, labeling, probability estimation, rejected proposals and validation costs
are explicit. No FLOP advantage or complete cost claim.

## Claims
[inferred, review pending] Total variance and oracle allocation distinguish a
useful partition from an implementable estimator. [speculative] A properly
controlled coarse-family test may discriminate alternatives left open byT227.
No new numerical success, method adoption or universal negative is claimed.

## Alternatives not ruled out
Full-family regularity, simpler weight-derived labels, intermediate labels,
deterministic conditional integration, or alternative transport/allocation may
help. Existing coarse labels may prove too expensive, or offer no residual gain.
The protocol is designed to retain either outcome without posthoc selection.

## Failure modes
Global-tail conditioning bias, label occupancy confounding, expensive label
construction, unknown family probabilities, missing zero/capped rows, reused
validation, quadrature bias and unpriced rejection can invalidate interpretations.
Incomplete original arrays block application; missing values are not zeros.
