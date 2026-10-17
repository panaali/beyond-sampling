# T193 first saved-only gate

Claim10c36c40 precedes snapshot8bdf7959 and implementation. No imports of author
estimators/leaderboard, no checkpoints, noisy generation, inputs, masses or MC.

Freeze all9 saved H8T4 rows, summary, source-documented10M diagnostics and source.
Reproduce the original cost convention: FPE=6000*640/5760=2000/3. Exact rational
arithmetic uses stored JSON decimal values; published floating formulas agree
within1e-12 relative+absolute. Fractional and floored draw ratios are separate.

Reconstruct MC integer successes from saved decimal frequencies, label absence
of raw hit ledgers. Compute fixed pointwise95% and nine-case Bonferroni95%
Clopper-Pearson intervals via scipy beta.ppf, verify numerical tail residuals.
These are numerical intervals conditional on source-documented binomial samples,
not directed rounding or historical model authentication. The same seed2026
stream reused across models forbids assuming independent model errors; union
bounds remain valid without independence.

For fixed estimate e in(0,1), r(p)=(e-p)^2/[p(1-p)] decreases to0 at e then
increases. Its derivative has only the interior zero p=e (the other algebraic
root e/(2e-1) lies outside(0,1)). Thus endpoint maxima and zero-or-endpoint minimum
are the range on a probability interval. Three exact scalar controls freeze
interior and both exterior cases. No grid search or confidence-level tuning.

For each stated numerical truth p, break-even draws B=p(1-p)/(e-p)^2 are
conditional, not a verified budget. Report B*(2000/3) under original convention
and B*512 as hypothetical explicit-source sampler headroom. The existing legacy
estimator proxy and selected native hybrid are NOT converted into512 units or
used to infer a sampling win/loss. That requires a matched operation convention,
actual recipe/case census and complete applicability evidence.

Reuse independently reviewed positive exact seed0/1 references as conditional
integration intervals only. Current T4 source is compared by AST with the
independently tested139a7b9e source, allowing only the unreachable d==5 dispatch.
Compare top-level old setup assignments, retain new extra5D setup as unpriced.
Source parity does not authenticate historical runtime/models. Published files
lack model/source/runtime hashes; namespace comes from frozen generator source.

Preserve earlier T185/E48 review199420a1 (one cleanH8T5 nominal.0097, independent
10M uncertainty0..106.65) without redoing its sixty controls or mass experiment.
T11584 actualT5 corpus decision remains blocked. No cost projection lower-bound
or complete-rung implication. Report nine-case membership accurately: five clean
seeds plus four noisy copies ofseed0, not noisy copies across allfive seeds.

One saved-audit attempt under20CPU/2GB review lock, durable per-case records;
refuse old outputs. Independent source gate then independent saved audit. A
later decisive actual reference/cost phase must be separately claimed/frozen.
