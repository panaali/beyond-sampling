# Independent T185 saved-evidence review

## Question
What does the saved H8T5 zoo0 result establish about accuracy, cost and scaling?

## Method
Immutable source/data snapshots in manifest.json; scalar arithmetic only under PROTOCOL.md. Source gates and independent review precede conclusions. No model/mass/MC replay.

## Status
Saved-data phase complete; independent final review pending. Claim3706fcfa;14 files pinned from latest DRI result5916a826 and README12c6e8bd. T55 source gate passed before execution.

## Results
The historical nominal ratio reproduces0.009706984995. Recipe0.8358034749864574 differs from the same-source numerical reference0.835805733133753 by−2.2581472956e−6. This is one clean model, not a completed five-seed/noisy rung.

The documented10M MC scalar implies8,357,964hits. Its ordinary numerical95% Clopper-Pearson interval is[0.835566643391,0.836025962173], giving historical-budget ratio range[0,106.647682836]. Thus the independent finite reference does not establish success or failure at this scale. RawMCpoint ratio is0.09528208. Beta inversion residuals are saved (largest2.31e−11 probability units); no outward enclosure or independently recovered runtime provenance is claimed. Historical-budget success requires true probability in[0.835780553813,0.835826393589]. A certified numerical reference could resolve this without hundreds of millions of new samples.

All60 saved thin-rule rows contract consistently;57 satisfy the production determinant threshold. GL6-versusGL10 differences are tiny, but neither is an exact reference. The lower-order log64 alternative has five negative derived masses; productionlog128 and saved thin6 have none in this finite set. Saved half sum is+7.527e−14, opposite the README sign. The saved validation rows have maximum mass4.418e−8, so the README's separate mass3e−6/3e−8-difference example cannot be reconstructed from this file.

Same-zoo0 T4-toT5 cell growth is18.4657 and nominal-cost growth245.5523. Extending13.5growth for five further steps gives9.7618e16 only conditionally. Prior unchanged-source factorial-chart completion lower bound27.6369B remains a separate qualified result; no method-family impossibility follows.

Source inventory identifies that leafsum diagnostics contribute wall time and global simplex counts but use a separate discarded Ops counter. The saved61minutes/4.9M simplices therefore are not pure recipe runtime/work. Recipe and exact share the numerical mass source and retain earlier gate-certificate qualifications. MC record contains one scalar, with generator provenance rather than saved hit/RNG/checkpoint ledger.

Reproduce in a fresh output copy (audit.py refuses to overwrite results):
```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/t185_evidence_review/audit.py
```
Original run used2GB/20CPU limits and `/workspace/work/codex-review.lock`; results.json and run1.log preserve all arithmetic. SOURCE_REVIEW.md explains source/provenance limitations; PROTOCOL.md was frozen before execution.

## Compute accounting
Scalar/hash work only; measured audit CPU is in results.json, excluding imports/finalserialization. Zero model forward passes, integrations, training or MC draws. The saved validation generator documents120M cone-validation draws plus10M main-model draws, not an independently recovered execution ledger. Original reported Ops remain an abstract proxy, not a complete installed native count.

## Claims
- [measured] The snapshot contains one clean T5 model, no T5 noisy or other-seed results.
- [measured] Saved-decimal historical ratio0.00970698 reproduces; conditional independent MC interval is inconclusive.
- [inferred] Agreement with the same numerical reference alone does not certify reference accuracy.
- [inferred] Finite growth observations do not prove recurring growth throughT10; existing unchanged-source lower bounds have narrower explicit assumptions.
- [speculative] Further source/error certification could establish the nominal single-model comparison; broader rung validation remains separate.

## Alternatives not ruled out
Numerical bias cancellation, valid reference accuracy, improved representations, cheaper mass rules, redesigned certificates and less expensive scaling all remain possible.

## Failure modes
Shared source errors, missing runtime provenance, numerical interval inversion error and abstract-cost mismatch can invalidate stronger conclusions. Finite validation does not establish universal error or growth bounds. Preserve all saved failures and qualifications.
