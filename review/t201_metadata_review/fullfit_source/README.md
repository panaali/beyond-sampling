# T201 frozen full reconstruction source/resource gate

## Question
Does the one fixed train/two-holdout reconstruction phase preserve independent evaluation, data provenance, failure accounting and its total resource budget?

## Method
Read author2c0b8311 and narrow correctionb6838e47/shared426b3e91, with previously reviewed core.py, pilot raw/source and independent saved audits. No author import, random draw, fit or forward performed by this reviewer. Assignment4226647e;910 owns the later independent saved audit.

## Status
Source/resource gate passes for the declared one fixed process after immutable reviewer/source/protocol pins are published. Any result remains pending execution and saved audit. No broader continuation is cleared.

## Checks and results
[inferred] Training reuses the exact1024 pilot prefix once, resumes its pinned PCG64 state and draws63 more chunks. Two fixed distinct-seed holdouts draw64 chunks each. Thus191 new1024-input chunks,195584 new vectors and1955840 normal values;196608 total including pilot. No excluded or replacement tied inputs.

[inferred] Reviewed helpers fit216 homogeneous state maps and one case2 separable map using training data only, fixed rcond1e-12. Coefficients are retained unchanged for both heldouts. Full inputs/hidden/logits, coefficients and labels suffice for independent score reconstruction without refitting or new network calls. Float64 wiring is supported by the prior fixed1024 pilot; no all-input binary equivalence is inferred.

[inferred] Network agreement and target accuracy remain distinct. Separable metrics use case2 only; hidden R2 uses each scored panel's own mean, allowing negative values and explicitly None for zero variance. K0/K1 blind errors and K>=2 case0/1 global-third errors are disjoint, and the remaining bucket makes exact error conservation. All tables and counts are retained.

[inferred] Ordinary95% Clopper-Pearson intervals are issued only on holdouts conditional on the fixed model/fit. Six task/agreement requests plus two blind-error requests per holdout, plus four boundary controls, gives20 requests; endpoint cases change actual beta-function call count, which is separately logged. The intervals are individual, not simultaneous or fit-variability intervals; numerical inversion is not outward certified. The random case2/K-bin denominators are valid conditional selected-sample denominators under the declared iid holdout law.

## Resource and failure accounting
Exactly one process: one thread, primary lock,90CPU soft total guard and110CPU external total limit,2500000KiB virtual cap and2GiB completion RSS check. This is one total reserve, not191 reserves. Chunk inputs/state precede forwards; output arrays precede scoring; core counters retain partial solve/forward attempts. Existing output-directory content now rejects before any new work. Exclusive archives and durable partial results prevent automatic replay. Internal CPU excludes interpreter/top-level imports and final writes; external Bash/process limit includes them.

## Claims
[inferred] This is a coherent bounded new reconstruction experiment. It cannot authenticate missing historical fits, establish a deductive estimator or an ARC rung, or prove generalization beyond the fixed model and sampling law. Reviewer performed only source/document/hash work, zero numerical candidate calls.

## Alternatives not ruled out
Different model/noise populations, feature families, input-dependent transformations and approximation failure remain open. Two untouched holdouts test in-sample optimism; they are not permission to select a better variant afterward. No performance-based stopping or sample-size adjustment is authorized.

## Failure modes
Treating training scores as held-out estimates, conflating target accuracy with network reconstruction, multiplying CPU reserve per chunk, silently replaying partial archives, claiming simultaneous CI coverage, or promoting pilot parity to universal floating equivalence would overstate this phase. Independent saved review is required before handoff.
