# T210 conditional first-index tie repair

## Question
Does the conditional affine-interval estimator respect the network's first-index argmax for identical logits, and can a local expression repair restore that semantics?

## Method
Claim50d5a5af precedes source work. Pin unchanged rb9 source and reuse T204 original zero-readout control outputs57b3679d; no baseline replay. THEORY.md separates strict earlier-index constraints from weak later-index constraints. A local loader/control phase will require an independently reviewed frozen protocol before any candidate call. Shared source adoption stays DRI-only.

## Status
Source/theory preparation; no estimator calls here. The original T204 saved audit is still a dependency before using its values as independently checked controls.

## Results
[inferred] For identically zero readout prediction is0; original Q's identical-competitor test excludes all labels. Conditional analytic truth depends on whether prefix index0 is first or second. Proposed repair changes only the constant-margin impossibility predicate.

## Compute accounting
Read-only source and saved JSON inventory only; zero candidate, model, reference, MC, integration or native-cost calls.

## Claims
[inferred] The proposed constant-margin predicate is necessary and sufficient for first-index ownership on a valid affine piece under a continuous conditional input law. Independent theory/source review pending. No universal floating-accuracy claim.

## Alternatives not ruled out
A direct whole-piece max-envelope implementation could avoid pairwise constraints; explicit interval-endpoint ownership could support discrete conditional laws. Those variants are unnecessary to this density-law tie defect and are not tested here.

## Failure modes
Tiny-slope gate cutoff, nonfinite or zero radial prefix, input-value ties, ill-conditioned margin roots and CDF tail cancellation remain. The code's stated exactness is not established globally by this repair. No trained bias or historical ranking conclusion follows from analytic fixtures.
