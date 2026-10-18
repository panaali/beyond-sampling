# T210 conditional first-index tie repair

## Question
Does the conditional affine-interval estimator respect the network's first-index argmax for identical logits, and can a local expression repair restore that semantics?

## Method
Claim50d5a5af precedes source work. Pin unchanged rb9 source and reuse T204 original zero-readout control outputs57b3679d; no baseline replay. THEORY.md separates strict earlier-index constraints from weak later-index constraints. A local loader/control phase will require an independently reviewed frozen protocol before any candidate call. Shared source adoption stays DRI-only.

## Status
Source/theory gatebc20b0f9 passed; original T204 saved review40ea5646 is pinned. Immutable gate0b68b425/sharedb1949b3c preceded sole22-call run9a3836d1. All fixed cases completed; independent T210 saved audit e0c73bc7/sharedb930d0b3 passed. Bounded local handoff complete; shared adoption remains DRI-only.

## Results
[inferred] For identically zero readout prediction is0; original Q's identical-competitor test excludes all labels. Conditional analytic truth depends on whether prefix index0 is first or second. Proposed repair changes only the constant-margin impossibility predicate.

## Compute accounting
22 toy Q batch calls (10 original,12 repaired),44 conditional prefix evaluations,24 analytic80-digit references. Two original Gaussian zero-readout values reused exactly once; no radial baseline replay. Recorded.224005195CPU includes scientific imports after main begins, excludes final serialization; under20CPU/2GB review lock. No random inputs, checkpoint/trained model, MC, quadrature or native-cost tracing. Structural repair overhead is two additional comparisons and two Boolean operations per N×10 competitor array plus index construction, not a complete source/runtime price.

## Claims
[inferred] The proposed constant-margin predicate is necessary and sufficient for first-index ownership on a valid affine piece under a continuous conditional input law. Independent theory/source reviewbc20b0f9 passes. No universal floating-accuracy claim.

## Alternatives not ruled out
A direct whole-piece max-envelope implementation could avoid pairwise constraints; explicit interval-endpoint ownership could support discrete conditional laws. Those variants are unnecessary to this density-law tie defect and are not tested here.

## Failure modes
Tiny-slope gate cutoff, nonfinite or zero radial prefix, input-value ties, ill-conditioned margin roots and CDF tail cancellation remain. The code's stated exactness is not established globally by this repair. No trained bias or historical ranking conclusion follows from analytic fixtures.

## Fixed outcomes (independently audited)
Independent halfline winner reconstruction and Gaussian erfc/Student incomplete-beta integration verify all24 repaired values match the analytic references within6.183e-17. Unique5 and sign-switch fixtures preserve original output bits for both laws. Later-index ties now retain their valid intervals; earlier-index ties remain excluded. Original zero-readout Gaussian0 becomesPhi(1)=.8413447460685429 for the inherited prefix whose second-largest index is0; its other prefix remains0. The all-OFF negative-halfline fixture adds.5 in the expected case for both laws. All outcomes retained, no trained/general accuracy inference. Run command: one-thread `python review/srb_tie_semantics/controls.py` under review flock/external20CPU/2GB; no overwrite or replay. Initial inventory.json pins snapshot54c719ce, including the pre-review wording; current run dependencies are protocol.json/gate.json.
