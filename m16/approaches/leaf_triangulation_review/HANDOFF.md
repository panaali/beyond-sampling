# T136 bounded handoff

Claim3e02ca4, protocol89eadbd, controls/implementation18f0ea1. Independent T55 review pending. No shared code edits.

Fixed H8T4 seed0 parents0,1,20,21: all337 raw leaf cones (1762 source pulling simplices) agree with independent ConvexHull volumes constructed from all raw vertices, in ordinary and full-SVD-conditioned affine coordinates. Both frozen parent-relative screens pass. Maximum parent aggregate relative residual2.11e-15; maximum per-leaf raw absolute discrepancy1.46e-16. Inequality/incidence residuals<=2.72e-16. Trained CPU2.10seconds, peakRSS221372KiB, no Gaussian integration or random inputs.

Interpretation: no evidence of a source-pulling volume defect in this fixed panel. NOT a certificate for other parents, clipping correctness, absence of overlapping tetrahedra compensated by holes, or Gaussian masses. Extreme rotated width2^-45 controls lose .2% relative ideal volume after floating transforms; preserved in controls.json. Conditioning cannot restore lost input bits.

Files: review.py pins source/dependency/checkpoint; controls.json retains all11 analytic controls plus expected flat rejection; results.json retains every leaf/parent diagnostic and source hashes; raw_leaves.npz retains each raw R/N/T and source tetrahedra plus anchor (no pickle). README gives bounded reproduction commands and complete scope.

Alternatives checklist: full raw vertex hull versus source simplex subset tested; full invertible SVD conditioning tested; explicit extreme thin precision control tested; Gaussian-mass error investigated separately by T131; independent clipping enumeration and all remaining92 parents not tested here (outside frozen finite panel). No broad negative or exhausted-method claim.
