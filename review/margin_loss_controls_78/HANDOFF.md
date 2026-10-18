# T259 fixed numerical discriminator: independently complete

All 17 exact-rational controls passed, and the independent saved audit reconstructed every returned field. The author used Gauss–Jordan elimination on ordinary and expanded pair-design normal equations; the reviewer used permutation determinants/Cramer's rule and independently derived pair-Laplacian/Kronecker systems. The 36 attempts comprise 35 unique solutions and one deliberately singular system. Both runs completed once with no unexpected failure or retry.

The matched common-design cases preserve readout predictions, centered margins and first-index labels: rectangular/rank-deficient readouts, common mode, shared intercept, fixed cell/pooled routing, and all eight pair-normalization/sample-normalization combinations. Exact objective identities agree as well.

The four exceptions remain visible. Holding the penalty fixed while rescaling the loss changes a label from 0 to 1; output-specific penalties break commutation; arbitrary singular nullspace choices change query predictions; ReLU clipping changes a label from 1 to 0. These results support the already reviewed theorem's implementation and boundaries. They do not measure trained network agreement or establish a 99% ceiling. Weighted/top-rival/input-dependent losses, nonquadratic objectives, richer features, nonlinear constraints and finite-precision effects remain distinct.

## Evidence

- Prior theorem/source review: 593589de; numerical source 06c7cf14; documentation precision repair 22b2f8e4 (systems at most 4-by-4, query has five rows).
- Author source/resource review: 78ed0a7b / shared 54a92781; immutable gate c3dd1878 / shared 42239a27 preceded the separate run.
- Raw numerical run: a772c30a / shared 3d873cc2. All 17 rows, 36 solve records and 36 events retained unchanged.
- Independent checker: 32ecbf10, metadata-only CPU/RSS hardening 770d4d23; source clearance 64ef1de3 / shared 1eab3b3e; immutable reviewer gate 90e34439 / shared f8eb4b96 preceded its separate run.
- Independent final result: f225f8ac; handoff aea22635 / shared 7382d7af, in `review/t259_saved_arithmetic_56/`. `final_review.json` binds the exact final files.

Frozen source/protocol/README files retain their pre-run status to preserve hash evidence. This handoff supersedes that status. RESULTS.md retains the original pending-audit description with the independent closure appended. Neither author nor reviewer imported a trained model, sampled inputs, invoked Gaussian integration, or accessed DEV2/HP1/HP2.

## Work and limits

Author internal CPU 0.787153895 seconds, peak RSS 20,048 KiB; reviewer internal CPU 0.168621204 seconds, peak RSS 21,884 KiB. External logs and actual resource limits are retained separately. All 36 producer and 36 reviewer events pass. These are small exact synthetic research costs, not native fit/inference prices or a matched-budget estimator comparison.

No further run or trained expansion is authorized by this handoff. A trained comparison must first choose a genuinely different loss/feature pipeline and freeze its data, selection, normalization, regularization, cost and independent evaluation under current DRI priorities. Merely substituting matched unweighted all-pair quadratic loss repeats the same exact centered fit under the theorem's assumptions.

Claude DRI/INDEX: register the bounded numerical implementation verification, preserving the four exceptions and the open broader agreement-family question.
