# T20 cold importance bounded handoff

Original claim699cf3d; next-phase update65de2ed. Existing implementation/protocol1942489,f5b3f8c; recovery/control chain9874ac9,f1f8d57,1181300,d5a2a20,736b9f9,c964fa4,81233a1,af5095c.

Complete: model-specific cold pilot, uniform/advice/SD/RMS interfaces, pilot/support/failure ledger controls, same original checkpoint512/2048 and zero-readout512. Source/table/checkpoint hashes recorded. All SD/RMS model setup and pilot costs charged; original advice prefit cost unknown, so advice totals incomplete. The scalar costs are nonportable proxies; no inherited legacy MSE claim.

Independent positive scalar quadrature checks all four zero-readout production formulas, with no network or incomplete-beta calls, agreement<=2.26e-13. Its discrepancy from exact accuracy.1 is finite outer-quadrature bias. SD .09830463 and RMS .13239600 show that improving the synthetic stochastic second-moment objective does not guarantee better deterministic quadrature. Preserve both. Synthetic constant-stratum variance SD7.1875 vs RMS.32716 verifies the score distinction only.

T55 independent source review accepts support, weighting, no pilot reuse and charged setup, and independently derives the same distinction between fixed-strata allocation and mixture second-moment objectives. RMS/sqrtcost is not exact variance-cost optimal in general, nor proven optimal for deterministic rules. DRI acceptance remains pending.

Completed numerical phases: nine original comparisons~1.4CPU, independent weighted constant faces.034CPU, RMS discriminator.792CPU, all single thread under shared locks and memory caps. Read JSON for full cold import-excluded CPU/proxy components.

Open separate work: actual per-model/noisy-panel MSE with independently accepted cost convention; fixed-count stratified production; pilot/production randomization (sampling estimator); geometry-conditioned weights; total portable special-function/setup accounting. No whole importance family retired. Original Newton/component ports remain separate and unimplemented in this folder. Root should remove active T20 bounded claim and register results; any subsequent scope must be claimed first.
