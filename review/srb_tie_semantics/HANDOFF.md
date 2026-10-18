# T210 local conditional tie repair handoff

Claim50d5a5af; frozen source88e3db94 hardened7df279b1; independent source/theorybc20b0f9/shared03912438; gate0b68b425/sharedb1949b3c published before sole raw9a3836d1; independent saved e0c73bc7/sharedb930d0b3. Original T204 zero-readout Gaussian records57b3679d independently40ea5646 reused once.

Proposed local change in rb9.Q only:

`never = (dB == 0) & ((dA < 0) | ((dA == 0) & (np.arange(A.shape[1])[None, :] < tgt[:, None])))`

The former dA<=0 test incorrectly excluded ties with later labels. Earlier competing labels still own identical ties. For nonzero slope, strict/weak endpoints have zero probability under the two continuous conditional laws. Source extraction/reversal proves all other function expressions unchanged. No shared source was edited.

All12 fixed records/24 conditional prefixes pass independent exact halfline winner and80-digit Gaussian erfc/Student incomplete-beta references, maximum error6.182508e-17. Unique/sign-switch non-tie outputs preserve bits. The inherited zero-readout prefix with true second-largest index0 changes0→Phi(1)=.8413447460685429. A negative all-OFF halfline restores.5 probability in its applicable fixture. Earlier-label identical competitors remain excluded.

Author22 toy batch Q calls/44 prefix evaluations,24 analytic reference evaluations,.224005195CPU; saved reviewer.059999273CPU, no candidate/model/quadrature calls. Two inherited original prefix values reused, no original radial baseline manufactured. Added source expression work: two comparisons/two Boolean operations per N×10 interval array plus index creation; no complete native/runtime price.

Scope: analytic fixtures with zero recurrence matrices, exact0/unit-scale slopes and strict prefix inputs; proof addresses constant affine margins on valid pieces. Existing1e-12 gate cutoff, input-tie convention, root/CDF numerical accuracy and nonfinite-domain issues remain. No trained-bias/general correctness, historical ranking, sampler competitiveness or ARC solution claim. DRI may adopt only as a scoped semantic repair, retaining these caveats. Registry update requested.
