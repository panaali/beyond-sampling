# T233 gauge-corollary addendum

PASS for frozen99dfa103, published e860abb6. This written review extends the independently reviewed main proof6c02b3aa under delegationc1343d8a. The exact source bytes and main review are bound in gauge_review.json. No implementation or experiment is cleared.

Positive diagonal D commutes with coordinatewise ReLU. Induction from zero initial state gives h′=Dh. The selected preactivations and their slopes and offsets scale by D_i and D_j; their roots, signs and boundary ownership remain identical. Consequently r′=D_j r, including a=0 or b=0. Conditioning on h′ is equivalent to conditioning on h because D is fixed and invertible. The main proof's fresh-input independence remains essential.

For every integer s≥0, entrywise absolute values give |Wh′|^s=D|Wh|^sD⁻¹, with the s=0 identity included. The absolute output contrast is the original contrast times D⁻¹. Thus the coordinate-j propagation coefficient becomes K′=K/D_j. Both K′r′ and K′E[r′|h′] equal their original versions, and K′|beta′|=K|beta| while the two roots stay fixed. The displayed density-cap bound is therefore invariant.

For row-vector ell, ell′=ellD⁻¹ and the raw second moment transforms as D E[HHᵀ]D. Substitution cancels both factors in the quadratic form. Together with the unchanged K|beta| prefactor, this verifies the complete raw-moment margin bound under its finite-moment premise.

This changes the weight parameterization while preserving the represented function. It is distinct from T229's fixed-weight automorphism. It does not make the bound small or cheap, establish numerical stability, show fewer regions, or control classification without margins. Negative/zero diagonal transformations and a different rank-conditioned fresh-input law are outside the reviewed claim.

Accounting: source byte comparison, hashing and written algebra only; zero author, numerical-library, model, random, target-array or mass calls. Future controls or target applications need separate registered scopes and review.
