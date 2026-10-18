# T238 independent centering review

## Question and status
PASS for frozen2d24feb8/shared45745b98 under delegation9d8db86d. Does the formal source-style cumulative/inverse interpolation preserve the symmetry of positive even density ordinates? This review checks the exact-real proof, source linkage and six frozen documents. It does not execute a grid or authorize numerical controls/source adoption.

## Method and independent derivation
The source subtracts its first cumulative value before normalization. On a uniform exact-real grid this assigns probability d_j/S to interval j, for S=sum from1 toN. Affine inverse interpolation of a continuous uniform coordinate makes each such interval uniform, so its mean is the midpoint, not the right endpoint. Positivity makes the CDF strictly increasing; endpoint choices are null only for this continuous latent law.

Pairing symmetric ordinates gives S=2L+d_m+d_N while the intervals below zero carry L+d_m. Subtracting one half proves the stated (d_m-d_N)/(2S). For the formal Gaussian at A8, d_N=exp(-64)*d_m and S<=N*d_m. Since exp(64)>2, the gap exceeds1/(4N)=1/640000 for N160000. This proof evaluates neither S nor the Gaussian numerically. The half-line event gives the same lower bound on sup-event total variation from either a full or symmetrically truncated normalized target law. It is not a fibre error witness.

The complete symmetric endpoint weighted sum is zero. Omitting the left endpoint leaves A*d_N. Dividing by S and subtracting Delta/2 proves the continuous mean formula. If all interior ordinates are at least the edge and one is larger, S>N*d_N and the mean is negative. Constant ordinates have zero mean and zero central gap. The Riemann-sum limit Delta*S -> integral d proves both displayed asymptotic coefficients on a fixed compact interval. It provides no directed finite-grid remainder.

Trapezoidal interval weights reverse under reflection. They therefore assign the same mass to reflected uniform intervals, proving symmetry, zero mean and central CDF one half. Equal endpoints give W=S directly. This restores a property of the continuous interpolated law, not its full analytic density: the finite-grid law remains piecewise constant, whereas a nonconstant Gaussian is not constant on any positive-width interval.

## Source and provenance
All six files match the frozen source commit. The three inherited payloads match their manifest lengths/hashes; original core.py also matches its60967f89 Git bytes and blob. Its T2 kind12 formula is the stated Gaussian, with formal endpoints/count -8,8,160001. The source base cumulative integral is Delta*(S+d0); actual Z additionally includes the kind's combinatorial position multiplicity. Its inverse-law normalization excludes d0. The centering alternative does not by itself reconcile or certify either that integral or the final Z-weighted functional. No actual floating array is asserted symmetric.

## Results and claims
[inferred, exact-real] The stated central gap, continuous mean, Gaussian formal lower bound and asymptotic limits are correct under their positive even-grid assumptions.

[inferred, exact-real] Trapezoidal interval probabilities restore reflection symmetry only; no Gaussian-law or trained-accuracy theorem follows.

[measured, metadata only] Frozen document and inherited source-byte bindings pass. No numerical experiment occurred.

## Alternatives not ruled out and failure modes
Floating construction and finite latent coordinates can differ from the formal real-arithmetic law. Particular trained integrands may be insensitive to this mismatch. Correcting symmetry need not correct normalization, tails, walker behavior, other conditional coordinates or integration error. Special T>2 symmetric densities are not excluded by the observation that such densities are generally asymmetric. Constant-density and other integration schemes remain valid alternatives. This is not a rejection of the tie-face method or validation of a repair.

## Compute accounting
Text/Git byte comparisons, hashing and written algebra only. Zero author imports, numerical integrals, grids, array decodes, model, face, RNG or mass calls. No CPU/FLOP benefit claimed; any controls, implementation or target use require a separate published scope and gate.
