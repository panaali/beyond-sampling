# T23: fixed-family, multi-seed reconstruction curves

## Question
Does the earlier fitted-scalar budget sensitivity survive improved solver convergence, different fixed feature dictionaries, and three source-model seeds? Task claim 8923ec6 was pushed before this folder was created. This is activation reconstruction, not an ARC deductive estimator.

## Method
Freeze six zoo models: H=16 and32, T=10, training2^30, seeds0,1,2. Use existing public checkpoints and record hashes. Shared independent Gaussian training/test inputs: 12000 seed72301 and200000 seed72302. Float32 model forward targets, float64 reconstruction arithmetic. Fit only final hidden activations; original readout is used only for reporting held-out reconstruction accuracy.

Freeze three nested dictionaries of20 features and budgets1,3,5,10,20:

1. Fourier: original T7 dictionary, m1,m2,x9; gap times cos/sin frequencies1..4 of the top-position among the first9; m2 times the same frequencies of the second-position; m2*cos(2pi*(p1-p2)/9).
2. Position polynomial: same first3 and last interaction; gap times degrees1..8 Legendre polynomials at2*p1/8-1; m2 times degrees1..8 at2*p2/8-1. Each polynomial is centered and unit-RMS on the fixed nine-position grid (universal constants, no data-dependent scaling). Full20-column span equals Fourier, a built-in basis/conditioning discriminator; smaller prefixes differ.
3. Rank values: first3; first-nine sorted values of ranks3..9; absolute first-nine sorted values of ranks1..9; abs(x9). This family omits the first-nine positions deliberately and tests whether stronger rank-value flexibility alone suffices.

Exactly K*H real fitted coefficients for a ReLU(Phi_K*beta) hidden approximation. No selected feature subsets, per-neuron dictionary choices, trained offsets, learned normalizers or test-driven choice. Report each family/seed curve separately, not a best-family selected score. Structural information is three universally specified algorithms, fixed K list and index/rank/frequency enumeration, plus a family identifier if a future consumer selects a family; fitted-scalar count is not total bit-description length.

For every cell: least-squares initialization, deterministic L-BFGS-B of rectified activation MSE, max2000 iterations per segment, ftol1e-15 and gtol1e-8, at most four segments. Stop only at infinity gradient <=1e-7*max(1, initial infinity gradient), preserving other termination codes and objective/gradient diagnostics. Continuation only uses training loss, never held-out accuracy. Solver convergence means this first-order criterion, not global optimality. Preserve any cap, failure, finite gradient or fit worse than initialization. Run one source model per bounded cycle; checkpoint each fit. Resource continuation may resume a partially saved model: skip completed fits, recompute only scoring needed for paired differences, and stop before a new fit after45CPU. This changes process boundaries only; no coefficient/solver criterion changes. Count repeated preparation and preserve interrupted attempts.

Report held-out accuracy/Wilson interval, paired adjacent-budget differences, source accuracy, chance-to-model gap recovery, neuron R2 counts, solver status, coefficients and all feature/fit/model pass costs. Intervals are conditional on these fits; three source seeds do not imply population confidence. Independently check feature construction, fixed basis-span equivalence, coefficient counts and arithmetic summaries.

Alternatives: (1) old cap-limited fitting explains the curve: compare frozen initial/first-segment/final training diagnostics, not just accuracy; (2) feature order matters: compare fixed prefixes and equal full-span Fourier/polynomial endpoints; (3) source seed explains apparent width effect: report all three paired seeds; (4) output sensitivity differs from activation error: report both metrics. No causal neuron-role claim follows from reconstruction.

## Status
Protocol frozen before fitting; implementation next.

## Results
None yet.

## Compute accounting
One thread,2GB, shared compute lock and <=110CPU per model cycle. Include input generation/sorting, forward targets, all optimization objective calls, reconstruction/readout work, setup and checkpoint time. No source-network training or LP solves. This research is not an estimator-cost benchmark; no equal-FLOP sampling claim.

## Claims
[speculative] Budget sensitivity persists after controlling solver status and source seed.
[inferred] Equal full feature spans distinguish basis/conditioning effects from different representational capacity.

## Alternatives not ruled out
Other families, nonlinear formulas, better optimization, more training data and fresh blinded investigators remain open. This finite study cannot exhaust compact descriptions.

## Failure modes
ReLU activation regression is nonconvex; first-order convergence is not global optimality. Rank-only features cannot encode first-nine position identities. Full-span fits can differ numerically because of conditioning. Fitted features do not establish causal interpretations. Fixed held-out noise intervals exclude feature search, source-seed and training uncertainty.
