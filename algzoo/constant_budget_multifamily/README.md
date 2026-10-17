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
All90 original fits and25 prescribed QR rescues complete.76/90 final cells meet the frozen gradient criterion;14 remain unresolved. The requested fully converged sweep is therefore incomplete. Independent T55 theory/source review145d6db passes full-span and parameter-count claims.

## Results
Original frozen phase:65/90 fits satisfy the declared first-order criterion;25 remain unresolved after the maximum four segments. Every solver status and failed criterion is retained. Independent scalar feature controls agree <=1.56e-15; full position-basis span residual1.33e-15, transform condition11.87.

At20 coefficients/neuron, Fourier reconstruction for H16 seeds0/1/2 is73.681%,70.629%,78.181% (gap recovery78.55%,77.70%,80.36%). H32 is93.5185%,90.8945%,92.596% (gap recovery93.38%,90.56%,92.66%). These H32 feasible predictors exceed the90% reconstruction criterion even though their fits are not converged; this is a reconstruction existence result, not qualitative neuron legibility or ARC deduction. H16 failure is specific to the tested fits/families, not a capacity impossibility. Full-span polynomial endpoints differ from Fourier accuracy by at most0.029 percentage points. Rank-only20-feature reconstruction is15.26–19.48%, with every20-feature rank fit unconverged. Source models are zoo seeds0/1/2; H16 source accuracies91.071%,88.030%,94.8485% differ from the historical example model.

`summary.json` preserves all per-model/family/budget values and convergence flags. Reproduce one model with `python run.py --H 16 --seed 0` (existing completed panels are preserved; use a fresh folder for refitting), controls with `python check.py`, audit with `python summarize.py`. QR follow-up is separately frozen in QR_PROTOCOL.md; original files are immutable during it.

QR rescue resolves11 of25 failed criteria, so76/90 final fits pass;14 remain unresolved (one H16,13 H32). All6 rank-only20-feature fits now pass the same first-order criterion and remain poor predictors (15.26–19.48%). The fixed full-span H32 reconstruction successes persist. QR changes test accuracy by at most0.040 percentage points across attempted cells; no threshold was relaxed. `qr_summary.json` reports original-plus-rescue curves without test-based selection. This finite optimization failure does not show that converged fits are unavailable.

## Compute accounting
One thread,2GB, shared compute lock and <=110CPU per model cycle. Include input generation/sorting, forward targets, all optimization objective calls, reconstruction/readout work, setup and checkpoint time. Original phase365.05948 recordedCPU and1,908,000 hidden sequence passes (nine process preparations, including resumed models), plus9,216 independent NumPy recurrence checks. QR phase228.05855 recordedCPU and1,908,000 additional hidden passes (nine preparations), plus142,604 recorded objective/gradient evaluations. Combined593.11803 recordedCPU, excluding Python imports/control scripts and tiny final write overhead. Raw per-panel original `model_hidden_forward_sequences=212000` is per preparation; summary explicitly multiplies by attempt count so repeated setup is charged. No source-network training or LP solves. This research is not an estimator-cost benchmark; no equal-FLOP sampling claim.

## Claims
[measured] Six20-feature Fourier/polynomial H32 predictors exceed90% gap recovery on the three frozen source seeds; their convergence failures remain explicit.
[measured] QR resolves11 original failures without relaxing their criterion.
[inferred] This study remains incomplete as a fully converged sweep; unresolved numerical fits cannot establish a feature-capacity limit.
[inferred] Equal full feature spans distinguish basis/conditioning effects from different representational capacity.

## Alternatives not ruled out
Other families, nonlinear formulas, better optimization, more training data and fresh blinded investigators remain open. This finite study cannot exhaust compact descriptions.

## Failure modes
ReLU activation regression is nonconvex; first-order convergence is not global optimality. Rank-only features cannot encode first-nine position identities. Full-span fits can differ numerically because of conditioning. Fitted features do not establish causal interpretations. Fixed held-out noise intervals exclude feature search, source-seed and training uncertainty.
