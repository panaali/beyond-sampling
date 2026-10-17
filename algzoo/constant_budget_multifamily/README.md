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
Bounded study complete: all90 family/budget/source-seed fits meet the unchanged original first-order gradient criterion. This does not prove global optimality. Original failures and each separately frozen rescue are preserved. Independent T55 source/theory checks passed; final artifact arithmetic review requested. Only DRI may confirm the result.

## Results
All three H32 source seeds have20-feature position-aware reconstructions above90% of the chance-to-model accuracy gap; all three H16 seeds miss that criterion. Final held-out accuracies (%):

| H | Source seed | Original model | Fourier20 | Position polynomial20 | Rank values20 |
|---|---|---|---|---|---|
| 16 | 0 | 91.0710 | 73.6810 | 73.6825 | 15.2555 |
| 16 | 1 | 88.0300 | 70.6290 | 70.6235 | 17.4025 |
| 16 | 2 | 94.8485 | 78.1815 | 78.1820 | 19.1585 |
| 32 | 0 | 99.4380 | 93.5110 | 93.5185 | 19.4815 |
| 32 | 1 | 99.3270 | 90.8775 | 90.8720 | 19.2140 |
| 32 | 2 | 99.1370 | 92.5955 | 92.6280 | 19.1760 |

The Fourier20 gap recovery ranges77.70–80.36% for H16 and90.54–93.37% for H32. Different source accuracies/representations and total coefficient counts accompany the width change; this is not a causal width experiment. These zoo seeds differ from the historical example model. Per-model/family/budget confidence intervals, R2 counts, and coefficients are in the artifacts; intervals concern finite held-out input sampling conditional on the fits, not training-seed generalization or qualitative legibility.

`final_summary.json` uses the prescribed original -> QR -> active-row -> null-preserving correction sequence only when the preceding fit failed its training-gradient criterion. No test-based choice. The initial90 fits had25 unresolved criteria. QR resolved11; active-row least-squares with guarded line search resolved11 of the remaining14; a null-preserving minimum correction and capped unfinished-step continuation resolved the last3. The two rank-deficient stalls arose because a minimum-norm target changes unconstrained active-null components. All historical outcomes remain in `results_H*.json`, `qr_H*.json`, `active_H*.json`, their summaries and Git checkpoints. No convergence threshold was relaxed.

Independent scalar feature controls agree <=1.56e-15; the complete Fourier/polynomial position spans agree to1.33e-15 and have the same representational capacity. Their fitted endpoints need not coincide exactly because the optimization is nonconvex.79 exact-rational line controls agree <=8.9e-16; floating candidate ranking is not an outward-certified line minimum. Source/model hashes were checked before each rescue, and earlier artifacts remain immutable.

A separate post-hoc exchangeability proof establishes a20% population-accuracy ceiling for ANY predictor invariant under permutations of the first9 input positions. The rank-only family has this invariance. This is a genuine scoped information obstruction, not a claim against other compact descriptions or ARC estimators; see RANK_SYMMETRY.md and all-permutation T3..7 controls. Attainability refers to an unrestricted invariant classifier, not necessarily this parametrization.

Reproduce with the one-thread environment and shared locks: `check.py`, `qr_check.py`, `active_check.py`, `rank_symmetry_check.py`; `run.py --H H --seed S` for each frozen source; then `qr_rescue.py` for each source, `active_rescue.py` for the four sources with remaining failures, and `correction_rescue.py --H 32 --seed 2`. Existing output files are continuation checkpoints; use a fresh copy of the folder without result artifacts for a full refit. Audit summaries with `summarize.py`, `qr_summary.py`, `final_summary.py`. Protocol files freeze each added phase before its measurements.

## Compute accounting
One thread,2GB; shared compute lock and <=110CPU per process. Recorded numerical CPU: original365.05948, QR228.05855, active5.39025, correction0.83594; total599.34422seconds. This includes data preparation and most checkpoints, but excludes interpreter imports, short controls and tiny final-write overhead. There are23 source preparations,4,876,000 hidden-sequence passes including all repeated training/test targets, plus9,216 independent NumPy recurrence checks. QR records142,604 objective/gradient evaluations; active stage records793 least-squares solves. Per-panel original `model_hidden_forward_sequences=212000` means per preparation; summary multiplies by attempt count to charge repeated setup. Full instruction/FLOP accounting is not claimed.

Every predictor has exactlyK*H fitted inference coefficients, with fixed universal feature algorithms and no per-neuron selected family or stored trained normalizer. Training-only QR is converted back to original beta. Family/algorithm identifiers and coefficient precision remain structural information; fitted-scalar budget is not total description length. T25 independently audits a concrete structural encoding. No source-network training, LP solves, or equal-compute sampling comparison.

## Claims
[measured] All90 fits meet the frozen first-order criterion after the recorded bounded repairs.
[measured] Six20-feature position-aware H32 predictors exceed90% gap recovery on three fixed source seeds; tested H16 predictors do not.
[inferred] The full Fourier/polynomial families have identical spans; endpoint differences are fitting/numerical effects, not different full-budget capacities.
[inferred, proved under iid continuous inputs] First-nine permutation invariance limits any such classifier to20% accuracy at T10.
[inferred] Neither these reconstructions nor their fitted formulas solve the ARC deductive accuracy-estimation problem.

## Alternatives not ruled out
Other families, nonlinear formulas, better optimization, more training data and fresh blinded investigators remain open. This finite study cannot exhaust compact descriptions.

## Failure modes
ReLU activation regression is nonconvex; first-order convergence is not global optimality. Rank-only features cannot encode first-nine position identities. Full-span fits can differ numerically because of conditioning. Fitted features do not establish causal interpretations. Fixed held-out noise intervals exclude feature search, source-seed and training uncertainty.
