# T247 independent saved-grid audit

## Question
Do the frozen stored arrays and reported centering diagnostics agree, without repeating the producer's table construction?

## Method
Pre-output checker3ebb7a2c/sharedc68151c7 independently decodes NPY bytes using stdlib, checks source/gate/archive bindings, contracts exact dyadic CDFs and integrates CDF trapezoids by parts to recover means. All source/staged cumulative and local alternative scalar expressions,15 progress events and declared call inventory are checked. Finite interpolation probes are diagnostics only. Producer6d2db5f4, gate89c208e1/shared9bff1348.

## Status
Complete/pass. Four cases, eight law records,15 events and18 finite probes pass once. No producer rerun or reviewer failure.

## Results
The source T2 stored-array inverse-linear law has central gap +2.8209479203455956e-5 and mean -5.000000028635739e-5. The local trapezoidal law has central gap +3.319566843629218e-14 and mean -3.382508134159753e-13. Both retain27,891 zero-CDF-increment intervals among160,000. Grid reflection is not exact (maximum stored dyadic discrepancy1.7763568394002505e-15). All three tiny controls agree, including the asymmetric example where trapezoidal weighting does not center the law. Exact fractions are in results.json.

Reproduction command (saved-only, results immutable): `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash -c 'ulimit -v 2097152; ulimit -t 20; flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/feynman_fundamentals/t247_saved_review/check.py'`. Existing results block another run.

## Compute accounting
Independent saved audit1.239037275 process CPU seconds. Producer main1.075249376 CPU seconds, actual reported soft/hard limits20/20 seconds and2GiB/2GiB. Reviewer used the same external limits. No NumPy/SciPy/author imports, model, grid generation, exp/ndtr/erf/interpolation, RNG, target or training calls. Exact dyadic decoding and arithmetic are review work, not free estimator work.

## Claims
[measured from saved arrays] The local alternative greatly reduces these two centering diagnostics for this one table. [inferred] The exact-real inverse-linear law of stored arrays differs from an actual finite RNG/Sobol program. No Gaussian density certificate, actual target-accuracy effect, broad method negative, shared-source adoption or full-cost comparison follows.

## Alternatives not ruled out
Library density outputs are retained, not independently recomputed or certified. Rounded erf normalization remains diagnostic. Finite endpoint/flat-interval behavior is not a proof of the entire floating input law. Other grids, target integrands and perturbations were not tested.

## Failure modes
Conflating centering with normalization/accuracy, or treating the trapezoidal law as exactly Gaussian, overstates this result. Flat intervals carry zero continuous-latent mass but can affect finite endpoint ownership. Complete provenance and source-call inventory do not prove all runtime/library behavior.
