# T144: counted determinant and outward interval

## Question
Can a factored4x4 determinant reduce known scalar arithmetic, while a separately counted outward interval makes rounding uncertainty explicit for the fixed T4 cone corpus? Claim e946452 pushed before implementation, coordinated with T143 owner34. This is a primitive review, not a mass algorithm or shared edit.

## Method
Freeze original6347 matrices at `/workspace/work/recover-small_h2_34/m16/approaches/class_region_t4/quadratic_exact_mass_calls.npz`, SHA07a0720e229bd2e0aeda511b60038efdc37f76c17aadfdeb17cf0caf3f264581. Retain existing16 setup fixtures0,2115,4231,6346,3222,3219,3221,3224,5962,3251,3254,3836,2567,2577,5448,2020.

For the six column pairs01,02,03,12,13,23, let s_ij be the2x2 minor from rows0,1 and c_ij that from rows2,3. Evaluate det=s01*c23-s02*c13+s03*c12+s12*c03-s13*c02+s23*c01. Twelve minors take36 multiply/subtract operations; six final products and five signed sums make47 scalar operations. Compare the pinned prior24-permutation95-operation evaluation without changing its summation order. No claim that the new summation always improves numerical error.

Independently evaluate the same factored identity with closed intervals. Point inputs are exact binary64 entries. Each interval multiply uses all four endpoint products, min/max, and nextafter toward minus/plus infinity; addition/subtraction similarly rounds each endpoint outward. Reject NaN/nonfinite intermediates explicitly rather than silently accepting a signed determinant. Underflow is retained with widened subnormal endpoints; intervals containing zero remain sign-ambiguous, never certify zero volume. Report arithmetic, comparisons, finite checks and nextafter calls separately; interval validation is not free. These inclusions assume the ordinary IEEE754 operations and nextafter behavior validated on controls.

Before trained matrices freeze analytic controls: zero/identity, all24 row permutations, duplicate rows, dyadic triangular matrices with known determinant, orthogonal Hadamard times a2^-k final-row scale for k=10,30,45,52, and diagonal2^300/2^-300 extreme scales. Exact Fraction Gaussian elimination is the independent reference, including for every6347 stored matrix. Save exact determinants, both floating errors, intervals/relative widths, sign status and all failures. Fixed analytic determinant-known cases additionally guard the independent reference. No selection from observed new errors. Process corpus in four fixed chunks of at most1600 matrices, <=20CPU per chunk; preserve cap outcomes.

## Status
Bounded implementation and full fixed-corpus exact reference comparison complete; independent T55 review complete851e2e6.

## Results
All6347 interval evaluations enclose the exact binary-entry determinant and excludezero; no trained unsupported cases. Maximum relative interval width6.45527e-9. The47-op point formula improves absolute error in3792 matrices, worsens1701, and ties854, so there is no universal floating improvement. Its largest relative error3.21799e-10 (index4707) versus old95's3.84647e-9 (index4730); maximum absolute errors8.28462e-17 and1.06795e-16. The prior16 fixtures are explicitly marked in raw rows.

All33 analytic controls behave as declared: identity/permutation, triangular and Hadamard-thin signs/values pass; zero/duplicate and2^-300 diagonal remain sign-ambiguous;2^300 diagonal is explicitly unsupported due to overflow. Subnormal multiplication containment is independently checked with exact Fraction. No ambiguous determinant is silently called a zero-probability cone.

Reproduce from repository root: `/workspace/clean-neurons/.venv/bin/python m16/approaches/determinant4_interval/review.py controls`, then `corpus 0` through `corpus 3`, then `summarize.py`, with OMP/OPENBLAS/MKL threads1, review flock and `ulimit -v 2500000`. Every exact determinant and interval is retained in corpus_0.json through corpus_3.json. Source95 code and original rays are pinned by SHA256.

## Compute accounting
One thread,2GB, review lock<=20CPU per process. Instrument scalar arithmetic for47/95 formulas independently of outward interval work. Count and time exact rational reference separately. Full process CPU includes imports, validation and persistence, except final write/return overhead. Measured corpus CPU1.921189591seconds across four cold chunks, controls .036363311seconds; peakRSS52368KiB. Exact reference CPU was included in those totals but was NOT separately instrumented as the planned component timing requested; this accounting limitation is explicit in summary.json. No hardware speed claim. Summary/reviewer arithmetic is additional short review compute.

Point arithmetic falls95→47 per matrix (304656 fewer named scalar operations across this corpus). Each outward interval separately executes154 arithmetic,243 comparisons,340 finite checks and94 nextafter calls, including all16 input interval validations; final sign/relative-width/report bookkeeping is outside the primitive count. Counts exclude Python dispatch, allocation and hardware internals. Final magnitude abs/sign conversion is not part of either point determinant count.

No Gaussian integration, neural forward passes or random model inputs. Arithmetic savings in one determinant do not imply a competitive full estimator.

## Claims
[inferred] Laplace expansion by first two rows yields the47-operation identity. [measured] Every6347 stored binary-entry determinant lies in its outward interval with a nonzero sign, independently compared with exact elimination. [inferred] The elementary interval construction encloses the binary-input determinant under stated IEEE754 assumptions. No statement about ideal ray geometry or full mass certificates follows.

## Alternatives not ruled out
Adaptive error-free expansion, row/column pivoting, compensated summation and interval radial/mass arithmetic may give tighter or cheaper bounds. Point identity47 and interval primitive have separate costs; no universal accuracy ranking. The interval applies to stored binary entries only, not errors in the rays or downstream integration.

## Failure modes
Overflow causes explicit unsupported status; underflow/cancellation can make intervals uninformative. IEEE/environment behavior is an assumption, not a platform-independent proof assistant certificate. A correct determinant interval alone cannot certify a Gaussian cone mass or topology. Existing point95 and new point47 can both lose relative precision on thin cones.

Independent T55 review851e2e6: all6347 saved interval containments/counts/error summaries reproduced;67 original-ray exact determinants independently recomputed by24-term Leibniz expansion (different from producer Gaussian elimination), covering prior16, evenly spaced and worst cases.47-op identity and elementary outward enclosure proof accepted under stated IEEE754/nextafter assumptions. Runtime .340CPU; full floating geometry/mass certification remains excluded.
