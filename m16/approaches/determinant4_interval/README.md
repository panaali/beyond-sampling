# T144: counted determinant and outward interval

## Question
Can a factored4x4 determinant reduce known scalar arithmetic, while a separately counted outward interval makes rounding uncertainty explicit for the fixed T4 cone corpus? Claim e946452 pushed before implementation, coordinated with T143 owner34. This is a primitive review, not a mass algorithm or shared edit.

## Method
Freeze original6347 matrices at `/workspace/work/recover-small_h2_34/m16/approaches/class_region_t4/quadratic_exact_mass_calls.npz`, SHA07a0720e229bd2e0aeda511b60038efdc37f76c17aadfdeb17cf0caf3f264581. Retain existing16 setup fixtures0,2115,4231,6346,3222,3219,3221,3224,5962,3251,3254,3836,2567,2577,5448,2020.

For the six column pairs01,02,03,12,13,23, let s_ij be the2x2 minor from rows0,1 and c_ij that from rows2,3. Evaluate det=s01*c23-s02*c13+s03*c12+s12*c03-s13*c02+s23*c01. Twelve minors take36 multiply/subtract operations; six final products and five signed sums make47 scalar operations. Compare the pinned prior24-permutation95-operation evaluation without changing its summation order. No claim that the new summation always improves numerical error.

Independently evaluate the same factored identity with closed intervals. Point inputs are exact binary64 entries. Each interval multiply uses all four endpoint products, min/max, and nextafter toward minus/plus infinity; addition/subtraction similarly rounds each endpoint outward. Reject NaN/nonfinite intermediates explicitly rather than silently accepting a signed determinant. Underflow is retained with widened subnormal endpoints; intervals containing zero remain sign-ambiguous, never certify zero volume. Report arithmetic, comparisons, finite checks and nextafter calls separately; interval validation is not free. These inclusions assume the ordinary IEEE754 operations and nextafter behavior validated on controls.

Before trained matrices freeze analytic controls: zero/identity, all24 row permutations, duplicate rows, dyadic triangular matrices with known determinant, orthogonal Hadamard times a2^-k final-row scale for k=10,30,45,52, and diagonal2^300/2^-300 extreme scales. Exact Fraction Gaussian elimination is the independent reference, including for every6347 stored matrix. Save exact determinants, both floating errors, intervals/relative widths, sign status and all failures. Fixed analytic determinant-known cases additionally guard the independent reference. No selection from observed new errors. Process corpus in four fixed chunks of at most1600 matrices, <=20CPU per chunk; preserve cap outcomes.

## Status
Protocol frozen; implementation and controls next.

## Results
None yet.

## Compute accounting
One thread,2GB, review lock<=20CPU per process. Instrument scalar arithmetic for47/95 formulas independently of outward interval work. Count and time exact rational reference separately. Full process CPU includes imports, validation and persistence, except final write/return overhead. No Gaussian integration, neural forward passes or random model inputs. Arithmetic savings in one determinant do not imply a competitive full estimator.

## Claims
[inferred] Laplace expansion by first two rows yields the47-operation identity. [speculative] The separate outward interval may certify signs and bound determinant uncertainty on this corpus without changing cone geometry.

## Alternatives not ruled out
Adaptive error-free expansion, row/column pivoting, compensated summation and interval radial/mass arithmetic may give tighter or cheaper bounds. Point identity47 and interval primitive have separate costs; no universal accuracy ranking. The interval applies to stored binary entries only, not errors in the rays or downstream integration.

## Failure modes
Overflow causes explicit unsupported status; underflow/cancellation can make intervals uninformative. IEEE/environment behavior is an assumption, not a platform-independent proof assistant certificate. A correct determinant interval alone cannot certify a Gaussian cone mass or topology. Existing point95 and new point47 can both lose relative precision on thin cones.
