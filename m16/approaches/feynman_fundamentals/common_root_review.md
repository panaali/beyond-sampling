# Mixed-precision circle discrepancy: common first-step factor

At step1, neuron j has preactivation Wi_j*(U0+t*V0). For nonzero Wi_j all
neurons share exactly the root -U0/V0, while the active side depends on
sign(Wi_j*V0). Wi_j=0 is identically inactive. This is an algebraic identity,
not a tolerance for merging nearby genuinely distinct roots.

mixed.cpp first rounds both Wi_j*U0 and Wi_j*V0 to double, then divides them
in long double. The promoted ratio exposes distinct errors in the rounded
products instead of canceling the shared factor. Stable long-double products
can also create a few different roots, but far fewer. Each chart sees the
same root-count difference, hence a predicted extra piece count twice the
difference in distinct first-step ratios.

Independent check_common_root_review.py reconstructs only the32 existing
fixed planes and the already tested weights. It computes the two modes'
first-step ratios and compares the count prediction with saved full walks.
No network walk or model input forward is run.

Results: predicted differences match all96 H16 model/plane pairs exactly,
including24,26,28,30 extra pieces. This independently isolates a concrete
common-factor precision mechanism rather than leaving a generic cancellation
hypothesis. For H2, only3/32 pairs match exactly;29 retain residual0–5 extra
pieces after subtracting the first-step explanation. Later rank-deficient
or cancellation events remain unexamined. Integral agreement alone does not
establish topology equivalence in either mode.

A separately frozen first-step-factored implementation is justified by the
identity above; retain both prior implementations and every failed count
criterion. If it changes the long-double reference's artificial first-step
subdivision too, explain that comparison explicitly rather than redefining
success after seeing results. This does not certify arbitrary downstream
floating roots or justify epsilon merging. Owner received findings before
implementation. .014492CPU after imports, zero forward calls. Raw counts
and unchanged saved-result hash in common_root_review_results.json.
