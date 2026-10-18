# Exact ceiling for the rank-only family

This is a post-hoc analytic explanation, independently checked by T55; it was not used to choose a feature family or fit.

For T iid continuous inputs, suppose a classifier sees only the multiset of the first T-1 values and the distinguished final value. Its output is invariant under permutations of the first T-1 positions. Let E be the event that the final input is second largest. E is measurable from these features (m2 < xlast < m1), and exchangeability gives P(E)=1/T. On E the target is the final index. Conditional on the invariant values outside E, each of the other T-1 positional indices is equally likely to hold the second largest input. Any invariant classifier therefore has accuracy at most

    P(E) + (1-P(E))/(T-1) = 2/T.

The unrestricted feature-measurable rule that predicts the final index on E and a fixed other index otherwise attains this ceiling. Attainability is not claimed for the particular linear-ReLU parametrization. The T23 rank-value features and any fixed output readout are invariant, so their accuracy is at most20% at T=10 under the ideal continuous-input distribution. More fitting or more invariant rank-value features cannot overcome this information loss. The measured15.26–19.48% fits respect the ceiling.

This is an exact obstruction for this explicit permutation-invariance restriction. It says nothing against position-aware compact descriptions, deductive accuracy estimators, or the full ARC challenge. Empirical runs use float32 inputs and finite samples; the theorem is about the specified continuous mathematical distribution, not a deterministic bound on finite empirical accuracy.

`rank_symmetry_check.py` independently counts every permutation at T=3..7 and verifies that the best invariant rule achieves exactly2/T. These finite controls supplement, not replace, the exchangeability proof.
