# Four cumulants cannot identify every argmax law

This is an exact limitation on an unrestricted finite-moment endpoint, not
an impossibility theorem for the specified Gaussian-input ReLU RNN family.
It helps distinguish a missing assumption from an implementation defect.

Take Y on nodes(-3,-2,-1,1,2,3), with probability numerators
P+=(4,9,0,10,1,6)/30 and P-=(6,1,10,0,9,4)/30. Both laws have moments
through degree4 equal to(1,0,14/3,0,98/3). Their cumulants through order4
are(0,14/3,0,-98/3). This follows because their signed difference is
proportional to(-1,4,-5,5,-4,1), which annihilates powers0 through4.
Yet P+(Y>0)=17/30 and P-(Y>0)=13/30.

Set three logits to(Y,0,-10), then add independent Uniform[-.1,.1] noise
to each coordinate, independent of Y. The resulting laws are atom-free,
full dimensional, with positive-definite covariance diagonal
(467/100,1/300,1/300). Independent addition preserves equality of all joint
moments, hence cumulants, through order4. The noise cannot change a winner:
the first two unperturbed logits differ by at least1 and each moves by at
most.1; the third is far below both. Winner vectors therefore remain
(17/30,13/30,0) and(13/30,17/30,0).

Any endpoint receiving only those identical first four cumulants must return
the same result for both laws. The triangle inequality forces maximum
coordinate error at least1/15 on at least one law. This is not specifically
an Edgeworth defect, nor caused by a positive-mass tie or singular covariance.
Additional support/mixture information or restrictive law assumptions can
resolve it. Approximate methods can remain useful on their intended family.

`check_four_moment_ambiguity.py` verifies all35 three-variable mixed moments
of total degree<=4 with exact Fraction arithmetic, nonnegative weights,
winner vectors, and the lower bound. No neural input sampling, upstream
endpoint code, or numerical integration is used. The compactly supported
continuous-density construction is not claimed to be realizable by the
fixed challenge RNN; extending the obstruction to that class would require
a separate construction or theorem. No such extension is inferred here.
