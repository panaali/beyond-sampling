# One-step global margin-state identifiability

Fix H hidden units, invertible square recurrent W, nonzero input weights w_i,
and readout margin columns v_i in R^(T-1). For a represented incoming state
h=Mx the future margin function is F_M(x,z)=sum_i v_i ReLU((WM)_i x+w_i z).
This is a statement about real arithmetic and all x,z, not the feasibility of
M on an incoming recipe region or floating execution.

Normalize u_i=((WM)_i/w_i,1). Every hyperplane has fixed final coefficient1.
Writing ReLU(w_i u_i y)=|w_i| ReLU(u_i y)+w_i u_i y when w_i<0 (and just
the first term otherwise), the hinge coefficient is c_i=|w_i|v_i, independent
of M. Group all hidden units sharing a hyperplane. The canonical vector
gradient-jump coefficient there is the sum of c_i over that hidden-unit subset.
Linear residues have no jump. Distinct hyperplanes have generic points away
from the other finitely many hyperplanes; hence equality of global functions
forces equal grouped coefficients at each normalized hyperplane. The scalar
input-space edge case has the same conclusion by its two slopes.

Assume every one of the 2^H subset sums of c_i is distinct. In particular no
nonempty group has zero coefficient. Equality of grouped coefficients now
identifies the exact same set of unit indices at each hyperplane. Therefore
each u_i(M)=u_i(N), so WM=WN. Invertibility of W implies M=N. The converse is
immediate. This proves no additional global margin-function quotient of the
state maps under these sufficient weight conditions.

If instead F_M=lambda F_N for a common positive scalar lambda, every grouped
coefficient on the left is lambda times the corresponding one on the right.
Summing over all normalized hyperplanes gives sum_i c_i=lambda sum_i c_i.
Subset injectivity includes the full set and empty set, so this total vector
is nonzero; hence lambda=1 and again M=N. There is no positive-projective
quotient either. This does not require a bound on M or rational M.

Failure of ANY tested condition is inconclusive. Exact reset histories may
still produce identical M and permit reuse; conditional domains may permit
different formulas to agree; identical labels do not require proportional
margin functions. Approximate merging, input-dependent score transformations,
and other horizons are not ruled out. No probability, merging cost or accuracy
claim follows. The exact finite audit uses stored rational weights and counts
its own preprocessing; it does not provide free per-model advice.
