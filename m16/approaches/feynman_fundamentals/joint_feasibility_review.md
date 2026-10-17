# T120 joint strict-feasibility certificate protocol

Gordan's alternative is appropriate for homogeneous gate/rank/readout
constraints: exactly one of x with A*x>0, or nonzero lambda>=0 with
A^T*lambda=0 exists. Normalize the latter by sum(lambda)=1. Exact
rational verification of either witness is decisive; a floating LP
status alone is not. Any box normalization used to find x must not be
mistakenly retained among the rows of the dual Gordan certificate.

Identically zero gate forms require OFF semantics and must be removed
before asking for strict inequalities. They are full-volume equalities,
not zero-measure boundaries. Identical earlier-index readouts reject
the candidate, while later identical readouts do not. Proper nonzero
margin boundaries may be excluded for positive Gaussian-mass existence.

Selecting ten histories by previously frozen gate mass determines the
diagnostic panel; it does not make a feasible point's whole history
uniformly correct/wrong or automatically resolve that mass. Uniform
classification requires certifying all competing joint alternatives
infeasible. This is a protocol/algebra review; source and exact saved
witness checks remain pending when the implementation is available.
