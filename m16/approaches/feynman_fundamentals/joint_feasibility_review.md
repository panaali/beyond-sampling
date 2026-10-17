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

## Conditioning repair and exact saved-witness audit

The thin feasible control initially produced no acceptable floating
primal or exact dual. Reporting UNKNOWN was correct; the row-scaled
LP result was not promoted to a mathematical negative. The frozen
column rescue uses positive exact scales. A proposed primal is divided
by the column scales before checking original integer inequalities.
Column scaling preserves the dual nullspace; reconstructing against
the exact row-normalized matrix, dividing multipliers by row scales,
and renormalizing their sum yields a valid original-row certificate.
An oversized/nonbasic support or failed rational reconstruction stays
UNKNOWN, not infeasible.

`check_joint_certificates_review.py` imports no author map, solver or
certificate code. It rebuilds constraint rows from true Fraction model
weights, including gate/rank/readout ownership, and matches all168
stored systems exactly. It verifies96 strictly feasible rational
witnesses and72 nonnegative, normalized, exactly balanced duals.
All96 primal inputs also reproduce their saved history, top/second
ordering and correct/wrong event under direct Fraction recurrence.
CPU .1003 seconds; eight distinct history maps are represented.

The240-LP cap leaves seven complete histories and a partial eighth.
The panel is explicitly incomplete. Mixed open witnesses establish
both outcomes inside tested histories but do not quantify their error
mass; selected gate mass and resolved accuracy coverage remain distinct.
No whole gate was resolved by this bounded panel. Tighter conditioned
mass integration remains a separate open task.
