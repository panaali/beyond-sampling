# Independent T67 bounded-phase interpretation

Read-only review of spatial.py, spatial_diagnostic.py and the README's final
q128/256 and target-value comparisons. No new experiment or production edit.

The weight-derived scale s_d=min_i|Wi_i/Wh_id| has the correct positive diagonal
coordinate covariance: Wi'=DWi, Wh'=DWhD^-1 implies s'_d=D_d s_d. With similarly
scaled hidden bounds, the exponential mesh transforms identically. It resolves
one-normal-standard-deviation movement of a scalar gate threshold, but neither
locates every layer nor weights it by reachable state mass. The curvature
monitor is also covariant because its probes use fractions of coordinate
bounds and invariant terminal probabilities. Its cube-root construction is a
heuristic, not an optimal-interpolation theorem: midpoint residuals already
contain the prior bin widths.

The diagnostic shows a local effect: reducing first-axis terminal residual
from.003819 to.000756 (curvature) or.001737 (gate) is evidence that mesh placement
can resolve a narrow layer. It does NOT falsify narrow layers as an explanation;
it supports them locally. Conversely it cannot prove they dominate the overall
accuracy error. The probe law is artificial, not the reachable law, and the
midpoints change with the tested mesh. A cheap stronger discriminator would
compare all meshes at the SAME frozen off-grid probes against exact terminal
values; this removes moving-probe selection while still not proving global
importance. The current global conclusion is correctly more cautious.

For input truncation[-L,L], let Z=Phi(L)-Phi(-L), F_L the normalized CDF.
A target rank of second-largest has q=F_L(u)~Beta(T-1,2). Thus target-value
integration density is T(T-1)F_L(u)^(T-2)(1-F_L(u))*phi(u)/Z. On a smaller target
interval[-A,A], omitted mass is the Beta lower tail at F_L(-A) plus upper tail
at F_L(A). Retained constant-class correctness is (1-omission)/T. Adding
Z^T*omission to the original-input box uncertainty1-Z^T is valid; neither
normalizing the retained target rule nor silently dropping this mass is valid.
The implementation's reported bounded controls conform to these formulas.

The full-u rule's row-normalization failure at extreme q is a floating-tail
API failure, not a failure of the measure identity. Restricting targets to[-4,4]
while retaining input support[-6,6] is an honest bounded workaround. Its T4
omission6.0179e-9 is much smaller than the observed1e-4-scale oscillations.
Thus endpoint reparameterization ALONE did not stabilize the tested scheme.
It does not rule out endpoint treatment combined with interior panel splitting.

Finite checklist independently accepted:
- normalization and elementary controls pass; they do not certify trained rows;
- local terminal mesh benefit is established under the diagnostic probe law;
- global rankings still change under q128/256 and bounded u32/64, so no winner;
- interior target/grid-crossing kinks remain untested and plausible;
- earlier-stage interpolation and reachable-mass weighting remain unresolved;
- fixed-target independent references or explicit kink panels would be distinct
  falsifiers; more blind resolution sweeps are not required by this evidence.

The scoped conclusion is INCONCLUSIVE global accuracy/cost benefit. Neither
rank-conditioned transport nor gate-aware spatial refinement is dead. Parent
has retained the interior-kink followup in T67 Ready; no new task is claimed
by this memo.

## Followup: complete breakpoints of the nonfinal-target interpolant

Read breakpoints.py, panel.py and StagePieces in polynomial.py. For fixed
stage-t grid node h, target insertion uses relu(Wh+wi*u). Its gate and next-grid
crossings occur at the SAME scalar endpoints stored by StagePieces for that
node and stage. Thus the union over every stage also covers target insertion;
it is not restricted to ordinary-input expectation steps.

Ordinary conditional transition matrices change active integration formulas
only when u crosses a stored full-input interval endpoint. For nonfinal target
labels, terminal correct intervals are independent of u; intersecting them
with above/below-u support adds their endpoints. Between union knots, terminal
functions, interpolation weights, Gaussian CDFs and conditional matrices are
analytic in u. Backward induction combines them by sums/products/division by
q and1-q, which are positive on the retained target interval. This introduces
no new mathematical kinks. Chronology is fixed by the chosen label and adds
no branch changes. Rejecting any possible target-last label is necessary for
this proof; the current dominance-based label screen is conservative.

The tiny-panel bound multiplies omitted width by a valid upper bound on target
density: separate maxima of the Beta polynomial and phi/Z are conservative.
It bounds the underlying probability interpolant, not arbitrary floating
overshoots. Completeness refers to exact-arithmetic topology of the finite-grid
method; fallback implementation branches, roundoff and inverse-CDF restoration
are not certified. Panel GL2/4 agreement is a useful convergence check, not a
quadrature-error theorem. The reported h9 terminal-only curvature/gate meshes
are worse than baseline after outer refinement stabilizes, so outer oscillation
no longer explains that particular h9 negative. Higher meshes and different
spatial refinement remain open.
