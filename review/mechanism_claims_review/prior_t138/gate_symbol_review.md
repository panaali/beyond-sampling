# T138 fixed gate/top-three information and constructive retained margin

`check_gate_symbol_review.py` independently runs the original checkpoint
with exact binary-rational weights and dyadic inputs. Two inputs have the
same full rank order, top indices0/3/6 and values1,63/64,31/32, and all160
gate signs. Their unique network winners are3(correct) and6(wrong).
Exact global per-hidden-coordinate Lipschitz bounds satisfy every saved
gate, rank and output stability inequality with common input-box radius
8.28669939158e-10. The two lower-coordinate boxes share an open top-three
box, so this is not merely ambiguity at an isolated conditional value.
No numerical probability, practically useful error floor or estimator
accuracy claim follows from the tiny conservative radius.

Thus top-three values/indices plus complete rank and gate symbols do not
determine this network's correctness. This concerns reconstruction of the
network's correctness/output, not prediction of the true second index
(which the rank summary already contains). Under independent Gaussian
coordinates, any deterministic reconstruction of correctness from this
summary incurs positive error at least P(common top box) times the smaller
of the two lower-box probabilities. This is a formal positive bound only,
not a measured or useful-sized bound.

The reviewer derives the fixed-cell margin by reverse adjoints, independently
of the author's ten prescribed-gate basis passes. All coefficients exactly
match; every lower-input coefficient is nonzero. The segment crosses
margin3-minus6 at lambda=.6988823149957404. Independent direct exact
recurrence at the crossing retains all160 gates and gives equal logits3/6,
with every other output below by at least2.43219660145. Gate/rank cells are
convex, and output margins are affine within the fixed cell; endpoint and
crossing dominance therefore establish winner3 before the crossing and6
after it. First-index ties select3 exactly at the crossing.

One retained weighted lower-input scalar recovers correctness on this
segment. This constructive alternative remains open and prevents a broad
claim that gate-based or compact state summaries are impossible. No global
sufficiency, probability integration, parameter-budget or FLOP claim was
reviewed. Combined exact review used.0492CPU excluding checkpoint load;
no new search or LP was run. The author's64-anchor/640-LP search and
uncertified other candidates are not promoted to additional certificates.
