# Scope and exact identities

The bias-free recurrence is h_t=ReLU(Wh h_(t-1)+Wi x_t), h_-1=0,
z=Wo h_(T-1). We read no sigmoid, bias or learned numerical advice.

On a strict activation region, the input-j logit Jacobian is
Wo D_(T-1) Wh ... Wh D_j Wi. Therefore every output-pair margin derivative
has magnitude bounded by |Wo_a-Wo_b| |Wh|^(T-1-j) |Wi|. The maximum over
output pairs is U_age. This bound also gives a global coordinate Lipschitz
bound: integrate piecewise derivatives along a line, including its finite
breakpoints (or use the ReLU Lipschitz recurrence directly). Consequently
sum_j U_(T-1-j)|delta x_j| bounds each margin perturbation globally.
This does not imply a merge when the existing margin is unknown/zero or the
prefix variation is unbounded. Gaussian inputs have no finite global box.

The exact local sensitivity is the range of entries of column j of the final
logit map. Integer recurrence with common dyadic weight denominator D stores
h_t and its map with denominator D^(t+1), and final logits D^(T+1).
A nonzero gate form B with Bx=0 makes the witness unresolved. Identically-zero
B is OFF throughout that local region and contributes zero. For every other
form, r<=|Bx|/(2||B||_1) preserves its strict sign under an L-infinity radius r.
Induction across steps proves the recorded map on this box. Adding one quarter
of the minimum input rank gap to the radius minimum preserves true rank.
No gate-boundary Jacobian is claimed; saved chosen-branch maps are descriptive.

Positive diagonal neuron rescaling and neuron permutation preserve both local
logit derivative spreads and U_age, since absolute positive scale factors cancel.
Adding an identical row vector to every readout row also preserves margins.

A growing absolute-weight envelope is only an inconclusive bound, not evidence
that every trajectory retains information. A strict witness with a nonzero old
sensitivity proves local dependence, not a probability, average decay or obstacle
to state-dependent merging. A fixed four-step panel cannot establish asymptotic
decay/nondecay. Other charts, reset events, reachable-state restrictions and
margin-conditioned merging remain open.
