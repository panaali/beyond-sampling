# Fixed analytic reconstruction-control gate

Scopeab950a3c follows reviewed initial0a2614e4/dd64fb85/1d352ce1. This phase
contains no checkpoint, original model, fitting to real hidden states, random
inputs, mass computation or historical sensitivity run.

core.py fixes216 legal state slots within243 original keys; stable input ranks
and strict no-tie input domain; exact state-to-target check; homogeneous3-feature
lookup and fixed18-feature case2 separable OLS with rcond1e-12, required ranks3/18
and explicit counts/singular values. Deficiency raises and cannot trigger data
resampling or an alternate fit. Predicted hidden values are not clipped.

controls.py has exactly seven durable groups: (1)864 deterministic strict inputs
cover all216 keys with three train/one holdout each; (2)known per-state3×16
coefficient recovery and216 heldouts; (3)known18×16 separable coefficients with
216 case2 train/72 holdouts; (4)seven domain/deficiency rejections; (5)fixed MSE,
negativeR2, zero-variance status and first-index output ties; (6)nine explicitly
disjoint K×case×global-pick cells; (7)six dyadic inputs through one synthetic2-neuron
recurrence, checked against scalar exact Fractions. All numerical coefficient
and prediction residual thresholds are fixed absolute1e-10, not an outward
certificate. No source-selected best-fit variant.

Count every least-squares call including intentional rank failures; no source
FLOP/native price is inferred. The initial constant220 is provisional until
independent source review; actual driver topology must be reconciled before
execution. Matrix-rank and model-free input-generation work is not a model
forward. Future original-model work needs a separate checkpoint pilot and scope.

Exactly one process after source/protocol hash pins and independent source gate
are shared. Review flock,20CPU,2GB, one numerical thread. Exclusive result and
progress files; exceptions preserve all completed groups and tracebacks. No
retry/threshold relaxation or partial replay without a new reviewed recovery.
Retain command stdout/stderr even on failure. Saved independent review required
before model/pilot expansion. No execution has been authorized by this protocol
alone; controls_gate.json will bind the actual independent clearance.
