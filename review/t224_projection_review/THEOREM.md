# T224 independent projection reasoning

Assignment: author56's published T224 claim `cce45a1b`, reviewer folder `review/t224_projection_review/`. This is a source/theorem review prerequisite, not execution clearance. The author's frozen code/protocol has not yet been inspected.

Let y be any nonnegative real hidden vector and let z be any real predicted hidden vector. Define P(z)_i=max(z_i,0). Coordinates with z_i>=0 are unchanged. When z_i<0,

`(z_i-y_i)^2 - (0-y_i)^2 = z_i^2 - 2*z_i*y_i >= 0`.

Summing proves `||P(z)-y||² <= ||z-y||²` pointwise. No distribution, training, independence or network-weight assumption is needed. Equality holds precisely when there are no strictly negative coordinates of z. The theorem concerns finite real values; it does not grant semantics for NaN/infinity or a bound on floating reductions. It applies separately to every retained model/input, without selecting a neuron or data subset.

This theorem does **not** establish monotonic classification agreement. With readout row0=0 and row1=(-1,-1,1), take true hidden y=(0,0,1) and predicted z=(1,-2,0). Both true and original predicted margins are +1, while P(z)=(1,0,0) has margin -1. Hidden squared error falls from6 to2 while first-index class agreement changes from correct to incorrect; all these margins are strict.

Conversely, row0=0, row1=(1,1,-1), y=(1,0,0), z=(1,-2,0) gives true margin+1, predicted margin-1, and projected margin+1. Hidden squared error falls from4 to0 while agreement improves. These are exact algebraic witnesses, not executed controls or evidence about a trained model.

Ties also require preservation of the same first-index ownership: row0=0, row1=(1,-2), y=(1,0), z=(0,-1) gives true/predicted label1 but projected zero logits select label0. Hidden squared error falls from2 to1. A control that changes tie semantics would confound projection with a different classifier.

Network agreement and task accuracy are separate: a projection may move toward the network's hidden state while correcting or introducing a task mistake. The fixed original T201 reconstruction remains the primary result. A newly proposed projection is a separately labeled follow-up; its already-observed panels are not fresh validation of this choice. No fitted correction becomes a deductive ARC estimator by this projection theorem.

Reviewer accounting: symbolic reasoning only. No author imports, fits, model calls, random inputs or saved-panel application.
