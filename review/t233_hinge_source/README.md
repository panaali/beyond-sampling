# T233 independent written theorem/domain review

## Question and status
PASS for frozen558eec16, under explicit delegationc1343d8a. This is a proof-only review, not a source/execution clearance for numerical controls or a target experiment. theorem_review.json binds the two original documents. No author module, numerical library, target array, model, mass or random generator was called.

## Independent derivation
For a<0, r=b+−b=(−b)+. For a>=0, r=b+. This proves nonnegativity and all zero/boundary ownership directly. The surrogate may be negative and is only a signed intermediate; adding r exactly restores the second activation.

For alpha beta<0, the affine forms have the same strict sign precisely between their roots u and v. Conditional on any full past state h and an independent fresh X~N(0,1), residual r=|beta||X−v| on that interval and zero elsewhere, up to null endpoints. If u=v, both forms vanish at the shared root and r is identically zero. Thus positive-residual probability is |Phi(u)−Phi(v)|, without a Gaussian assumption on the hidden state.

The first moment divided by |beta| is the oriented integral from v to u of (x−v)phi(x). An antiderivative is −phi(x)−vPhi(x), giving exactly the displayed density/CDF expression. Its u derivative is (u−v)phi(u), and its value at u=v is zero; positivity holds on both sides. This proof does not bless the cancellation-prone difference as a numerical recipe.

Distance is at most delta, giving delta times interval probability. A uniform density cap bounds the triangular distance integral by delta²/(2sqrt(2pi)). Density continuity gives the fixed-v small-delta limit. For bias-free offsets, delta²=(ell·h)², so tower expectation gives ell^T E[HH^T]ell, using the raw second moment rather than covariance. Its availability and computation are not free.

A single-step state replacement changes only coordinate j by r. The update map is componentwise1-Lipschitz after linear Wh on all real states, including a signed surrogate. Induction gives |Wh|^s e_j r; the final absolute readout contrast gives the margin bound. This is pointwise for every identical future input sequence. Small expected margin error does not imply small classification error without additional margin/tie reasoning.

## Limits and alternatives
Zero/same-sign slopes, rank-conditioned fresh inputs and other input laws need separate formulas; they are not treated as zero residual or failures. Cosine similarity alone does not control threshold separation. No target pair structure, rare event, region-count saving, cheap estimator or speed advantage is established. Approximate grouping, alternative anchors, multiple hinges, exact conditional integration and state-domain bounds remain open. Any implementation, controls or target use needs a fresh published protocol and separate review, respecting DRI priority.

## Compute accounting
Document byte comparison/hashing and written algebra only. No numerical experiment or operation-count claim; no reproducibility command beyond the pinned theorem files.
