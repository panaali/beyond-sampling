# T259: fixed-design quadratic loss discriminator

This is exact-arithmetic algebra, not a fitted result. K>=2 classes; all matrices below are finite real matrices. No empirical 99% ceiling is asserted.

## 1. Output-linear fitting commutes with a fixed readout

Fix training features X (n by d), query features Z, and an output-independent symmetric positive-semidefinite feature penalty Lambda. Suppose A=X^T X+Lambda is invertible. For any training target matrix Y, the common ridge solution is B(Y)=A^{-1}X^T Y. For any fixed right transform R,

    B(YR)=B(Y)R, and Z B(YR)=(Z B(Y))R.

This follows by associativity and distributivity, without requiring R invertible or square. It applies separately to every discrete cell and to a pooled fallback. Query routing must be fixed independently of the target columns. Thus it also applies to the combined prediction assembled from those cells.

An intercept is allowed as a shared feature column, with the same penalty mask in both fits. Output-specific penalties, target-specific routing, output-specific feature choices, or different training weights invalidate this argument. In a singular problem a fixed common Moore-Penrose map or other specified linear solution map suffices; arbitrary minimizer choices do not. A training-null feature direction can affect held-out predictions.

A fitted feature bank may depend on training targets. The claim conditions on that *same frozen bank* for both fits. It does not claim linearity of a pipeline that refits the bank when changing targets. Floating reductions, target formation, solves and casts need not commute bitwise; no numerical tolerance is deduced here.

## 2. Unweighted all-pair squared margin loss is centered logit loss

Let 1 be the K-vector of ones and C=I-11^T/K. For any row error e,

    sum_{a<b}(e_a-e_b)^2 = K sum_a e_a^2 - (sum_a e_a)^2
                         = K ||e C||^2.

The first equality follows by collecting each square K-1 times and each cross-product twice. Consequently the factors alpha multiplying ||e C||^2 are:

| Pair convention | alpha |
|---|---:|
| Unordered pairs, sum | K |
| Ordered distinct pairs, sum | 2K |
| Unordered pairs, mean | 2/(K-1) |
| Ordered distinct pairs, mean | 2/(K-1) |

These constants do not incorporate an additional sample mean or a class mean on the logit comparator. A sample mean divides the data term by n. Every such normalization must be recorded before matching penalties.

Consider ordinary logit ridge

    J(B)=||XB-Y||_F^2 + tr(B^T Lambda B).

Its unique solution under the invertibility assumption is B0=A^{-1}X^T Y. Write B=D+b1^T, D=BC, D1=0. Orthogonality splits both the residual norm and the shared quadratic penalty into centered and common-mode terms. Therefore the centered part of B0 is D0=B0 C.

The matched all-pair objective is

    J_pair(B)=alpha ||(XB-Y)C||_F^2 + alpha tr(B^T Lambda B).

Its centered minimizer is exactly D0. Its common mode minimizes a separate penalty and may be nonunique if Lambda has a nullspace; it cannot change a class difference at any query. Equivalently one can penalize only D and fix b=0, or choose any common-mode gauge independent of D. Under the same centered solution-map convention the predicted pairwise margins equal those of ordinary logit ridge. Adding a common scalar to all class scores preserves argmax and the full tie set, including deterministic first-index tie ownership.

Hence changing *only* a common-design quadratic fit to unweighted all-pair squared margin loss, with matched normalization/regularization and solution convention, cannot improve any prediction label in exact arithmetic. This statement does not promise equality of common-mode scores or finite-program outputs.

If the numerical penalty is held fixed while the data term is multiplied by alpha, the effective penalty becomes Lambda/alpha. Any resulting change may be an ordinary regularization change. The same warning applies to sum versus mean over samples or classes.

## 3. Boundaries and written separating cases

1. Common-mode target perturbation Y -> Y+u1^T changes fitted scores only by a common scalar per query; margins remain fixed.
2. Arbitrary fixed readout R, including rank-deficient R, satisfies section 1; hidden targets need not have as many columns as logits.
3. Fitting hidden preactivations then applying ReLU is nonlinear and is outside section 1. Fitting post-ReLU targets but subsequently clipping the predictions also breaks the identity.
4. A common intercept with the same penalty is included. A hidden fit with an unpenalized intercept and a logit fit without that feature is not included.
5. Distinct per-output regularizers generally fail commutation: for one feature with X=1, targets (1,1), penalties (0,1), fitted outputs are (1,1/2); fitting their sum with zero penalty gives 2, not 3/2.
6. Unspecified singular solutions are insufficient: X=(1,0), query Z=(0,1), zero penalty. Training loss fixes the first coefficient but leaves the query prediction arbitrary.
7. Per-output selected feature families need not share one left-linear map. Their predictions cannot be replaced by a common-design logit fit using this theorem.
8. Weighted class-pair, top-rival or input-dependent margin losses need not reduce to the scalar alpha C metric. Nonquadratic losses, nonnegative hidden constraints, recurrent rollout losses and new features remain distinct candidates. Some special weighted designs may still be redundant; no general distinctness guarantee is claimed.
9. Even when the feature map is fixed, this proves redundancy of the specified loss substitution, not insufficiency of the features for 99% agreement or any universal error floor.

## 4. Source specialization

At pinned a9b09a14, phaseC_diag.py fits both htr and htr @ Wo.T using the same CellLinear defaults, same training/query designs and frozen hinge arrays, and compares the hidden fit after multiplication by Wo.T without another ReLU. Subject to successful solves, the exact-real idealization has identical logits by section 1. This is not an empirical bit-parity finding. In contrast phaseC2.py selects families separately per neuron/time and applies ReLU after replacement; its comparison is not covered by that identity.
