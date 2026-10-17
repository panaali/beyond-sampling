# T11591 original Duffy source-constant bound

## Question
Can exact ideal-node/weight enclosures quantify the contribution of source constants to the fixed original T5 positive rule?

## Method
Claim ef3bdfa4, theorem gate T55 and overlap check34. Use the original4D Duffy integrand only; fixed global derivative bound and tensor-weight telescoping, with exact rational contraction. Original11 stored-ray fixtures and orders6/10/16, no quadrature replay. See PROOF.md.

## Status
Bounded phase complete;16 controls independently reviewed95bacea8. All33 target contractions preserved d2cf76bf; independent saved-result review pending.

## Results
[measured] All33 contractions bounded,21 below1e-10, no unknowns. Seven fixtures at all3 orders are useful; wide10/22/35/48 remain loose at every order, ranging245 to1.66e71. Those values are conservative upper bounds, not actual errors. [inferred] The global q-min derivative bound is inadequate for these nearly antipodal original charts; no failure of the entire source-constant method follows. No mass evaluations were made. Reproduce saved summary with `python summarize.py`; original once-only bounded run was `ulimit -v 2500000; ulimit -t 20; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python run.py` (refuses existing outputs).

## Compute accounting
One thread2GB review20CPU. Target CPU0.012001152;16-control timer reported0.0 at its resolution, not zero work.96 source constant mapping operations and exact Gram/setup/bound categories are saved separately. Inherited root/certificate construction remains in original artifacts. Exact setup/categories/operand widths retained, isqrt internals and GCD/runtime unpriced. No tensor, mass, geometry, model or MC evaluations. Source Duffy/kernel/determinant floating errors remain excluded.

## Claims
A component error bound, never a full source-reference or speed certificate.

## Alternatives not ruled out
Global q-min and derivative estimates may be too loose; local interval or analytic-inner bounds remain possible. Original-rule evidence does not automatically transfer to34's altered3D exact-inner rule.

## Failure modes
Using half-L1 as delta, omitting s^4, assuming unit raynorm, rounding reciprocal/sqrt inward, applying exact ideal moment identities to source weights, or charging exact arithmetic as cheap float units.
