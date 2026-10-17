# T11591 original Duffy source-constant bound

## Question
Can exact ideal-node/weight enclosures quantify the contribution of source constants to the fixed original T5 positive rule?

## Method
Claim ef3bdfa4, theorem gate T55 and overlap check34. Use the original4D Duffy integrand only; fixed global derivative bound and tensor-weight telescoping, with exact rational contraction. Original11 stored-ray fixtures and orders6/10/16, no quadrature replay. See PROOF.md.

## Status
Source and16 controls frozen for independent gate; no numerical execution yet.

## Results
No measured results. [inferred] The bound controls ideal real-arithmetic sums with source versus exact constants. [speculative] It may be useful only on well-conditioned cones.

## Compute accounting
One thread2GB review20CPU. Exact setup/categories/operand widths retained, isqrt internals and GCD/runtime unpriced. No tensor, mass, geometry, model or MC evaluations. Source Duffy/kernel/determinant floating errors remain excluded.

## Claims
A component error bound, never a full source-reference or speed certificate.

## Alternatives not ruled out
Global q-min and derivative estimates may be too loose; local interval or analytic-inner bounds remain possible. Original-rule evidence does not automatically transfer to34's altered3D exact-inner rule.

## Failure modes
Using half-L1 as delta, omitting s^4, assuming unit raynorm, rounding reciprocal/sqrt inward, applying exact ideal moment identities to source weights, or charging exact arithmetic as cheap float units.
