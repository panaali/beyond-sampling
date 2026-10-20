# Independent T463 review

Verdict: the six thin-a counterexamples refute the source's general floating mass-upper claim. The local replacement mass and exact-CROWN constructions pass this bounded mathematical/implementation review under their stated real-network and interval-arithmetic contracts. This does **not** adopt them into T455, certify ordinary CROWN, certify a whole search, or establish that the producer encountered a failing box.

Reviewed final `review/late_bound_soundness/{README.md,local_verifier.py,results.json,PROTOCOL.md,REFINEMENT.md,run_controls.py}` plus original `rmass.py` at `4e29c33861a0983535cd9bc80d660cfd9f2dd412`. Local-verifier SHA256: `7ccf219a00e45c870f1cc35690d492b94be428433f0b233da27e5651f1fc05c6`. The preserved first pass and explicitly disclosed witness refinement are distinguishable; an upper-expression discrepancy was not silently promoted into a true-mass counterexample.

## Independent mathematical checks

1. **Change of variables.** Use linear coordinates `(g=x_b-x_c, x_c, x_a-x_b, x_c-x_l1,...,x_c-x_l7)`, a signed unit-determinant transformation. Substitution `(g,gr,ga,gl1,...,gl7)` contributes nine powers of g, so the Jacobian is g^9. Integrating a and the seven lows against their Gaussian factors cancels eight powers, giving the stated g/(2pi) kernel and eight CDF differences.
2. **Local mass upper.** Each transformed argument is bilinear in positive g and r plus a fixed endpoint; extrema occur at rectangle corners. The CDF-range bound and interval-width-times-maximum-density bound are separately valid upper bounds, so their minimum is valid. The Gaussian r factor has exact minimum q=2(r+1/2)^2+1/2 on a rectangle. Multiplying nonnegative factor bounds and covered rectangle areas is valid. The universal tail follows from ||y||^2>=r^2+(r+1)^2>=1/2; substituting w=g^2/4 gives integral `512*Gamma(5,100)=512*24*exp(-100)*sum(100^k/k!,k=0..4)`. This matches the implementation's16*32*24 constant. Exact rational endpoints/grids avoid the producer's erased-width defect. Arb enclosures and upward final float conversion are required; this proof does not transfer to the producer's ordinary floating expression.
3. **True lower witnesses.** The refined subregion lies inside each original thin-a box. Bounding g below, the squared Gaussian exponent above and every conditional density below gives a genuine positive mass lower bound. Root's separate rational witness also checks: coordinate maxima `.5,.75,.75+a_hi,7 copies of .625` bound all squared coordinates, the parameter volume is `(1/4)*(a_hi-a_lo)*(1/8)^7*(1/8)`, and `e<3`, `pi<22/7` give the conservative rational density lower bound used there. Thus the six reported failures concern actual mass, not merely looseness or differing grids in two upper estimates.
4. **Exact CROWN.** Forward interval arithmetic encloses every preactivation. In backward propagation, positive coefficients use a valid ReLU lower affine function (zero or identity); negative coefficients reverse the upper secant into a lower bound. Stable units, affine recurrence/input propagation and final box minimisation are exact. Original output-row subtraction in Fractions is appropriate. The remaining initial-state term is zero because h[-1]=0. Any strictly positive verified competitor-minus-target margin excludes that target from the event. This proves the local real-arithmetic implication, not equivalence to machine float inference or an externally rounded coordinate map.

## Independent numerical checks

[review_t463.py](review_t463.py), seed462463, checks32 new synthetic 1D, width2, three-step networks with three logits. It enumerates all64 activation patterns and intersects their exact rational linear inequalities; this differs from the author's breakpoint-splitting oracle and uses different networks/seed. Every returned lower bound is <= the exact region minimum; every positive prune is valid. Results in [T463_REVIEW.json](T463_REVIEW.json); numerical CPU .109s.

[review_t463_mass.py](review_t463_mass.py) independently reruns the pinned producer on an **off-suite k=57** box. It uses a different included subregion (`r in [-3/8,-1/4]`, `g in [5/2,11/4]`, lows in `[1/8,1/4]`) and a rational density/volume witness, with no Arb calls. Float charge7.02630423937274e-55 is below true mass lower9.33807118044858e-34 by at least1.329e21. Results in [T463_MASS_REVIEW.json](T463_MASS_REVIEW.json). The numerical body used .003CPU-s plus Torch import overhead. This reproduces the mechanism on a new input rather than relying only on saved original failures.

Reproduce from repository root:
```
ulimit -v 2500000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python review/is_v4_contract/review_t463.py
git show 4e29c338:m16/approaches/late_triple_bound/rmass.py > /workspace/work/is-contract/t455_rmass.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python review/is_v4_contract/review_t463_mass.py /workspace/work/is-contract/t455_rmass.py
```

## Scope and open obligations

No trained checkpoints, neural forwards, actual frontier reconstruction, or global search were used. The repaired primitives need integration with verified original weights/coordinate semantics, exact box coverage and bookkeeping, certified root-domain tails and outward-rounded accumulation before a global certificate can follow. Arb's enclosure contract is assumed rather than formally verified here; exact-rational CROWN scalability is untested. Strict/lenient estimator F bars do not apply to these soundness controls. No blocking mathematical defect was found in the scoped local replacements.
