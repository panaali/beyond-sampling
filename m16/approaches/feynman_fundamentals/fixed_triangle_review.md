# T104 fixed tangent: independent review

The fixed-frame formula is valid in exact arithmetic. For endpoints
`E_i=P+r*v_i`, both tangent differences scale by r, so the orthonormal
tangent basis can be prepared from the three fixed rays. The perpendicular
offset is common to all endpoints and the determinant is
`delta*r^2*lu*lv`. This changes frame construction, not the spherical
moment identity or the positive original-domain fallback.

Two alternatives were checked before accepting the finite implementation:
an incorrect Jacobian would survive increasing independent positive-rule
order; cancellation in a thin triangle could instead depend on batch
shape or trigger a fallback. `check_fixed_triangle_review.py` tests the
owner's fixed stored cone 2/order 8/node 447 and three new shifted,
coalescing triangles. Its reference directly integrates the original
two-dimensional Duffy domain with positive GL64 and GL128, without the
spherical formula. The rules agree within 4e-16 relative. New-formula
relative discrepancies are 1.54e-11 and 9.77e-11 in the two analytic
cases; the smaller/coincident cases use the unchanged fallback and agree
within 3.4e-16. CPU was 0.0323 seconds. This is a finite numerical check,
not a roundoff enclosure or full T6 integration.

Run with one BLAS thread under the shared review lock:

```
/workspace/clean-neurons/.venv/bin/python m16/approaches/feynman_fundamentals/check_fixed_triangle_review.py
```

The sensitive stored node gives 2.222033652230075 as a single-row call,
whereas the owner's original batch gives 2.2220336522738324. The positive
reference is 2.2220336522643143. Therefore algebraic equivalence does not
imply batch-independent floating results. The owner's separately frozen
rowwise projection rescue makes scalar and batch calls identical but
moves this value to 2.222033652317682, farther from the reference. That
rescue establishes consistency, not uniform accuracy improvement. The
original remains the default. The heuristic cancellation screen remains
a qualification in either variant.

The owner's 64-cone pilot retains 167352 triangles, 3641 fallbacks and
58256 positive nodes. Its partial masses agree to one final binary64 ulp;
reported numerical CPU changes from 0.49909 to 0.43794 seconds. This is
a bounded performance result, not a full-run or portable-cost claim.

For a separately frozen p3-only follow-up, the proposed midpoint identities
`A^2=(u0+v0)^2/[2(1+c)]`,
`B^2=(v0-u0)^2*(1+c)/(2*s^2)` and
`I2=A^2*(theta+s)/2+B^2*(theta-s)/2` are correct for unit endpoints,
`c=cos(theta)`, `s=sin(theta)` and 0<theta<pi. Coincident edges require
their limiting branch; small theta still needs stable evaluation of
theta-s. Normalizing the computed (s,c) is a distinct floating variant,
not an exactness certificate.
