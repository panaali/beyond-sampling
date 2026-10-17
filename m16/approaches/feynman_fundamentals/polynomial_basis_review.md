# Independent matched angular-basis review

T55 review of T56, 2026-10-17. Owner source in `angular_polynomial_cycle` was read only. This is a finite basis comparison, not a general rejection of nonperiodic bases.

On the root lune, t=2(phi-phi0)/WIDTH-1 has conditional measure dt/2. Thus sqrt(2n+1) P_n(t) is normalized and the constant remains coefficient zero. Angular Gram products through degree eight have degree at most sixteen; nine positive Gauss-Legendre nodes integrate each clipped interval exactly in real arithmetic. The implementation includes the required half interval width divided by WIDTH. Arc orientation and three wraps cover the supported window. Matrix-free application inherits the existing radial quadrature and shared geometry. Costs include the new angular basis construction and nine-cubed Gram product events, so a lower measured CPU does not imply fewer ordinary arithmetic operations. Opaque setup/special functions remain excluded from the portable count.

## Different primitive, same fixed H2 experiment

`check_polynomial_basis_review.py` replaces positive angular cubature with Legendre-product antiderivatives. It retains the same model, gate branches and radial Q64. Full-window identity and half-window complement errors are 3.55e-15 and 2.22e-16. The resulting estimate is 0.4083876713432273, only 1.11e-16 from the owner's result. Recorded CPU is 1.254 seconds, including cold setup; no estimator cost claim is made for this validation.

The independent reference is 0.41017126172687207 and the prior Fourier estimate is 0.4090408211303979. Therefore the fixed polynomial basis has about 1.578 times the absolute error on this H2 model. The negative survives a different angular primitive. Owner radial Q64/Q128 agreement further separates this result from the tested radial integration error, but neither check proves exact global Galerkin accuracy.

The matched H4 case improves, so the frozen requirement that both pilot cases improve is not met. No seventeen-case expansion is justified by that protocol. Other polynomial degrees, mixed bases, internal gate support and changed allocations remain open. The finite endpoint-coupling hypothesis was a plausible explanation, not a theorem that polynomial replacement must improve every model.
