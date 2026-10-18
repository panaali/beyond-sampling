# T278 fixed numerical result — independent application audit passed

One run completed all2,718 face points,5,436 exact partitions and5,439 events. Internal CPU26.677619646s, maxRSS113,600KiB; external28.46wall/18.55user/8.34sys. Point records5,958,656bytes. The18-centre scheduling projection was41.401686173CPU (<75), so the frozen continuation proceeded. No failure, retry, node/order change or new reference/RNG occurred. Immutable gate309f44b5 was successfully shared48eabfc2 before a separate launch.

The inherited H4T4 reference interval is [.4770130989463459,.4770131530418916]. All values below are finite single-model diagnostics; independent saved audit64e4bea9/sharedab0cd608 passes. Composed kernel CPU excludes observer/catalogue/aggregation overhead and is not full runtime or FPE.

| Fixed rule | Method | Accuracy | Composed CPU | Absolute-error interval (approximately) | CPU × squared-error upper |
|---|---|---:|---:|---:|---:|
| GL3 | frozen/zero | .491871107849 | 1.08875 | .014857955–.014858009 | 2.40352017e-4 |
| GL3 | first | .492956314108 | 1.56838 | .015943161–.015943215 | 3.98660377e-4 |
| GL3 | second | .492849804612 | 1.56872 | .015836652–.015836706 | 3.93437085e-4 |
| GL3 | direct correction | .493355873184 | 1.56876 | .016342720–.016342774 | 4.18993968e-4 |
| GL3 | actual only | .493355873184 | 1.26478 | .016342720–.016342774 | 3.37805746e-4 |
| GL5 | frozen/zero | .479316129345 | 2.82955 | .002302976–.002303030 | 1.50078088e-5 |
| GL5 | first | .480140519712 | 4.90133 | .003127367–.003127421 | 4.79387561e-5 |
| GL5 | second | .480047470973 | 4.90056 | .003034318–.003034372 | 4.51214943e-5 |
| GL5 | direct correction | .480569448034 | 4.90116 | .003556295–.003556349 | 6.19879909e-5 |
| GL5 | actual only | .480569448034 | 3.60869 | .003556295–.003556349 | 4.56412957e-5 |

No corrected method meets the fixed halving diagnostic on either rule. Direct residual and actual integration agree in aggregate. The global inner first/second remainder totals are .00712513/.00190288 (GL3) and .00638343/.00175092 (GL5), not outer-error certificates.

The full window-mass diagnostic is1.0076550115331184 (GL3) and1.0036668170042893 (GL5). Thus these coarse outer rules do not even normalize the whole face decomposition closely. The frozen estimate's smaller final bias can reflect cancellation with outer/table approximation; the result does not show that the uncorrected classifier is more faithful. It also does not isolate ideal Taylor accuracy from outer integration. Full band discovery makes corrected kernel cost larger, as explicitly charged.

## Negative scope and alternatives

This is a finite implementation result, independently checked by78 within the stated saved scope, not a broad C1 impossibility or ARC pass/failure. Tried: exactlyGL3/5, oneH4T4 seed0, face-frozen zero/first/second/direct and actual-only integration. Not tried: higher rules or a different outer transformation (resource/fixed-scope limit), cheaper approximate discovery (different algorithm), other models/noise (separate scope), conditional residual-law construction (not implemented here), or rigorous outer/table enclosures. These remain open, and no outcome-based extra rule is run.

The archive preserves every point's exact intervals and signed residuals,64 predetermined full affine-cell records, the source and runtime pins, resource projection and progress. Independent review contracted all saved rows and reconstructed those64 recurrences (5,808 feasible-mask attempts), using1.00345CPU; direct Gaussian masses/table/reference and operation-bit internals retain their stated qualifications.
