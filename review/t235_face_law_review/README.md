# T235 independent face-family law review

## Question
Does the proposed input-only family `(kind, ub)` admit model-independent probabilities and conditional generation under its stated laws?

## Method
Read-only review delegated by TASKS claim `112cee94`. Reviewed author proposal `89363df1`, published `2b3e4e12`, against four original Git payloads at `60967f893fb649ca9933bd026ca76c3c19402e15` and the independently reviewed T230 derivation. `source_gate.json` records all ten direct file pins and five payload identities. No author module was imported.

## Status
Source and theorem review passed. This is a documentation-only result and does not authorize an implementation or numerical execution.

## Results
The ideal densities are strictly positive at finite height and bounded by the integrable density `exp(-u²)/sqrt(pi)` for integer T >= 3. Consequently their normalizers and four bin probabilities are positive. Conditioning the height through a restricted inverse-CDF coordinate, while retaining the independent configuration and other latent coordinates, gives the stated conditional face law. Uniform selection of kind gives joint probability `p[k,j]/2`; the co-area multiplier `Z[k]` is a weight, not a selection probability. The original co-area decomposition itself remains an inherited assumption.

The frozen-grid theorem is also correct. Each interval with increasing CDF values maps a uniform coordinate affinely onto its grid interval. A flat CDF interval carries no latent probability and creates a gap, not an atom. Thus the CDF is the stated piecewise-linear interpolation in the grid coordinate, including flat sections. Its bin masses are rational when the frozen inputs are rational. Zero-mass bins have no conditional generator. Values assigned at finitely many latent endpoints do not affect this continuous-law statement.

The source uses `(c-c[0])/(c[-1]-c[0])` for inverse interpolation but `c[-1]` for its co-area normalizer. This distinction is correctly retained. Floating interpolation, finite random numbers or Sobol coordinates, clipped Gaussian quantiles and the fibre-window cap do not inherit exact ideal-law probabilities from the theorem. In particular, a finite node at the conditional lower endpoint can violate an open-left bin convention. No endpoint repair is silently authorized.

The proposed interpolation-error bound also has the right conditional form: if an independently proved uniform error is at most eta, a changed classification at boundary b requires the real interpolant to lie within eta of b. Its continuous-latent mass is bounded by `G(b+eta)-G(b-eta)`. The proposal supplies no eta and correctly leaves CDF construction, finite latent law and conditional-map rounding errors separate.

Source inspection confirms `ub` depends only on tied height. Computing it before the walk extends the label to every face, including zero-integrand faces omitted by the old tail-label archive. It does not reconstruct missing historical labels or preserve the predictive information of outcome-derived winner/flip fields.

There are no numerical result commands: no probability, grid, face or performance computation was performed. Provenance checking consists only of file SHA256/size comparison, `git show <commit>:<path>` byte equality, Git blob identity, and the completed T230 review's direct derivation pin.

## Compute accounting
Text reads, hashes and Git-object comparisons only: five payloads checked, four against original Git bytes, and the fifth against the completed independent T230 derivation pin. Zero author imports, numerical integrals, grid construction, array decoding, face generation, model calls, RNG draws or mass evaluations. Human review and metadata work are not an estimator cost measurement.

## Claims
- [inferred] The ideal-law and frozen-grid real-arithmetic conditioning theorems hold under the stated assumptions.
- [inferred, source] The cheaper input-only label avoids the full walk required by winner/flip labels.
- [speculative] Such strata could improve an estimator; no variance, accuracy, deductive eligibility or cost gain has been shown.

## Alternatives not ruled out
Coarsening may remove almost all useful information. Setup or conditional integration may dominate. A useful ideal partition may perform poorly under finite quadrature. A future bounded phase could test a specified implemented law, but requires its own scope, endpoint policy and independent gate.

## Failure modes
Invalid/nonmonotone grids, zero normalization or strata, unsupported T, finite endpoint atoms and inaccurate quantile/walker operations prevent transferring these ideal conclusions. No runtime grid validity or floating-program probability certificate is asserted. Fixed-rule empirical proportions are not population probabilities or iid uncertainty estimates.
