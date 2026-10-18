# T273 target recipe decision

The requested target calibration is not yet an executable reproduction. The pinned E32 `est_A` defines a specific H4T3 construction: four near-antisymmetric second-step wedges, 14 likely and four unlikely rank subregions, conditional final-gate probabilities, and a shared latent Bernoulli(q=.2) matching each gate's first two moments. It computes 3-dimensional cone volumes. It neither finds a modal gate word nor defines a generic H16T10 prefix partition.

Directly passing target weights cannot preserve that algorithm: `e=eye(3)`, four explicit H2 rows, four gate marginals, three readout indices and length-three normals remain fixed. This is a source-domain mismatch, not an executed target failure or proof that an extension cannot work. No scientific calls were made.

`rank_exact` is a different available target estimator. It uses 720 ordered position triples with one fixed permutation of remaining positions,11 sign-count classes, canonical inputs from2,000,000 fixed-seed Gaussian draws at import, and a frozen gate word at the canonical point. Its cone probabilities use scrambled Sobol sorted half-normals. A canonical gate word is not an established mode; these probabilities are not exact cone volumes. Importing that module would already perform unapproved RNG/setup work. Its existing `DEDUCTIVE=True` label is historical and does not override current classification rules.

The following choices must be fixed before a target calibration has an unambiguous scientific interpretation:

| Component | Literal E32 A | Target prerequisite |
|---|---|---|
| Prefix surrogate | Four structural H2 wedges | Explicit H16T10 rank/position partition and prefix affine-map construction |
| Final gates |16? No: exactly four marginal probabilities, shared q=.2 | Sixteen marginals with defined conditional law; keep latent q=.2 or explicitly designate a new approximation |
| Probability computation | Exact 3D solid-angle formula | A specified 10D cone-probability algorithm, precision/error status and complete cost; fixed quadrature is not exact integration |
| Sparse rank coverage | Hand-derived14+4 regions | Defined weighting and treatment of omitted orderings, not one unlabelled arrangement per top-three triple |
| Budget | Historical ARC/producer counts | One stated64kFPE budget and actual cost between32k and64k; no arbitrary cost padding or underbudget pass |

Recommended next decision: label a new generic recipe **ARC-inspired**, preserving q=.2 and the final-gate independence construction, rather than calling existing `rank_exact` an ARC A reproduction. Require a deterministic finite class design, a specified prefix surrogate, and a controlled deterministic volume approximation with full construction cost. If the DRI instead intends rerating the existing rank_exact implementation as the calibration, say so explicitly and retain its inductive/quadrature/setup distinctions; it is not a literal reproduction.

The target experiment should then use one fixed recipe and64kFPE, all individual model costs/errors and group intervals. LB3 is development data; final claims follow the new HP1/cluster-CI directive without reading sealed truths. A cheap constant or a method exceeding its declared budget does not pass by its own-cost score. Neither a20–50x failure prediction nor the old E32 bar is a measured target outcome.

This decision note does not authorize a new model, RNG, cone, fit or panel call. It identifies a missing method definition so the next numerical experiment answers the intended question. C1 residual-band integration and T267 cancellation remain separate work; there is no inference against those routes.
