# T20 counted Laplace hybrid port

Status: claimed/pushed9e70bd7 after supported network-auth retry; previous publication blocker resolved. Own folder only. Gaussian and uniform radial ports remain frozen.

Question: retain sci_laplace's fixed per-configuration shifted Sobol nodes and near-plus-tail model while replacing unsafe exact-near walking and historical recurrence-only charges. Preserve k0=.04, both original face laws, same fixed seed0 and blind rule. Corrected radial walker integrates the near region to completion or raises. The remaining tail keeps the original one-node-per-side rule and original model-free inverse radial table for node locations, while stable radial interval mass weights it. Each tail forward/readout is fully charged; no claim that one tail point is exact.

Track for every evaluated face both exact-near mass and full unresolved tail mass. Conditional on the near integral and fixed outer nodes, the true face mass lies between near and near+tail_total; tail-point error is at most tail_total. Aggregated tail uncertainty excludes outer quadrature/table/roundoff error and must never be called a total accuracy certificate.

Frozen alternatives/falsifiers:
1. Variable/target changes could corrupt near/tail partition. Independent constant-label positive radial integrals must match near+tail masses even with both target conventions; zero output also exposes legacy tie handling.
2. Changed fixed node generation could mask a walker difference. Preserve/pin original geometry/estimator sources and compare node, face and tail-coordinate fingerprints on public checkpoint512/2048 plus zero-output512. No new MC/noisy sweep. Any trained discrepancy beyond1e-10 gets diagnosis, not ignored.
3. Original tail inverse interpolation may produce outside-interval nodes or a cap may silently lose mass. Check interval containment; retain explicit near failure, and separate inherited legacy cap semantics from new completion. Report positive tail mass even when the chosen tail node predicts correctly.
4. Hidden setup/transform costs can dominate the promised budget. Charge all numerical categories below and expose overshoot; no historical score is reused.

Numerical operation proxy: cold tube setup500*(200001+20001); per-stratum Sobol/RNG setup100000*d+20000+64*d; generated coordinates30*d*m; stratum shifts64*nstrata*d and4*nstrata*m*d combination; per face geometry1000+230*9+80; radial preparation300+20*T; all radial/native counters; each tail weight350, original radial CDF pair500, inverse interpolation100, coordinate setup10*T; tail affine/ReLU/readout plus argmax counted in full; aggregation1000+10*n per family; blind transforms230*T+1000 per row plus full forward and target sort. Opaque special-function/RNG/Sobol costs are conventions, not certified portable bounds. No fitted importance arrays. CPU measured separately, imports/interpreter excluded from scalar proxies.

Run limits:<=110CPU primary lock;<=20CPU review checks on review lock; one thread/2GB; batches<=65536. All small controls and failures preserved. Root/DRI independent review remains required. Boundary and other tube variants remain unported.

## Frozen independent constant-label hybrid diagnosis
The trained paired-node checks pass within1.4e-13, while zero-readout differs materially from legacy as expected from its near-region tie bug. Before handoff, reconstruct each full radial and near radial constant-label face integral using independent70-digit positive quadrature on the same fixed faces. Their difference gives exact tail error for this constant-label case, so predict both new full-face estimate and old tail-only estimate; preserve the whole-rule discrepancy from true accuracy1/10. This diagnostic uses no recurrent model integration and does not score the trained model. The conditional tail interval must not be promoted to an outer-quadrature accuracy guarantee.
