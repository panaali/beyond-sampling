# T260 source and representation inventory

Frozen owner head b9a39a163911b3e9f0864aec03db380c2eaaa4e7, reached after the T260 claim. Its relevant encode/C/C2/diagnostic sources are unchanged from T258's a9b09a14 according to the parallel source owner; this package binds the reached bytes directly. Fifteen text payloads are frozen with Git blob, SHA256 and length. AUTHOR_TREE.txt lists the reached owner subtree without opening binary payloads. No result JSON, learned coefficient array, model or prediction was decoded here.

## Deployed data that the formula actually requires

`encode.CellLinear.fit` learns a coefficient array of shape(C,d,q), a pooled fallback of shape(d,q), and a sorted vector of observed cell IDs. It also computes cell counts. `predict` uses IDs, coefficients and pooled fallback; training counts, ridge hyperparameters and training normal equations are not required for that unchanged predictor. Selected family identity and output-column correspondence ARE required when only chosen neurons/steps are retained.

Cell IDs are arithmetic encodings of input order-statistic ages and, in some families, a clipped positive-count K. They are not learned arbitrary neural-gate region IDs. The feature/routing program must be declared: prefix sorting, absent-rank conventions, current input, top values, positive/negative remaining sums, age clocks and integer encoding. Numerical evaluation also includes searchsorted, unseen-cell fallback, readout and intermediate casts. Counts of table cells or scalar coefficients do not price those operations.

The base dimensions follow the source schema: m3 has4 columns, m3clock10, m5 has6, m5rest8, all11. Hinge families append four comparator outputs for each available previous lag, so d=8+4*min(lags,t-1); the largest declared t10 design is44. This is a feature-dimension derivation, NOT the number of independent identifiable coefficients, actual trained cells or bits. Redundant columns at early steps or under a cell do not by themselves eliminate stored coefficients without a new exact/controlled representation.

`HingeBank` fits one A123K comparator model for each step. These learned tables are part of the representation, not free interpretable inputs. Each model outputs four comparator preactivations, then applies ReLU. PhaseC2 transforms these outputs to float32 before using them as features, and transforms each fitted formula output to float32 before storing Fev. A decoder matching the numerical predictor must retain these casts; coefficient precision is a different property.

## What needs counting for different outputs

1. **One final-hidden/readout predictor:** retain only chosen t10 formula columns plus their transitive hinge dependencies. If no selected final family uses hinges, the bank is unnecessary for this semantic target. With lags L, required hinge steps are10-l for the used lags; the t10 bank output itself is not a previous-lag feature. Charge each shared required bank table once.
2. **Every neuron at every step:** retain all selected (t,j) formula columns and their dependency union. This is a larger reconstruction target and cannot be compared directly to H-A1's final-hidden labels.
3. **Partial recurrent replacement:** the unmodified recurrence and its weights remain dependencies for unreplaced units. They cannot be silently free in a standalone predictor description.
4. **Direct-logit predictor:** q=10 outputs, its own fitted coefficients and hinge bank; no original readout is needed after the direct prediction. It is a different output abstraction, so hidden-state reconstruction metrics are not applicable.

For all16-neuron replacement, the source overwrites every coordinate of h_t with the input-derived formula at each step. Algebraically, the final result therefore depends only on the final-step replacement and the readout, not earlier replaced states. A future minimal dependency decoder may remove that dead computation. This is NOT an assertion that the existing source executed less work, nor a bitwise parity certificate after compiler/evaluation-order changes. Required casts/readout shape must be tested before numerical use.

If selected columns J_(t,f) share a family/step table, its stored floating coefficient count is (C_(t,f)+1)*d_(t,f)*|J_(t,f)|, including the pooled fallback; IDs and selectors are additional information. The bank contributes corresponding shared four-output tables. This formula is conditional on actual retained table dimensions. It does not justify substituting the README's approximate maximum cell count for every table or multiplying it by all160 neuron-step slots.

## Retained artifact limitation

PhaseC2 explicitly deletes each fitted CellLinear after accumulating formula predictions and selected-family scores. It writes aggregate JSON and a separate evaluation archive containing x, ref and p_all. It does not serialize mdl.ids, mdl.coef, mdl.pooled or HingeBank.models. The direct-logit diagnostic similarly writes metrics rather than its fitted predictors. The reached tracked tree has no deployable cell-table archive. External/untracked artifacts may exist; their absence is not proven globally. Source-generated expected archive fields are not independently decoded archive contents.

Consequently exact deployed parameter counts, coefficient bitstrings and pruning/merging effects are NOT available from these source/report files alone. Training seed plus fit source is not a stand-in for actual historical floating coefficients. A generative description that reruns training would require its teacher/data/randomness/runtime and decoding computation to be declared, and would not be comparable to the explicit-payload code proposed here.

## H-A1 and the earlier declared code

H-A1's labels.py returns final hidden-state predictions from x. `score.py` applies the original readout, reports task accuracy, and uses float32 model inputs/reference calculations in its historical campaign. This is not the same quantity as T241's reported network agreement. A future common comparison must evaluate both under one frozen numeric/model/input contract and keep historical values separate.

T25's reviewed prefix grammar counts executable helpers, chosen operators/positions and coefficient bits, conditioned on a fixed Python AST schema/NumPy interpreter. It frames the complete predictor as (AST, argument_payload). Its original M16 formula audit counted78 fitted scalar literals and a declared36182-bit program, not a universal five-times-sixteen-bit bound. These are inherited reviewed numbers, not recalculated here. Library, identifier and formatting conventions affect the code; no shortest-description or intrinsic legibility conclusion follows.

The H-A1 limit was at most five fitted scalars per neuron, NOT five total model-dependent choices or a generic description-length budget. A comparison must charge routing/feature choices on both sides, choose the same conditioning and output target, and separate source/model search cost from deployed description and execution cost.
