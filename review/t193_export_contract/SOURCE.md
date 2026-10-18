# Frozen source trace and acceptance risks

These are static findings on a5ab8757, not results from a new exporter run.

## Geometry and selection

`export_simplices.py` calls load_model(8,5,str(seed)), `_init(model,"exact",.1)` and iterates the current half list in integer index order. The default start is `halves`, not the later truth-cone start. `_half(i)` collects the simplices emitted by wrong-class parts of exact-mode terminal cells, not the whole input partition or all leaf masses. Its auxiliary gate-pattern diagnostic is not part of the selected rays' numerical mass.

Selection increments n only over dimension5 arrays returned by each half and retains n modulo stride equal to zero. With a positive stride and a complete successful run, global selection indices are0,stride,2*stride,... below total, so K=ceil(total/stride). The script itself does not validate positive stride, seed0/2, finite arrays or that expected coverage formula. It saves neither half indices nor within-half offsets. These would be useful as a sidecar; absent them, exact emission provenance cannot be reconstructed from only total/stride.

Rays are ROW generators: B.Cone stores R(m,d), collection copies each dxd simplicial matrix, and estG integrates U @ R. Unit norms are intended floating constructions, not exact equalities. Independent integration should use the exact stored dyadic entries without transposing or renormalizing them. Gaussian cone mass is invariant to positive ray rescaling, but determinant/radial bounds depend on the supplied representation.

## Model, dispatch and rules

`zoo44.zoo` obtains alg_zoo.zoo_2nd_argmax(H,T,seed), then copies float-converted state_dict Wi,Wh,Wo. The exporter does not save checkpoint bytes/hash, resolved weights, alg_zoo version or their binding. The nominal H8T5 seed name alone is insufficient to establish identical weights with a previously published model.

Importing estP defaults T192_FAST to1, imports geomfast, patches B._adjacent_pairs and B.mass, and runs its compilation warmup. The intended emission order is therefore subject to pinned fast/pure geometry choice and numerical dependencies, not just the exporter file. m43 also incidentally loads m43_weights.npz at import even though H8T5 weights come from zoo44. manifest.json records this ancillary blob without claiming H8T5 dependence.

B.set_rules5 defaults to6 thin nodes/axis and128 log nodes; `_init` honors T192_RULES5 if set. DEV initially is cpu and the exporter never assigns CUDA. `mass_estT` is the Torch estG rule, not a direct call to scalar B._orth5. The NPZ does not record environment values, actual nodes/weights, library versions, rule thresholds or device. Preserve the actual run's settings rather than infer intended defaults from comments.

## Conditional per-simplex source-mass hazard

estG initializes SLIVER from environment T192_SLIVER. The exporter computes `m_estT=estG.masses_each(R,5)` BEFORE explicitly setting SLIVER=.6 for the reference arm; it does not reset SLIVER toNone beforehand.

In `_orth5`, when SLIVER is non-None and thin rows exist, all thin entries of out are set to0 and `_thin_split(Rt_,...)` returns the SUM over all thin parents, which is stored only at the first thin row. This is intentional for summed `masses`, but `masses_each` returns that array directly. Therefore with an inherited non-None SLIVER and at least two thin input rows, `mass_estT` is not a per-parent mass vector: later thin rows are zero and the first combines their contributions. The same issue exists for its d4 counterpart, outside this T5 scope.

No actual exporter environment or arrays are available here, so occurrence is unestablished. Default SLIVER=None avoids this particular issue. A qualifying source-mass comparison must bind the actual initial SLIVER value; affected arrays should be retained and labeled aggregate-only, not silently repaired or treated as zeros. DRI owns any source repair and separately bounded regeneration decision.

## Reference dependence and termination

The second arm calls B.set_rules5(10,256), clears estG._CACHE and sets SLIVER=.6, then invokes summed `estG.masses` separately on each singleton r. Singleton calls avoid cross-parent redistribution. Thin cones use repeated midpoint splitting and the positive Duffy rule; wide cones still use the same cofactor/log family with changed node count.

`sliver_split` has maxlev24 and returns the remaining R after the cap. It does not return a convergence flag, actual achieved minimum pair dot or per-parent residual. Thus comments saying pairwise cos>=.6 describe the intended exit condition, not a guaranteed postcondition. Floating midpoint normalization also lacks zero/underflow/outward-geometry certification.

`mass_ref` shares source formulas, determinant/normal constructions and numerical environment with `mass_estT`. It is a higher-rule/refined diagnostic, not an independent implementation or a certified truth. Its error cannot be inferred merely from positivity or its agreement with the baseline. Any independent comparison uses the saved rays, not `mass_ref` as a certificate.

## Missing execution evidence

The exporter writes its NPZ only at the end, permits overwrite, retains selected arrays in memory, and calls masses_each on the entire selection without CHUNK slicing. No per-half durable completion manifest, bounded resource protocol or environment log is saved by this script. This does not prove an existing run exceeded resources; it precludes claiming our local resource/restart protocol from this source alone. This task authorizes no execution or rerun.
