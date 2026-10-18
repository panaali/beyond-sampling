# T275: bounded review of T272 Part 3

This review changes the interpretation of three diagnostics; it does not recompute their reported values. Exact author text is pinned at edbc2bfd789edc6a1ea1c46b851577bb97ea2b2a, with the earlier f1e14e9a preregistration preserved separately. Result files were inventoried by Git tree metadata only. No result JSON, arrays, models, face rules or candidate functions were evaluated.

## Decision-relevant findings

1. **F0c is an additive component diagnostic, not necessarily the probability of one frozen classifier's error event.** `f0.py::F_and_F0c` collects disjoint actual error components, then extends the entry-cell affine logits from each component's near endpoint to the same window edge and adds every extended mass independently. Different extensions can overlap. This agrees with the preregistered componentwise definition, but the sum must not silently be interpreted as a union probability or a globally consistent frozen classifier. In contrast, F0a freezes one gate word at the face and uses its affine logits over the window.

   A purely algebraic illustration on the positive half-window (0,5) is the continuous scalar margin
   `m(t)=1-t+2 ReLU(t-3/2)-2 ReLU(t-5/2)+2 ReLU(t-7/2)`.
   Its negative components are (1,2) and (3,4). Their entry-cell affine margins are 1-t and 3-t, so their frozen negative extensions are (1,5) and (3,5). The source's additive construction counts (3,5) twice. Endpoint ties have zero Gaussian measure. This is a scalar piecewise-linear counterexample to disjointness, not an observed trained-model failure or a claimed realization by the pinned recurrent model. It does not invalidate the author's explicitly secondary componentwise experiment; it limits what that experiment establishes.

2. **The measured kink statistic is sampled signature-change density.** `kinks.py::walk_sig` hashes wrong-piece boundary and gate-flip metadata into a 64-bit signature. `paths` counts `sig[1:] != sig[:-1]`. The preregistration explicitly defines “kink” this way. Stable signature supports a fixed local formula away from other source branches, but signature change does not prove a nonzero derivative jump of the integrated mass: a representational split may cancel from the sum. Multiple intervening changes, hash collisions and unsampled narrow changes can also be missed. The finite-difference “jump” flag is a heuristic with an additional absolute threshold, not a certified discontinuity test. Thus the reported finite association concerns this declared statistic. It does not establish a universal nonsmoothness count, convergence exponent, or that every model is free of resolution limitations. The reported resolution comparison covers only three models and finite sampled paths.

3. **F0a's eight aggregate scramble totals are not constructed from disjoint scramble seeds.** The seed formula is `700+r` for kind23 and `707+r` for kind12, with r=0,...,7. Seed707 appears in kind23/r7 and kind12/r0. Both use the same dimension and prefix length for one model, hence the same underlying Sobol rule. The corresponding aggregate rule totals share randomness. `std(ddof=1)/sqrt(8)` therefore is not justified merely by an independent-rule assumption; it is a descriptive spread-based statistic unless this covariance is accounted for. Pairing F and F0 on the same faces is appropriate and remains intact. F0c's four seeds (800...803 versus807...810) do not have this particular overlap. No direction or magnitude of the missing covariance is inferred. This observation alone does not reverse the author's large reported F0a differences.

## Source and evidence limits

- `core.py` constructs a finite [-8,8] tabulated CDF and interpolated face map, clips inverse-normal probabilities, and caps tau at8.5. The observed integrand is the frozen numerical pushforward/window, not an automatically certified ideal Gaussian integral.
- Vector F has a2000-cell limit; scalar F0c and signature walks have4000-cell limits and epsilon advances. Their return values do not supply a complete cap-exhaustion certificate. The scalar/vector check is finite and shares numerical conventions; it is not an independent error bound.
- Near-zero constant margins are treated with a relative tie tolerance. Exact first-index semantics for arbitrary near ties requires a separate qualification.
- F0 outputs retain paired aggregate rule totals, not component intervals or per-face overlap. Kink outputs retain aggregate counts and jump summaries, not full sampled paths/signatures. A saved-only scalar contraction cannot retrospectively prove missing per-face geometry or absence of exhaustion.
- The model loader calls current AlgZoo and applies the specified noise law; these source files alone do not bind original run-time checkpoint bytes. `analyse_kinks.py` also consumes `ht_seeds/table.json`; its numerical joins and correlations are not independently checked here.
- Shell launchers and reported elapsed seconds do not establish full charged FPE or a current-budget estimator pass. The old numerical claims remain author-reported, not independently confirmed by this note.

## Consequence and alternatives

The narrow F0a frozen-at-face candidate may still fail its preregistered finite accuracy threshold by a large margin; nothing here proves otherwise. F0c remains a secondary diagnostic requiring the original walk, not a cheap estimator. Neither negative excludes all restricted-domain, weighted, corrected or residual-band alternatives. The kink result should be called the finite signature-density test; it cannot kill dimension-driven or weighted-boundary explanations.

Two cheap discriminators, if later prioritized, are (i) a synthetic constant-integrand or redundant-gate control separating signature changes from actual derivative changes, and (ii) an overlap/union ledger for F0c extensions on fixed retained components. Neither has been executed or authorized here. An independent-cap/reference check is a separate alternative for numerical walk bias, not a reason to repeat the campaign automatically.

Under the newer DRI priority, broader payload arithmetic, additional source archaeology and new numerical experiments are paused. T267 and the H4T4 residual-band numerical discriminator take priority. This bounded note needs independent written review only; it grants no execution permission.
