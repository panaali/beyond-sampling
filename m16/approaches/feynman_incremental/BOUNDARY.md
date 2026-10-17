# T20 stochastic boundary baseline port

Status: next bounded phase claimed/pushed2b3dc47. This is a sampling baseline, never a deductive solution.

Method: preserve boundary's model-free A/B/R/blind decomposition and two-stage allocation, but explicitly replace its adjacent-error walk plus random remainder with completed corrected A/B masked-line integration. New line value on mlo<=|t|<=min(tau,mhi) comes from the existing native Gaussian adapter. No hidden approximate remainder. This changes RNG consumption and cost-driven allocation; whole-run legacy parity is not expected or a success criterion. The inherited two-stage reuse of pilot observations remains subject to adaptive-allocation bias; no new unbiasedness claim.

Charge every generated random coordinate, candidate rejection input (including discarded/overshoot candidates), table/normalizer setup, model forward/readout, source geometry and corrected line call. Original gen*n/got rejection discount is removed. Batches<=65536, explicit generation cap1000000 per requested zone call. Failures retain known work, not a default estimate. Tables200001 points charged500 each; adaptive scalar setup charged1000 per quadrature invocation plus1000 per integrand evaluation, with callback count recorded. Opaque quadrature internals remain unproved conventions. RNG proxy64 per uniform,100 per normal,50 per integer; surface geometry1000+230*9+80 per accepted face; blind transform geometry1000 per point; zone classification500 per generated candidate; exact affine/ReLU/readout+T² rank sort+argmax for direct model inputs; per-term aggregation1000+10*n plus10000 global allocation. Counts overlap conservatively where noted, not portable instruction bounds.

Frozen falsifiers:
1. A/B target/mask conversion may be wrong. Use one frozen RNG seed to generate16 faces per family, zero output; independently compute constant-label wrong Gaussian mass with70-digit erfc and compare every line, first-index ties and empty masks included.
2. Rejection accounting may still discard overshoot work. Direct R draw of8 accepted inputs must report its full generated batch, and cold component sum must equal returned cost. Cap failures are explicit.
3. Full line completion may change allocation or RNG sequence and look like an accuracy gain. Run only public checkpoint budget512 seeds0/1 and zero-output budget512 seed0; log model-input/RNG counts, no paired-point legacy claim, no ground-truth MSE sweep or confidence interval.
4. Adaptive reuse/table approximation/special-function costs may invalidate historical unbiasedness or compute claims. Keep these limitations explicit; full scoring requires separate arc_eval protocol.

Cycles<=110CPU under primary lock (<=20CPU primitive checks on review lock), one thread/2GB. No shared source edits. Feynman/tube/Laplace estimates remain unchanged. Fitted-importance/Newton variants stay open.

## Frozen setup-failure interface check
Initial normal runs complete. Before handoff, preserve spent charges if model-free setup itself fails, not only line/estimate failures. Inject a setup callback that charges17 and raises; require known total11017 and unresolved partial setup flag. This exercises interface failure accounting only, with no model integrations. Original source hashes and successful estimates stay unchanged.

## Bounded results
[measured] Thirty-two independent70-digit constant-label surface controls pass: maximum A/B absolute differences3.34e-16/1.12e-16. Rejection control requests8 points, generates and charges1000 candidates; cap exhaustion is explicit. All costs sum to returned ledgers. Three small end-to-end budget512 runs complete in total0.455CPU including primitive checks: trained seeds0/1 estimates0.9574513211429095/0.9380348782783853, zero output seed0 estimate0.12572999763491643 versus analytic0.1. These are stochastic point estimates, not bias/MSE estimates or accuracy wins.

[measured] Trained seeds0/1 cold proxies106534984/105989017 (17755.8307/17664.8362 conventional FPE), native CPU0.1184/0.1111sec. They evaluate34 sampled surface lines and45/47 direct model inputs, draw2000 rejected-zone candidates (1963/1961 discarded or overshoot), and charge all20087 standard normals,680 uniforms,8 integers plus777 model-free quadrature callbacks. No rejected draw is discounted by n/got. Cold table/setup dominates; opaque RNG/quad costs remain conventions. The zero-readout draw uses1000 candidates and has104969425 proxy operations. See boundary_results.json.

[measured] Injected setup failure retains11017 known operations and explicit unresolved partial cost, with zero model integrations. See boundary_interface_results.json. No default estimate is returned on failure.

Status: completed-line stochastic boundary variant complete for review. Public boundary_estimator.est(model,budget,rng) is a baseline, explicitly not deductive. It changes the old stochastic attached-component remainder and thus the RNG stream/allocations; no unchanged-estimator parity is asserted. Reproduce `python -m m16.approaches.feynman_incremental.check_boundary` and `python -m m16.approaches.feynman_incremental.check_boundary_interface`. Original-model-fitted importance and Newton variants remain unported; adaptive pilot reuse, model-free inverse interpolation, floating line accuracy and opaque numerical proxy charges remain separate unresolved qualifications.
