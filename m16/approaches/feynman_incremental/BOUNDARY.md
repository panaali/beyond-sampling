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
