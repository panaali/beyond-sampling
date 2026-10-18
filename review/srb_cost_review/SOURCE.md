# Pinned call path and semantic boundaries

1. `leaderboard.load_estimator` imports the entry, reports default support
   M16T10, marks it sampling and overrides its advice to fitted_on=orig.
   `estimator.py` imports cv, reads alloc.json['plain'], and builds strat=True,
   cv=None. There is no control-variate fitting or basis9 evaluation on this
   entry's runtime path; basis9 is imported and creates model-free arrays.
2. `cv.make.est` chooses c=1.6+0.1, n=int(budget//c). Python binary64 floor
   division semantics must be preserved, not replaced by rational floor in a
   numerical parity test. Seven quotas are max(4,floor(alloc_h*n)); M=sum quotas
   need not equal n, and small budgets can be exceeded because M>=28.
3. `pc()` lazily loads all precomp.npz arrays once per module. Reusable model-free
   tables were generated with20M Gaussian prefixes; they are finite estimated
   stratum probabilities/feature means, not exact analytic probabilities merely
   because a docstring says exact. alloc.json is original-model pilot advice.
4. `draw_all` uses batches of50000 nine-dimensional prefixes, computes pstratum
   for every candidate and retains only unmet quotas. The stopping condition is
   all quotas satisfied. Generated candidates D are a multiple of50000. It
   returns both the kept arrays and D, but `xs,_=draw_all(...)` discards D.
   Original `draw` and precompute/feature/CV fitting paths are not invoked here.
5. `pstratum` sorts all nine values and computes relative top gaps, including
   division by abs(m1) without the floor used for m2. Zero synthetic prefixes can
   produce0/0; any such warning/output must be preserved, not suppressed as an
   unreported semantic repair. Labels use seven strata (blind plus six bins).
6. `Q` receives one concatenated M-by9 batch. It calls nine dense recurrent
   steps, one dense last-step offset product, sorts sixteen (for H16) candidate
   gate breakpoints and two target breakpoints, then adds infinite endpoints.
   The H+3 loop iterations test `(hi>lo).any()` over the WHOLE batch. If any row
   is active, all M rows execute midpoint, gate, both logit products and CDFs.
   NumPy where evaluates its branch expressions eagerly; even unselected divide
   or midpoint branches execute. radial=False here; stdtr/norm branches unused.
7. `_charged` multiplies the returned integer cost by3.3/1.6 at report time;
   the raw estimator charge is retained in raw data. This is no new cost
   measurement and cannot recover discarded candidate or interval activity.
8. `leaderboard.correct` uses first-index argmax; target uses sorted second
   largest input. In Q, a competing row with identical logit has dA=dB=0 and
   `never=True`, regardless of index. For Wo=0 all competitors trigger never,
   hence every interval contributes zero. For continuous independent inputs,
   actual first-index prediction0 has accuracy1/10. At a fixed prefix where
   input0 is the second largest, target0 remains correct when the last input is
   below that prefix second largest, a positive Gaussian interval. This is an
   exact source counterexample to universal correctness, not a measured trained
   failure or evidence that the historical scores are invalid.

No fix is applied. Tiny nonzero Wi cutoffs, coincident breakpoints, infinities
and repeated logits are controls to preserve and qualify. The universal tie
repair is distinct from the source-cost question and must have its own claim.
