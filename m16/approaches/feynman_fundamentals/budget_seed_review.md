# Independent T44 budget and replicate review

The original R23 seeds r and R12 seeds r+7 overlap across the16 replicates.
Writing e_r=A(U_r)+B(U_(r+7)), with independent scramble streams U, gives
within-rule variance v=Var(A)+Var(B) but cross-rule covariance c=Cov(A,B)
at lag7. Thus

E[s^2]=v-2(R-7)c/[R(R-1)].

Since |c|<=sqrt(Var(A)Var(B))<=v/2, the relative expected sample-variance
bias is at most3.75% for R16 and2.5202% for R32. The mean-variance inflation
can nevertheless reach1+(R-7)/R=1.5625 at R16. The sign of covariance is
unknown. Ordinary iid-rule bootstrap assumptions fail; this does not by
itself explain orders-of-magnitude gain decline. Individual finite-sample
ratios are not bounded by the expectation calculation. Original panels are
retained as correlated-seed diagnostics rather than silently replaced.

The separately frozen rescue uses r and r+100000, preserving counts and cost
plans. Independent `check_budget_review.py` reads the raw clean panels and
pinned truth JSON, imports no author analysis code, verifies disjoint seed
namespaces, and recomputes all30 cell gains with60-digit Decimal arithmetic.
Both variance and truth-variance-subtracted MSE gain formulas agree to a
maximum relative difference1.90e-12. Raw source hashes are retained.

| Declared cold budget | T4 MSE gain | T5 | T6 |
| --- | ---: | ---: | ---: |
| 16000 | 2.675818848 | 1.247383723 | .798333735 |
| 64000 | 43.060239088 | 5.035738753 | 1.858432338 |

At64000 the geometric decline factors are8.550927918 and2.709670214.
The independent arithmetic took .011771421 CPU seconds and no model calls.

Fixed sample counts are chosen from an independent cost-only pilot before
production outcomes. The pilot/setup charges and actual budget deviations
are retained, avoiding an outcome- or realized-cost-dependent stopping rule.
These are declared cost proxies, not a portable instruction certificate.
The fixed five-model panel does not establish general scaling over models.

Read-only review of `analyse.py` confirms that the paired bootstrap uses the
same rule-index resamples across the model/budget panel and shares each
independent truth perturbation between its two budgets. This preserves the
intended common-random-rule pairing. Truth-variance subtraction assumes an
independent reference with the stated variance. The normal truth model,
finite16-rule bootstrap, clipping of truth draws, and nonlinear ratios are
approximations, not certified95% coverage. `truth_resolves_mse` is a heuristic
scale flag, not a proof that bias is resolved.

Nonpositive denominators are retained as unresolved, not dropped from the
bootstrap sample. Mapping an invalid gain to infinity is a declared reporting
convention; an undefined decline ratio has no universal upward direction.
Any intervals affected by undefined draws must retain an unresolved warning,
and cannot be described as conservative coverage solely from that mapping.
No independent bootstrap coverage claim or additional input simulation is
made here.
