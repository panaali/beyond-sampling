# lit/web_methods: web scan for deterministic, better-than-sampling ways to compute M_16,10's accuracy (T22b)

↑ [lit/ index](../README.md) · [TASKS](../../TASKS.md) · [INDEX](../../INDEX.md) · [web log](../web_log.md)

## Question
Which published techniques (2015-2026, any field) could compute P(argmax W_o h_10(x) = index of 2nd-largest x),
x ~ N(0, I_10), **without random inputs** and with MSE below plain sampling's p(1-p)/N at N ~ 1e3-1e6 forward-pass
equivalents (FPE)? The hard facts, from the [ledger](../../INDEX.md#4-hypothesis-ledger): more than 1e7 linear regions (H-M7);
gate pieces are far finer than the correct/wrong pattern (H-E7); additive models of the band miss by 4% even as an oracle
(H-E8); deterministic summaries of the low inputs overestimate accuracy (H-E10, and E28: the stochastic reduction is
also negative); the deterministic tie-face rule wins at T <= 4 and collapses at T ~ 5 as the face integrand F turns
heavy-tailed (H-E11, H-E12).

## Method
Web search plus reading of abstracts and pages (URLs below; nothing about ARC/AlgZoo after Jan 2026 was used except
what is logged in [web_log.md](../web_log.md)). Topics already covered in [lit/](../README.md) or in T19's
[literature leads](../../m16/approaches/crossover_compute/literature/README.md) (functional tensor train, HGM facet
flux, rank-respecting EP) are not re-proposed, only cross-referenced. Three tiny probes (one core, under 10 CPU-minutes
in total) check premises that decide the ranking:

| file | what it measures |
|---|---|
| [`monotone_probe.py`](monotone_probe.py) | Is the integrand *monotone* along sci_tao's tie segments (one wrong run that touches the tie)? This is the condition preintegration theory needs (Gilbert, Kuo and Sloan 2022) |
| [`cond_var.py`](cond_var.py), [`cond_var_last.py`](cond_var_last.py) | Share of the error variance explained by blocks of inputs in *time* coordinates (closed Sobol index of the first t or last k inputs) |
| [`split_rank.py`](split_rank.py) | Singular-value decay of the indicator matrix A[prefix, suffix] (the Hankel / tensor-train rank behind weighted-automaton and TT methods) |

## Status
Done (scan + probes). No estimator was built or scored.

## Results

### Probe results [measured]
1. **Monotonicity mostly holds along tie segments.** H=16, 2000 random segments per face, a 256-point grid per segment,
   segments clipped to sci_tao's region of the face. Share of error mass on non-monotone segments (two or more wrong
   runs, or a run away from the tie point): T=4: 0.01 / 0.00 (faces 23 / 12); T=5: 0.02 / 0.01; T=6: 0.05 / 0.07;
   T=8: 0.03 / 0.27 (7 lines, noisy); T=10: 0.04 / 0.02. So the T~5 collapse is **not** a loss of monotonicity in the
   Gilbert-Kuo-Sloan sense. The preintegrated F is a band width that is continuous almost everywhere; what goes
   wrong is its size distribution (heavy tails), not one-dimensional smoothness. Caveat: the grid misses runs thinner
   than 1/256 of a segment, so the counts are lower bounds.
   `OMP_NUM_THREADS=1 python lit/web_methods/monotone_probe.py 2000 16 3 4 5 6 8 10`
2. **Time-block ANOVA: almost no low-order structure in raw coordinates.** Var(E[f | first t inputs]) / p(1-p) is
   0.001, 0.003, 0.006, 0.018, 0.044, 0.065, 0.158, 0.240, 0.441 for t = 1..9. Conditioning on the last k = 1..6
   inputs explains 0.001, 0.007, 0.015, 0.032, 0.049, 0.098. Even five inputs explain under 5%, so the variance sits
   in high-order interactions. A consequence: an *exact* integral over the last 4 inputs, done for each outer node,
   pays off under outer plain MC only if it costs under ~15 FPE (5 inputs: 23 FPE). Generic QMC/sparse grids in
   x-coordinates have no low effective dimension to exploit. Work in order-statistic/face coordinates (as sci_tao does).
   `OMP_NUM_THREADS=1 python lit/web_methods/cond_var.py 2000 400 1` and `... cond_var_last.py`
3. **The raw indicator is far from low rank across time splits.** Error-indicator RMS is 0.21. The best rank-r
   approximation of the 800x800 matrix A[prefix, suffix] leaves RMS 0.165 (r=10), 0.131 (r=50) and 0.108 (r=100) at
   t=5. Every split t in {2,3,5,7,8} looks similar. For comparison, an iid Bernoulli matrix would leave about 0.20 at
   r=100. Weighted-automaton / TT representations of the *indicator* would need ranks in the hundreds. This does not
   test TT on the exact *preintegrated* band mass, which is T29's question (Codex tensor_probe: low oracle rank at a fixed skeleton, unreliable held-out cross means; global test in T32). It is an independent
   cross-check of the raw-indicator case only.
   `OMP_NUM_THREADS=1 python lit/web_methods/split_rank.py 800 0`

### Ranked techniques (most promising first, for this problem)
Below, "beats MC?" is judged at 1e3-1e6 FPE, d = 10, with more than 1e7 regions.

| # | technique (source, year) | what it computes; cost; known accuracy | beats MC here? |
|---|---|---|---|
| 1 | **Change of measure inside a deterministic face rule**: importance transform + preintegration + QMC (RQMC-IS with preintegration, [arXiv:2510.06705](https://arxiv.org/abs/2510.06705), 2025; conditional QMC with constrained active subspaces, Liu & Owen, [arXiv:2212.13232](https://arxiv.org/abs/2212.13232), 2023; minimax-tilted proposals, Botev [JRSS-B 2017](https://arxiv.org/abs/1603.04166)) | Integrates F/q under a node set mapped to a proposal q proportional to a cheap band-width proxy times the face density. Cost: nodes x (walk + proxy). For discontinuous payoffs it restores the expected QMC rates where preintegration alone loses them to concentration | **Most plausible.** It attacks the measured obstruction directly (tail mass of F, H-E12), and probe 1 shows the 1-D preintegration step is already sound. Depends entirely on a cheap proxy correlated with F in the wide-band tail |
| 2 | **Numerical smoothing + dimension-adaptive sparse grids / lattice rules** (Bayer, Ben Hammouda & Tempone, [Quant. Finance 2023](https://www.wias-berlin.de/people/bayerc/files/hammouda_tempone_21.pdf), preprint [arXiv:2111.01874](https://arxiv.org/abs/2111.01874); Griewank, Kuo, Leövey & Sloan [2018](https://arxiv.org/abs/1712.00920); monotonicity condition: Gilbert, Kuo & Sloan [arXiv:2112.11621](https://arxiv.org/abs/2112.11621), 2022) | Root-find the discontinuity along one variable, integrate that variable exactly, then use ASGQ/QMC on the smooth rest. On option-pricing problems up to d~20 it gives orders-of-magnitude gains over MC. Needs monotonicity, otherwise square-root singularities appear | Possible. Probe 1 says the monotonicity premise holds for more than 93% of error mass at T=10. Remaining risk: kinks of F at 1e7 gate faces may stop hierarchical surpluses from decaying (sci_tao's Sobol gained only 1-4x at T>=5) |
| 3 | **ARC cumulant/Hermite propagation for wide random MLPs** (Hilton et al., [blog 2026-05-07](https://www.alignment.org/blog/mechanistic-estimation-for-wide-random-mlps/), [arXiv:2605.05179](https://arxiv.org/abs/2605.05179); roots in "Formalizing the presumption of independence", [arXiv:2211.06738](https://arxiv.org/abs/2211.06738), 2022) | Propagates cumulants / Hermite expansions layer by layer with no inputs. Reported to beat MC over 7 orders of FLOPs (up to 100x) at depth 4, width 256. Guarantee: MSE eps^2 in O(n/eps^2) vs MC's n^2/eps^2, but only asymptotically in width, and for random (untrained) nets | **Not as an estimator** (width 16, trained, depth 10; E13 ceiling -0.35 pp with 4 cumulants; E28 negative). Worth trying as the **cheap proxy** that #1 needs (conditional band-width prediction given a face point) |
| 4 | **Deep importance sampling with squared tensor trains** (Cui, Dolgov & Scheichl, [SISC 2024, arXiv:2209.01941](https://arxiv.org/abs/2209.01941)) | Builds an IS density by a chain of TT-based maps along bridging densities. Cost per estimate is nearly flat as the event probability goes to 0 | Sampling (a baseline, not a solution). Could be the strongest sampling reference; the fitted TT density of the error set is an inspectable object. Probe 3: the raw indicator is high-rank, so it needs smoothed bridging levels |
| 5 | **Single-polytope Gaussian measure**: Genz SOV + lattice; Botev minimax tilting; holonomic gradient method (Koyama & Takemura [arXiv:1211.6822](https://arxiv.org/abs/1211.6822); Koyama [arXiv:1512.06564](https://arxiv.org/abs/1512.06564), [arXiv:1311.6905](https://arxiv.org/abs/1311.6905)); EP (Cunningham et al. [arXiv:1111.6832](https://arxiv.org/abs/1111.6832)); LIN-ESS/HDR (Gessner et al. [AISTATS 2020](https://proceedings.mlr.press/v108/gessner20a.html)) | One cone/polytope to 1e-5..1e-8 in ms (HGM is exact up to ODE error; rank 2^m in dimension m) | Not globally: more than 1e7 cones (E14: 2000 cones = 4% of error mass at 1.27M evals). Useful only for exact stratum weights or T31's facet flux |
| 6 | **Probabilistic / quantitative verification**: ProbStar/StarV (Tran et al., [HSCC 2023](https://unpaywall.org/10.1145%2F3575870.3587112), [StarV CAV](https://par.nsf.gov/biblio/10649230)); PREMAP (Zhang, Wang, Kwiatkowska, Zhang, [JMLR 2025, arXiv:2408.09262](https://arxiv.org/abs/2408.09262)); exact CDF bounds (Kofnov et al., [ICML 2025](https://proceedings.mlr.press/v267/kofnov25a.html)); characteristic-function propagation (Pilipovsky et al., [L4DC 2023](https://proceedings.mlr.press/v211/pilipovsky23a.html)); BaB-prob ([arXiv:2509.25647](https://arxiv.org/abs/2509.25647), 2025); #DNN-Verification (Marzari et al., [IJCAI 2023](https://www.ijcai.org/proceedings/2023/25)); activation-pattern laws via orthant probabilities ([arXiv:2503.22082](https://arxiv.org/abs/2503.22082), 2025); parallel exact region enumeration ([arXiv:2403.00860](https://arxiv.org/abs/2403.00860), 2024) | Enumerate or relax polytopes (star sets, preimage unions, BaB splits) for sound bounds or threshold decisions (P >= eta). Tested on small low-dimensional nets or loose eta. BaB-prob even computes per-branch probabilities by MC | No. Cost grows with region count; precision targets are about 1e-2, not 1e-3 to 1e-4; E19 (certified B&B) certified nothing at 1.7e7 evals |
| 7 | **Discrepancy theory as a ceiling** (Beck; Bilyk & Brauchart, [arXiv:2502.15984](https://arxiv.org/abs/2502.15984), 2025; harmonic ensembles [arXiv:2308.06216](https://arxiv.org/abs/2308.06216)) | Spherical-cap L2 discrepancy is at least of order N^(-1/2-1/(2d)) on S^d, and this is attainable. By homogeneity f lives on S^9 | Not a method. [inferred] For generic node sets on indicators of sets with curved/polyhedral boundaries, the MSE gain over MC is limited to about N^(1/d) (about 3.6x at N=1e5, d=9). This matches the 1.3-3x seen for fixed Sobol rules at T=10. Only exact smoothing (#1, #2) can escape it |
| 8 | Weighted automata / TT / spectral (Hankel) learning of RNNs (Rabusseau, Li & Precup, [AISTATS 2019](https://proceedings.mlr.press/v89/rabusseau19a.html); [arXiv:2010.10029](https://arxiv.org/abs/2010.10029)) | Represents a sequence function by a low-rank Hankel/TT structure; the Gaussian integral is then a product of small matrices | Raw indicator: no (probe 3). Preintegrated band mass: T29 found low oracle rank at a fixed skeleton; global test is T32 |
| 9 | Transfer operators / Ulam / sparse Ulam (Junge & Koltai [arXiv:0802.4006](https://arxiv.org/abs/0802.4006)); kneading/symbolic dynamics | Discretises the transfer operator of a map on boxes | No: 16-D state plus order-statistic state, with no finite automaton (E27: ~2 bits/step unpredictable); box counts are astronomic |
| 10 | Rare-event sampling (subset simulation, adaptive multilevel splitting, HDR) | Sampling | Deprioritised (p = 0.047 is not rare; sampling class) |
| 11 | Moment/SOS volume hierarchies (Henrion-Lasserre-Savorgnan), tropical rational-map views of ReLU nets, Edgeworth/saddlepoint for order statistics | SDP sizes C(10+2k,10); tropical forms have as many monomials as regions; order-statistic face laws are already exact 1-D tables in sci_tao | No new leverage |

### Concrete plans for the top 3

**#1 Proxy-weighted deterministic face rule** (backlog T33). Keep sci_tao's exact pieces: the co-area split into tie
faces, exact face laws, and the exact segment walk that returns F(y) at a face point y. Replace the uniform face measure
by q(y) proportional to phi_face(y) * w~(y), where w~ is a *model-derived, cheap* band-width proxy. Candidates are the
linearised tube width |d margin / ds|^-1 * margin-noise scale at y (one forward + one tangent pass), or a
conditional-cumulant width (#3). Map a fixed lattice/Sobol rule through the inverse Rosenblatt map of q (or a
Botev-style tilt per position configuration) and integrate F/w~ * Z_q. If corr(F, w~) is high in the top 1% of faces,
F/w~ is light-tailed and QMC's rate returns. Expected cost: about (1 walk + 1-2 proxy passes) per node, i.e. about
sci_tao's cost + 20-50%. Gain needed: the tail share of E[(F/w~)^2] must fall from 50-76% (sci_tao `diag_var.py`) to
under ~15%. Main risk: in the wide multi-cell bands the linear proxy is off by 0.95-3.7x (E24), and the proxy itself
needs per-model fitting (advice). **Cheapest test** (light, no estimator): on a stored set of sci_tao face points
(DEV models, T=5 and T=10), compute F and each proxy and report Spearman corr, the top-1% tail share of F^2 versus
(F/w~)^2, and the implied variance ratio. Go only if the variance drops more than 3x after charging the proxy cost.

**#2 Numerical smoothing + adaptive sparse grids / weighted lattices on the face** (backlog T34). Probe 1 shows the
error set along each tie segment is one run touching the tie for more than 93% of error mass, so the preintegrated F
is continuous almost everywhere (the Gilbert-Kuo-Sloan condition). Use Bayer-Ben Hammouda-Tempone style ASGQ, or a
CBC lattice with product weights, in face coordinates ordered by sensitivity (top gaps first, positions exact).
Handle the non-monotone 2-7% of segments with the existing exact walker, which already returns all runs. Cost: the
same per-node walk; ASGQ adds index-set bookkeeping only. Main risk: F has kinks wherever a band edge crosses one of
about 1e7 gate faces, so the hierarchical surpluses may not decay, giving back the 1-4x of Sobol. **Cheapest test**:
along 200 random 1-D lines in face coordinates (T=5 and T=10 DEV models), compute F on dyadic levels 1-8 and fit the
decay of the hierarchical surplus. ASGQ is worth building only if the decay is faster than 2^(-1.5 l) in the leading 3
directions.

**#3 Hermite-corrected conditional cumulants as the proxy for #1** (backlog T35). Read arXiv:2605.05179 for its
corrections (the blog gives no details; the abstract mentions cumulants and Hermite expansions). Apply them not to the
whole network (E13 ceiling) but to the low-input noise term of the tie margin *conditional on a face point*, which
E28 showed is the hard part. The output is a predicted band width w~(y) and band-mass F~(y) at O(H^2 T) FLOPs, with
no forward passes on random inputs. Use it (i) as #1's proxy and (ii) as a control variate F - F~ inside the
deterministic face rule. Main risk: E28's frozen-skeleton reductions are off by 2-2.5 pp per skeleton, so F~ may
correlate poorly in exactly the wide-band tail. **Cheapest test**: the same stored face points as #1; report corr(F, F~)
in the top 1% of faces and Var(F - F~)/Var(F).

## Compute accounting
Model forward passes (1 FPE ~ 6e3 FLOPs at T=10) used by the probes:
- monotone_probe: 2 faces x 6 T values x 2000 segments x up to 256 points, about 6.1e6 passes at T <= 10.
- cond_var: about 2000 x 400 x 9 = 7.2e6 (partial-length) passes. cond_var_last: about 3.6e6.
- split_rank: 5 splits x 6.4e5 = 3.2e6.

Total about 2e7 short passes, about 6 CPU-minutes on one core. No LP solves, no estimator runs. Web reading is not
compute.

## Claims
- [measured] Along sci_tao tie segments, at least 93-98% of error mass (except one noisy cell) lies on segments with
  one wrong run that touches the tie, for H=16, T=4..10. Grid-limited, one training seed per T.
- [measured] Blocks of up to 5 consecutive inputs explain under 5% of the error variance of the original M_16,10.
- [measured] The error indicator across any time split has slowly decaying singular values (rank 100 removes about
  half its RMS).
- [inferred] The T~5 collapse (H-E11) is not caused by non-monotone preintegration. It is a variance-concentration
  (heavy-tail) problem in the band width, so change-of-measure ideas (#1) are the most direct literature remedy.
- [inferred] Fixed generic node sets cannot beat MC by more than a small polynomial factor on this indicator
  (discrepancy lower bounds), which is consistent with the measured 1.3-3x of fixed Sobol rules.
- [speculative] A cheap, model-derived proxy correlated with the rare wide bands exists. If it does not, #1-#3 all fail.
- [measured] No published method found computes Gaussian measures of unions of more than 1e6 polytopes to about 1e-3
  relative accuracy deterministically. Probabilistic-verification tools target bounds or thresholds on small nets.

## Alternatives not ruled out
- The monotonicity probe uses a 256-point grid. Thin secondary runs (exactly the rare multi-cell bands?) may be missed,
  so a walker-exact recount (sci_tao core) is needed before relying on it.
- Probe 1 used the zoo's default seed per T; models with other seeds may differ (T2 is replicating seeds).
- The ANOVA result concerns raw time coordinates. Order-statistic (Renyi) coordinates may have low effective dimension.
  This was not measured here and would sharpen #2.
- The split-rank test is on random rows/columns with n=800. Rank in a Gaussian-weighted continuous sense, and after
  smoothing, may be lower (T29, T32).
- The ARC 2026 paper's details were not read beyond the abstract and blog. Its corrections may already be what E13 tried.

## Failure modes
Every top-3 plan needs a proxy or smoothness that survives in the rare wide multi-cell bands, which is where every
previous method broke (H-E12). If the proxy is learned per model, it becomes advice and its cost must be charged.
Discrepancy limits apply to any fixed node rule that lacks such structure.
