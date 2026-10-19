# Novelty review of the positive results (revised 2026-10-19, second pass)

↑ [lit/ index](README.md) · [INDEX](../INDEX.md) · [web log](web_log.md)

**Question (owner).** Was each result reported before, especially in adversarial robustness and certification, and
how should we describe it? Which methods from that literature have we not tried?
**Method.** The first pass read mostly abstracts. This pass read the method sections of [V1], [V6-V9], [R5], [L5],
[N1], [B9] and StarV's code [V2b], and targeted passages of the other PDFs downloaded ([V3], [V4], [V10-V12], [R1],
[R7], [N2-N5], [P1], [P6], [P7], [B8], [RS1], [T1]); "(abs.)" and "(snippet)" mark weaker reads. "I found no ..." claims rest on targeted searches and are unverified. No new ARC/AlgZoo source was read
(ARC placement: first pass, [web_log](web_log.md) 2026-10-19). F = MSE / sampling MSE at equal compute.

**Bottom line.**
- Say "we apply X (cite)" for probabilistic branch and bound [V1], region enumeration with Gaussian masses [V2, V8],
  exact line restriction [N1], exact ellipse intersections [L5], safe LP bounds [B5, B9] and reliability
  sensitivity [S1-S4].
- Not found elsewhere: outward-rounded Gaussian masses of 10-D polytope cells; a floating-point-rigorous bound near
  1e-11 on an unrolled RNN; deterministic rare-event estimates near 1e-10 without full region enumeration; repair
  aimed at a 1e-9 distribution-level failure probability; a sealed equal-FLOP win over sampling for this model class.

## 1. Adversarial robustness and verification literature

**Sound probabilistic verification** bounds P_x[net(x) violates a spec] for a given input distribution.
- **PV** [V1] splits the input box into hyperrectangles and bounds each with CROWN. Its lower bound is the mass of
  boxes proven safe; its upper bound is 1 minus the mass proven violating, so undecided boxes are charged in full.
  Gaussians enter by a prepended linear map (box masses are products of Phi); floating-point error is "not
  account[ed] for" (§4.1); benchmarks have 1-8 inputs and networks up to 10x10 (or 1x10,000).
- **ProbStar** [V2] runs exact star-set reachability (that is, enumerates linear regions) for truncated-Gaussian
  inputs and sums the star probabilities. StarV's code computes each star's mass with SciPy's Genz routine
  (`mvn.mvnun`), which is an estimate, not an enclosure [V2b].
- **Kofnov et al.** [V8] enumerate ReLU linear regions and sandwich the input density between piecewise polynomials
  on simplices of a compact box. They integrate exactly over the triangulated region polytopes (Lasserre). The
  experiments use 1-3 input dimensions (upper-lower CDF gap ~.18-.20 at 3-D), and I found no floating-point
  treatment.
- **WMI** [V9] is exact for piecewise-polynomial densities and handles Gaussians only by Monte Carlo. It notes that
  deterministic error bounds for polytope volumes are NP-hard.
- **PREMAP** [V7] builds preimage under- and over-approximations by LiRPA with input/ReLU splitting, for uniform
  inputs. It is sound given exact volumes, but its implementation estimates them by Monte Carlo.
- **Rigorous Gaussian masses.** I found a self-validating multivariate-normal method for rectangles only [B10]
  (snippet), and none for general polytopes.

**Probably-sound or statistical certificates.**
- PROVEN [V6] takes one CROWN linear-bound pair on an l_p ball and bounds P[g(X) > a] by the CDFs of the linear bounds
  (Hoeffding-type or Gaussian). It does no branching.
- CC-Cert [V10] (empirical Chernoff-Cramér bounds on sampled transformations), NPAQ [V4] (PAC model counting on
  binarised networks), #DNN [V3] and ε-ProVe [V11] (counting or tolerance limits on uniform boxes, on top of ProVe
  [V5]) and Fazlyab et al. [V12] (SDP bounds from input moments only).
- Randomized smoothing [RS1] estimates P_{δ~N(0,σ²I)}[f(x+δ) = c] around one input by Monte Carlo, with a
  1-α Clopper-Pearson certificate. It is the same kind of estimand as ours (the Gaussian measure of a decision region)
  but local, not rare, and sampled.
- Global robustness [V13] (abs.) is a worst-case Lipschitz notion.

**Rare events.**
- Adaptive multilevel splitting [R1] (uniform perturbations, Metropolis-Hastings moves).
- Bai et al. [R5]: importance sampling for ReLU networks under Gaussian inputs, a mixture of exponential tilts at
  dominating points found by sequential MILP, asymptotically efficient. The validation survey [R7] is black-box.
- LIN-ESS [L5] estimates Gaussian polytope probabilities by nested domains (Holmes-Diaconis-Ross), with analytic
  elliptical slice sampling that intersects ellipses and hyperplanes exactly; it reaches 3e-38 in 500-D.
- I found no work that integrates a network's decision exactly along lines to estimate a failure probability.

**Exact restrictions.** ExactLine [N1] (1-D; decision boundaries, exact integrated gradients) and SyReNN [N2]
(2-D). Neither computes Gaussian probabilities.

**RNN verification.**
- RnnVerify [N3]: unrolling multiplies network size by the sequence length, and verification is worst-case
  exponential in size; it avoids unrolling with invariants.
- POPQORN [N4] and Prover [N5] bound local robustness on short sequences; Prover's precision loss "accumulates along
  the frames".
- I could not identify the "Prabhakar et al." paper the owner recalled.

**Floating point.** Jia & Rinard [B9] show that verifiers which ignore floating-point error can be exploited. They
recommend conservative MILP via Neumaier-Shcherbina safe bounds [B5].

**Repair.** PRDNN [P1] repairs points and 1-D/2-D polytopes by an LP on one layer and reports drawdown; REASSURE [P6]
patches one linear region; NNrepair [P2], Arachne [P3] and [P4] repair points or regions; CARE [P7] localises by
causal attribution and drives sampled violation rates below 1%. None targets a probability far below 1e-3. The
classical analogue is reliability-based design optimisation [S4] (abs.); probabilistically robust learning [T4]
(abs.) trains on a sampled misclassification probability.

## 2. Our methods against this literature

### (a) Exact line walks with a Gaussian radial integral (H-E16, H-E17; great circles H-A3)
- **What the literature already has:**
  - exact 1-D integration along conditioning lines: line sampling [L1] and preintegration with QMC [L3];
  - directional simulation over rays [L2];
  - exact piecewise-linear restriction of a ReLU network to a line [N1] or plane [N2];
  - closed-form intersections of Gaussian great circles (ellipses) with hyperplanes, used for polytope probabilities
    [L5]; this is exactly H-A3's primitive for linear constraints;
  - fixed spherical designs exploiting homogeneity, on random (untrained) MLPs [A1, A2].
- **What is new in ours, if anything:** no single component. New are the combination (exact walks of a trained
  ReLU RNN, a radial Student-t weight from homogeneity, co-area tie faces, fixed scrambled-Sobol lines) and its
  equal-FLOP wins: T5 (T10, T379), T10 from 1e6 FPE (T429, T431), the 1-layer MLP (T387, T394).

### (a') The T431 strict pass
- **What the literature already has:** preintegrated QMC beating Monte Carlo on kinked integrands, in finance [L3].
  For networks I found region enumeration (≤ 8-D, [V1, V2, V8]) and sampling, but no equal-FLOP, held-out comparison
  of a deterministic accuracy estimator against sampling. ARC calls M_16,10 open and asks for a mechanistic estimate [M1].
- **What is new in ours:**
  - the result: F .540 [.343, .762] at 4e6 FPE on 100 sealed HP5 retrains, scored once by an independent scorer;
  - its limits: QMC-class, not deductive (DRI ruling); retrains, not noisy copies; the target model untested (T434).

### (b) Rare error probabilities along one coordinate (H-R1)
- **What the literature already has:**
  - line sampling for small failure probabilities [L1];
  - multilevel splitting [R1] and dominating-point importance sampling [R5] for networks;
  - nested-domain estimates for polytopes down to 3e-38 [L5];
  - sampling-free probabilities by full region enumeration, in ≤ 8-D [V1, V2, V8].
  - Every other network rare-event estimator I found samples or extrapolates, ARC's included [R2, R3].
- **What is new in ours:**
  - deterministic estimates within ~1.4x from 1.8e-10 to 7e-7 on non-triple-tie (model, threshold) pairs (T364,
    T366, T368);
  - ~80% within 2x on tie-dominated points (T403);
  - all on a 10-step RNN, without enumerating its regions.

### (c) Weight gradient of a rare failure, localisation, repair and conversion (H-R2, H-R3, T411)
- **What the literature already has:**
  - failure-probability derivatives as boundary integrals [S1], estimated by line sampling [S2] or sequential
    importance sampling [S3], and used for design [S4];
  - second-order constrained weight edits (Optimal Brain Surgeon [S5], abs.) and curvature penalties (EWC [S6], abs.);
  - localise-then-edit repair [P2, P3, P5, P7]; ARC's proposal to train against an estimated catastrophe
    probability [R3];
  - for the conversion effect: the robustness-accuracy trade-off [T1], repair drawdown [P1] and "negative flips" after
    updates [T2] (abs.).
- **What is new in ours:**
  - the target, a distribution-level failure probability of 1e-6 to 1e-9 (repair papers target points, polytopes or
    violation rates ≥ 1%); the intervention-checked comparator account (T385);
  - a 12-weight edit, 17-19x rarer for -.011 pp; random control 0/40 (T402, T410);
  - the rare-to-common conversion, measured; its mechanism (training near a local optimum of total error, T411) makes
    it expected in kind.

### (d) Certified upper bound (T398, T400, H-C1/T412, T414, T416)
- **What the literature already has:**
  - summing certified region masses in branch and bound [V1]; region enumeration with Gaussian masses [V2] or with
    polynomial density bounds integrated over triangulations [V8];
  - neuron splitting with LP triangle relaxations [B1, B2]; CROWN and β-CROWN [B3, B4]; cutting planes [B8];
  - safe LP bounds [B5], recommended for network verifiers [B9];
  - exp-linear simplex integrals [B6]; Genz-type MVN estimates [B7];
  - unrolling blow-up and bounds that loosen with sequence length [N3-N5].
- **What is new in ours:**
  - outward-rounded Gaussian masses of general 10-D polytope cells: tangent bounds on the log-density, exact
    triangulation, Hermite-Genocchi and ball arithmetic (H-C1). I found no rigorous Gaussian polytope enclosure;
  - an end-to-end floating-point-rigorous certificate, P(E1(2.5) on triple (1,8,2)) <= 1.36e-11. Caveats:
    provisional on T420; this slice holds none of the 17 Monte Carlo events (all on (9,3,2)); no all-triple
    certificate;
  - the late-ordering obstruction, measured: a sharp instance of the known unrolling problem. The event stays
    feasible on every prefix until late inputs are read, so feasibility pruning cannot act and PV-style charging pays
    the full mass. T414's CROWN looseness is known in kind.

### (e) Safe-use rating (T401, T407)
- **What the literature already has:** fragility curves P(damage | intensity) [F1]; operational-profile reliability,
  i.e. misclassification probability assembled over input cells with confidence [F3] (abs.); statistical model
  checking [F2].
- **What is new in ours:** rates computed from the weights, with a trust flag; partial (target unrated; 3 of 6
  checkable points rated, T407).

### (f) Mechanism-derived M_16,10 accuracy (T317f; H-L1, H-S1)
Unchanged from the first pass: ARC's M_4,3 estimators [M1] and heuristic arguments [M2] are the template. F 2.63
[1.93, 3.39] on a sealed panel (lenient pass, strict fail). No verification-literature analogue.

## 3. Revised verdicts and phrasing

| # | Result | Verdict | Confidence | Change |
|---|---|---|---|---|
| a | Walk quadrature (H-E16/H-E17) | known + novel aspect (combination, result) | medium-high | primitives now also in [L5] |
| a' | T431 sealed strict pass | likely new as a result; not new as a method | medium | new row |
| H-A3 | Great circles, 1-layer MLP | known, applied | high | [L5] is the same primitive |
| b | Deterministic rare-event walks | known + novel aspect | medium | sampling alternatives named |
| c | Rare-failure gradient and repair | known + novel aspect | medium | RBDO/OBS added |
| c' | Rare-to-common conversion | known in kind, new as a measurement | medium | was "likely new" |
| d | Certified-bound pipeline | known (we apply [V1, V2, B5]) | high | was "known + novel" |
| d' | Rigorous Gaussian polytope masses | likely new | medium | Kofnov [V8] is the closest prior |
| d'' | Late-ordering obstruction | known in kind, specific form new | medium | |
| e | Safe-use rating | known framing, applied | medium-high | [F3] added |
| f | Mechanism estimate | known approach + novel aspect | medium-low | unchanged |

**Phrasing to use.**
- **(a)** "We apply line sampling / preintegration with quasi-Monte Carlo lines (Koutsourelakis et al. 2004; Griewank
  et al. 2018), exact on each line by ExactLine restriction (Sotoudeh & Thakur 2019). On great circles it is the
  ellipse-intersection step of analytic elliptical slice sampling (Gessner et al. 2020), extended to ReLU networks."
- **(a')** "On 100 held-out retrains of the M_16,10 architecture, a fixed-node quasi-Monte Carlo estimator over exact
  line walks has 0.54x the MSE of plain sampling at equal FLOPs (95% CI .34-.76). It is QMC-class, not a mechanistic
  estimate in ARC's sense."
- **(b)** "Line sampling made deterministic by exact piecewise-linear walks. To our knowledge, these are the first
  sampling-free estimates of a trained network's failure probabilities near 1e-10 that do not enumerate all its linear
  regions. The sampling alternatives are multilevel splitting (Webb et al. 2019) and dominating-point importance
  sampling (Bai et al. 2020)."
- **(c)** "We apply reliability sensitivity analysis (Uryasev 1994; Lu et al. 2008) to network weights, with a
  second-order constrained edit as in Optimal Brain Surgeon (Hassibi & Stork 1993). Unlike point or polytope repair
  (PRDNN; REASSURE; NNrepair; CARE), the target is a 1e-9 failure probability over the input distribution. The repair
  converts rare errors into the common error, a trade-off expected near a trained optimum (cf. Tsipras et al. 2019),
  which we quantify."
- **(d)** "We apply probabilistic branch and bound (Boetius et al. 2025) with neuron splitting and LP relaxations
  (Ehlers 2017; Bunel et al. 2020), enumerating activation regions as in ProbStar (Tran et al. 2023) and Kofnov et al.
  (2025), and make it floating-point rigorous. LP certificates are re-verified with Neumaier-Shcherbina safe bounds,
  as Jia & Rinard (2021) recommend. Gaussian masses of polytope cells are outward-rounded, which we have not found
  elsewhere."
- **(e)** "A fragility-curve / operational-profile rating (Shinozuka et al. 2000; Zhao et al. 2021), computed from the
  weights and checked by conditional sampling."
- **Avoid:**
  - "first certified rare-event probability for a neural network" (PV, ProbStar and Kofnov certify or compute such
    probabilities in low dimension);
  - "new method" for (a), (b) or (d) as a whole.

## 4. Methods to try next (ranked)

Goal (i): ARC's accuracy estimate, deductive or with a weight-derived error bound. Goal (ii): estimate, locate, rate,
repair and certify rare failures, including late orderings. Costs are rough.

1. **Dominating-point importance sampling** [R5] (goal ii: estimation plus a labelled high-confidence bound).
   - **What:** MILP over the unrolled net (160 ReLUs) finds the highest-density event points per triple; a mixture of
     tilts gives unbiased estimates with confidence intervals.
   - **Why it may work:** it does not depend on the triple-tie closures that break the walks (T364, T367, T369/T370;
     T403 ~80%). Plain MC needed 1.1e11 draws for a Clopper-Pearson bound of 2.3e-10 (T400).
   - **Cost:** 1-2 agent-days, < 20 CPU-h.
   - **First experiment / done when:** E1(2.5) on the target; relative s.e. <= 10% at <= 1e8 forward passes, within
     2 s.e. of T400's 1.55e-10; then T403's 15 points within 2x on >= 14.
2. **Halfspace charging inside our branch and bound** (PROVEN's principle [V6] inside PV [V1]; goal ii, late
   orderings).
   - **What:** charge each undecided cell with the Gaussian mass of cell ∩ {x : the cell's linear lower bound on the
     event margin, from the LP dual or CROWN, allows the event}, not the whole cell. Dual certificates (T400) and
     polytope masses (H-C1) exist.
   - **Why it may work:** late triples fail because the event is feasible on every prefix (T398) and undecided cells
     are charged in full (T414 stalls at ~50% of slab mass); a thin but feasible event can have a small halfspace mass.
     Even partial success helps: a certified superset S gives P = m(S) P(E | S), with P(E | S) sampled inside S
     (truncated-Gaussian samplers [L5]).
   - **Cost:** 1-2 agent-days, 10-50 CPU-h.
   - **First experiment / done when:** triple (3,7,8) at k = 3-4 (792 cells at k = 3); GO if the summed charge is
     < 1e-9 at <= 1e5 cells, KILL if still > 10% of slab mass at k = 4 with no fall per extra step.
3. **GCP-CROWN cutting planes** [B8] on T418's late-triple spec (goal ii), only if T418's α/β-CROWN does not GO.
   - **What / why:** MIP-generated cuts in bound propagation, aimed at the residual looseness (176 margin units, T414).
   - **Cost:** CPLEX (academic licence) for cuts; 1 agent-day, ~10 CPU-h.
   - **Done when:** root looseness < 10 margin units (then BaB), or > 50 (KILL).
4. **Multilevel splitting with exact ellipse moves** (LIN-ESS [L5] generalised by our great-circle walks; goal ii,
   estimation and checking repairs, which T402 verified by sampling only).
   - **What / why:** rejection-free slice moves inside {margin >= level}; no dominating points needed, which suits
     events spread over several triples and thin g12 bands (T373, T386).
   - **Cost:** 2-3 agent-days, CPU-light.
   - **Done when:** T403's points within 2x on >= 14/15 with valid CIs at <= 1e7 FPE.
5. **Hybrid labelled certificate** ("probably sound" reporting as in [V3, V11]; goal ii, a deliverable now).
   - **What:** "<= A certified on triples S, plus <= B at confidence 1-δ on the rest", never merged.
   - **Cost / done when:** ~0.5 agent-day after method 1; both parts reported for E1(2.5).
6. **Time-split forward/backward bounds** (PREMAP's preimage over-approximation [V7] over the last steps;
   RnnVerify's [N3] avoidance of full unrolling; goal ii).
   - **What / why:** bound the event's preimage in (h_b, late inputs), then the prefix's chance of reaching it; each
     side relaxes fewer steps, against looseness that grows with depth (E19: 5e5x interval blow-up; T414).
   - **Cost:** 3-5 agent-days.
   - **First experiment / done when:** (3,7,8), b = 8; GO if the prefix's chance of reaching the preimage is
     rigorously <= 1e-9, KILL if the preimage holds > 10% of sampled h_8 states.
7. **QMC point sets matched to preintegration** (theory in [L3]; goal i).
   - **Why:** T429's walk MSE falls faster than 1/N, a QMC-rate signature, but the T = 10 walks used only Sobol with
     one rule seed; lattice rules were tried only on the 1-layer MLP (T387, defective generator) and in T357.
   - **Cost:** 1 agent-day, 5-20 CPU-h.
   - **Done when:** on DEV3 at 1e6/4e6 FPE, a lattice rule (with periodising transform) gives a CI upper bound on F
     below .5, or no gain (stop). It stays QMC-class.

**Not recommended:**
- interval, zonotope or Lipschitz bounds (E19: 5e5x blow-up over 10 steps; spectral radius of |W| 3.76);
- ProbStar or full enumeration (T373: >= 2.7e15 regions; Genz masses are not rigorous);
- PROVEN alone (one linear bound over the whole space, as loose as T414);
- control variates from linear proxies (T380: correlation 0-.5);
- PV-style certified accuracy bounds for goal (i) (E19 certified nothing).

I found no literature method that makes goal (i) deductive; a weight-derived bound on the variation between lines
remains open.

## References
- [L1] Koutsourelakis, Pradlwarter, Schuëller 2004, Prob. Eng. Mech. 19(4):409-417. https://researchportal.hw.ac.uk/en/publications/reliability-of-structures-in-high-dimensions-part-i-algorithms-an/
- [L2] Nie, Ellingwood 2000, Struct. Safety 22:233-249, doi:10.1016/S0167-4730(00)00014-X
- [L3] Griewank, Kuo, Leövey, Sloan 2018, J. Comput. Appl. Math. 344:259-274. https://arxiv.org/abs/1712.00920
- [L5] Gessner, Kanjilal, Hennig 2020, AISTATS (LIN-ESS). https://arxiv.org/abs/1910.09328
- [N1] Sotoudeh, Thakur 2019, NeurIPS (ExactLine). https://arxiv.org/abs/1908.06214
- [N2] Sotoudeh, Thakur 2021, TACAS (SyReNN). https://arxiv.org/abs/2101.03263
- [N3] Jacoby, Barrett, Katz 2020, ATVA (RnnVerify). https://arxiv.org/abs/2004.02462
- [N4] Ko, Lyu, Weng, Daniel, Wong, Lin 2019, ICML (POPQORN). https://arxiv.org/abs/1905.07387
- [N5] Ryou, Chen, Balunović, Singh, Dan, Vechev 2021, CAV (Prover). https://arxiv.org/abs/2005.13300
- [A1] jamesrahenry 2026-08-18, Phase 1 technique census, ARC White-Box Estimation Challenge. https://discourse.aicrowd.com/t/18157
- [A2] AIcrowd submissions 325170 and 327651. https://discourse.aicrowd.com/uploads/short-url/bIyC1NJefuKuVEAZY8jk1PIKn8L.pdf , https://discourse.aicrowd.com/uploads/short-url/fKTnYwvOCAYWzpeh6w8uSUuKomJ.pdf
- [R1] Webb, Rainforth, Teh, Kumar 2019, ICLR. https://arxiv.org/abs/1811.07209
- [R2] Wu, Hilton 2025, ICLR. https://arxiv.org/abs/2410.13211
- [R3] Xu et al. 2024-09-13, ARC blog. https://www.alignment.org/blog/estimating-tail-risk-in-neural-networks/
- [R5] Bai, Huang, Lam, Zhao 2020, arXiv 2010.04890; 2022 ACM version at NSF PAR 10416590 (journal unverified). https://arxiv.org/abs/2010.04890
- [R7] Corso, Moss, Koren, Lee, Kochenderfer 2021, JAIR (survey). https://arxiv.org/abs/2005.02979
- [RS1] Cohen, Rosenfeld, Kolter 2019, ICML. https://arxiv.org/abs/1902.02918
- [S1] Uryasev 1994, J. Comput. Appl. Math. 56:197-223. https://pure.iiasa.ac.at/id/eprint/12769/
- [S2] Lu, Song, Yue, Wang 2008, Struct. Safety 30(6). https://pure.nwpu.edu.cn/en/publications/reliability-sensitivity-method-by-line-sampling
- [S3] Papaioannou, Breitung, Straub 2018, Struct. Safety 75:24-34. https://portal.fis.tum.de/en/publications/reliability-sensitivity-estimation-with-sequential-importance-sam/
- [S4] Valdebenito, Schuëller 2010, Struct. Multidisc. Optim. 42(5):645-663 (abs.). https://www.iaorifors.com/paper/68810
- [S5] Hassibi, Stork 1992, NeurIPS 5 (Optimal Brain Surgeon; abs.). https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html
- [S6] Kirkpatrick et al. 2017, PNAS (EWC; abs.). https://arxiv.org/abs/1612.00796
- [P1] Sotoudeh, Thakur 2021, PLDI (PRDNN). https://arxiv.org/abs/2104.04413
- [P2] Usman, Gopinath, Sun, Noller, Păsăreanu 2021, CAV (NNrepair). https://arxiv.org/abs/2103.12535
- [P3] Sohn, Kang, Yoo 2023, ACM TOSEM 32(4) (Arachne). https://arxiv.org/abs/1912.12463
- [P4] Goldberger, Adi, Keshet, Katz 2020, LPAR-23. https://easychair.org/publications/paper/CWhF
- [P5] Meng, Bau, Andonian, Belinkov 2022, NeurIPS (ROME). https://arxiv.org/abs/2202.05262
- [P6] Fu, Li 2022 (REASSURE; venue unverified). https://arxiv.org/abs/2110.07682
- [P7] Sun, Sun, Pham, Shi 2022, ICSE (CARE). https://arxiv.org/abs/2204.09274
- [T1] Tsipras, Santurkar, Engstrom, Turner, Madry 2019, ICLR. https://arxiv.org/abs/1805.12152
- [T2] Yan et al. 2021, positive-congruent training (abs.). https://arxiv.org/abs/2011.09161
- [T4] Robey, Chamon, Pappas, Hassani 2022, probabilistically robust learning (abs.). https://arxiv.org/abs/2202.01136
- [F1] Shinozuka, Feng, Lee, Naganuma 2000, J. Eng. Mech. 126(12):1224-1231. https://ascelibrary.com/doi/10.1061/(ASCE)0733-9399(2000)126:12(1224)
- [F2] Legay, Delahaye, Bensalem 2010, RV, LNCS 6418. https://arxiv.org/abs/1005.1327
- [F3] Zhao, Huang, Banks, Cox, Flynn, Schewe, Huang 2021 (abs.). https://arxiv.org/abs/2106.01258
- [V1] Boetius, Leue, Sutter 2025, ICML, PMLR 267:4660-4699 (PV). https://arxiv.org/abs/2405.17556
- [V2] Tran, Choi, Okamoto, Hoxha, Fainekos, Prokhorov 2023, HSCC (ProbStar; read via [V2b]), doi:10.1145/3575870.3587112
- [V2b] Tran et al., StarV tool paper, CAV (artifact-evaluated). https://par.nsf.gov/servlets/purl/10649230 ; code StarV/set/probstar.py, `estimateProbability`. https://github.com/V2A2/StarV
- [V3] Marzari, Corsi, Cicalese, Farinelli 2023, IJCAI (#DNN). https://arxiv.org/abs/2301.07068
- [V4] Baluta, Shen, Shinde, Meel, Saxena 2019, CCS (NPAQ). https://arxiv.org/abs/1906.10395
- [V5] Corsi, Marchesini, Farinelli 2021, UAI, PMLR 161:333-343 (ProVe). https://proceedings.mlr.press/v161/corsi21a.html
- [V6] Weng et al. 2019, ICML, PMLR 97:6727-6736 (PROVEN). https://arxiv.org/abs/1812.08329
- [V7] Zhang, Wang, Kwiatkowska, Zhang 2025, JMLR 26 (PREMAP). https://arxiv.org/abs/2408.09262
- [V8] Kofnov, Kapla, Bartocci, Bura 2025, ICML. https://arxiv.org/abs/2502.11672
- [V9] Morettin, Passerini, Sebastiani 2024 (WMI verification). https://arxiv.org/abs/2402.04892
- [V10] Pautov et al. 2022, AAAI (CC-Cert). https://arxiv.org/abs/2109.10696
- [V11] Marzari, Corsi, Marchesini, Farinelli, Cicalese 2024, AAAI (ε-ProVe). https://arxiv.org/abs/2308.09842
- [V12] Fazlyab, Morari, Pappas 2019, CDC. https://arxiv.org/abs/1910.04249
- [V13] Leino, Wang, Fredrikson 2021, ICML (global robustness; abs.). https://arxiv.org/abs/2102.08452
- [B1] Ehlers 2017 (Planet). https://arxiv.org/abs/1705.01320
- [B2] Bunel, Lu, Turkaslan, Torr, Kohli, Kumar 2020, JMLR 21(42). https://jmlr.org/papers/v21/19-468.html
- [B3] Zhang, Weng, Chen, Hsieh, Daniel 2018, NeurIPS (CROWN). https://arxiv.org/abs/1811.00866
- [B4] Wang, Zhang, Xu, Lin, Jana, Hsieh, Kolter 2021, NeurIPS (β-CROWN). https://arxiv.org/abs/2103.06624
- [B5] Neumaier, Shcherbina 2004, Math. Program. 99(2):283-296. https://optimization-online.org/2002/06/494/
- [B6] Baldoni, Berline, De Loera, Köppe, Vergne 2011, Math. Comp. 80:297-325. https://arxiv.org/abs/0809.2083
- [B7] Genz, Bretz 2009, Lecture Notes in Statistics 195, doi:10.1007/978-3-642-01689-9
- [B8] Zhang, Wang, Xu, Li, Li, Jana, Hsieh, Kolter 2022, NeurIPS (GCP-CROWN). https://arxiv.org/abs/2208.05740
- [B9] Jia, Rinard 2021, SAS (floating-point soundness). https://arxiv.org/abs/2003.03021
- [B10] Wang, Kennedy 1992, self-validating multivariate normal probabilities for rectangles, Comput. Stat. Data Anal. (snippet). https://www.sciencedirect.com/science/article/abs/pii/0167947392900073
- [M1] Hilton 2026-01-26, ARC blog (AlgZoo). https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/
- [M2] Christiano, Neyman, Xu 2022. https://arxiv.org/abs/2211.06738
