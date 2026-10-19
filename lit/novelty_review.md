# Novelty review of the positive results (2026-10-19)

↑ [lit/ index](README.md) · [INDEX](../INDEX.md) · [web log](web_log.md)

**Question (owner).** Was each positive result reported before, how novel is it, and how should we describe it?
**Method.** Web search, 2026-10-19, primary sources, mostly abstracts (Boetius et al. [V1] and [A2] read further).
Only works found online are cited; "unverified" marks points I could not confirm. Post-Jan-2026 ARC/AlgZoo sources are
in [web_log.md](web_log.md). F = MSE / sampling MSE at equal compute (strict F <= 1, lenient F <= 10); numbers are
in the INDEX rows named.

## 1. Walk quadrature (H-E16; T10, T379, T389, T391, T393)
- **Closest prior:** line sampling [L1]; directional simulation with deterministic point sets [L2]; preintegration
  plus QMC [L3]; ExactLine [N1]; QMC designs on the sphere [L4]. Separately, several teams in ARC's 2026 white-box
  contest used deterministic spherical designs on bias-free random MLPs, exploiting homogeneity [A1, A2].
- **Same:** exact integration along conditioning lines (line sampling, preintegration); exact piecewise-linear line
  restriction (ExactLine); fixed low-discrepancy nodes; radius removed by homogeneity.
- **New:** exact line restrictions used for Gaussian-measure probabilities (ExactLine uses them for integrated
  gradients and decision boundaries; the contest entries evaluate points). Also the estimator itself for a trained
  ReLU RNN (Student-t line law, model-free strata, deterministic allocation) and its result: strict pass at T = 5,
  on fresh models and in wall time (T393); at T = 10 on the sampling line (CIs span 1).
- **Verdict:** known + novel aspect. **Confidence:** medium (large literature, abstracts only).
- **Phrase:** "Line sampling / preintegration with quasi-Monte Carlo lines (Koutsourelakis et al. 2004; Griewank et
  al. 2018), made exact per line by ExactLine-style restriction (Sotoudeh & Thakur 2019). Our contribution is a
  deterministic, FLOP-matched estimator for a trained ReLU RNN that beats sampling at T = 5 and ties it at T = 10."

## 2. Fixed great circles on the 1-hidden-layer MLP (H-A3; T345, T387, T394)
- **Closest prior:** directional simulation [L2]; SyReNN's exact 2-D restrictions [N2]; spherical QMC [L4]; [A1].
- **Same:** exact restriction to a 2-D plane, the radius removed by homogeneity, fixed point sets.
- **New:** only the measurement. Each first-layer unit flips twice per circle at closed-form angles, so circles are
  cheap. Fixed Halton/Sobol sets give F .03-.66 per set (strict pass on all 10 sets in T387 and in T394's independent
  check; T387's cyclic and lattice sets failed). Fixed sets are no better than random circles (T394). On M_16,10 the
  walks give F 15.4 (random circles, lenient fail).
- **Verdict:** known, applied. **Confidence:** medium-high.
- **Phrase:** "An application of directional simulation with exact 2-D restrictions (SyReNN; Sotoudeh & Thakur 2021)
  to a one-hidden-layer ReLU MLP, which beats sampling at ~4k FLOP-equivalents."

## 3. Rare-event walks (H-R1; T340, T360, T366, T368, T388, T403)
- **Closest prior:** line sampling for small failure probabilities [L1]; adaptive multilevel splitting for neural
  network property violations [R1]; ARC's rare-output estimation [R2, R3]; conditional Monte Carlo [R4].
- **Same:** a rare probability estimated by conditioning on lines through the failure region, one coordinate exact,
  quadrature across the rest (preintegration).
- **New:** fully deterministic: exact gate walks along x_p3 plus quadrature over the exact transverse law, within 2x
  (mostly 1.4x) from ~2e-4 to ~1e-10 on points not dominated by triple ties, reproduced by other agents (T360, T368);
  plain MC at that compute expects ~0 events. Also the triple-tie diagnosis and the graded-g12 fix (~80% within 2x on
  tie-dominated points, T403; panel F 4.57, strict fail, lenient pass). All NN rare-event work I found samples; none
  reports deterministic estimates near 1e-10.
- **Verdict:** known + novel aspect. **Confidence:** medium.
- **Phrase:** "Line sampling (Koutsourelakis et al. 2004) made deterministic with exact piecewise-linear walks and
  quadrature over the exact transverse law. To our knowledge, these are the first sampling-free estimates of a trained
  network's error probabilities down to ~1e-10 (within 2x where triple ties do not dominate)."

## 4. Rare-failure gradient, localisation and targeted repair (H-R2, H-R3; T372, T385, T402, T410, T411)
- **Closest prior:** derivatives of probabilities over sets defined by inequalities, as a volume term plus a surface
  term [S1]; reliability sensitivity by line sampling [S2] and by sequential importance sampling [S3]; conditional-MC
  gradients [R4]; network repair: PRDNN [P1], NNrepair [P2], Arachne [P3], minimal modifications [P4];
  locate-then-edit [P5]. ARC proposes training against an estimated catastrophe probability [R3].
- **Same:** the gradient is Uryasev's boundary term evaluated by line-conditional MC (a sampling-class
  reliability-sensitivity estimator); localise-then-edit is NNrepair's, Arachne's and ROME's plan.
- **New:** the target is a distribution-level failure probability of ~1e-6 to 1e-9 (the repair papers I found target
  points, polytopes or misclassified sets). Also: the n6/n7 comparator account checked by intervention (T385); a
  12-weight edit with accuracy-Hessian correction, 17-19x rarer for -.011 pp, reproduced (T410), random edits 0/40;
  and the finding that rare-error repair converts mass into the training-optimised common error (T411, mechanism
  inferred; the robustness-accuracy trade-off literature was not searched, so unverified).
- **Verdict:** known + novel aspect. The conversion finding is likely new, with low confidence. **Confidence:** medium.
- **Phrase:** "Reliability sensitivity analysis (Uryasev 1994; Lu et al. 2008) applied to network weights to locate
  and repair a rare failure, in the spirit of localise-then-edit repair (NNrepair; ROME). New are the
  intervention-checked attribution, a 17x targeted repair that beats a random control, and the finding that repair
  trades rare errors into the common error."

## 5. Safe-use rating (T401, T407)
- **Closest prior:** fragility curves, i.e. P(damage | intensity) [F1]; probabilistic robustness [R1]; statistical
  model checking [F2]; violation rates [V5].
- **Same:** a conditional failure probability under an operating condition is a fragility curve; checking it by
  sampling is statistical model checking.
- **New:** rates computed from the weights, with a trust flag. Partial: the target is unrated by K3 and 3 of 6
  checkable points are rated by the graded rule; dev3_s531 is rated at every t.
- **Verdict:** known framing, applied. **Confidence:** medium-high.
- **Phrase:** "A fragility-curve-style rating (Shinozuka et al. 2000): conditional error probabilities under stated
  input conditions, computed from the weights and checked by conditional sampling."

## 6. Certified upper bound (T373, T390, T398, T400, T412, T414, T415)
- **Status:** components verified (259,723 pruning certificates re-verified exactly; rigorous normalisation and mass
  bounds); no complete certificate yet (T416-T420 open; late triples obstructed; input-space BaB killed, T414).
- **Closest prior:** probabilistic verification by branch and bound [V1]; ProbStar Gaussian star sets [V2]; #DNN
  [V3]; NPAQ [V4]; ProVe [V5]; PROVEN [V6]; PREMAP [V7]; exact CDF bounds under random inputs [V8]; branch and bound
  with LP relaxations and neuron splits [B1-B4]; safe LP bounds [B5]; integration over simplices [B6]; Genz-Bretz
  MVN probabilities [B7], which are estimates, not enclosures.
- **Same:** BaB with LP relaxations and ReLU splits is standard; summing input-region masses is Boetius et al.'s
  approach (hyperrectangles, Gaussian CDFs) and ProbStar's; safe dual bounds are Neumaier-Shcherbina's; exp-linear
  simplex integrals are classical.
- **New:** rigorous (ball-arithmetic or outward-rounded) Gaussian masses of general polytope cells, by exact
  triangulation, Hermite-Genocchi and tangent bounds on log Phi. Boetius et al. leave floating-point error
  unaddressed; I found no NN verifier with validated Gaussian polytope masses (limited search, unverified). Also the
  ~1e-10 target on an unrolled 10-step RNN and the late-ordering obstruction (scoped).
- **Verdict:** known + novel aspect. **Confidence:** medium-low.
- **Phrase:** "Probabilistic verification by branch and bound (Boetius et al. 2025; β-CROWN-style splitting), with
  exactly re-verified LP certificates (Neumaier & Shcherbina 2004) and rigorous ball-arithmetic Gaussian masses of
  polytope cells, which we believe are new in this setting. No complete certificate for the 1e-10 event exists yet."

## 7. Mechanism-derived, sampling-free accuracy (T317f; H-L1, H-S1)
- **Closest prior:** ARC's M_4,3 estimators (Hilton's notes, linked from [M1]; see [web_arc §2](web_arc/README.md));
  heuristic arguments [M2]; surprise accounting [M3]; compact proofs [M4]; the matching-sampling principle [M5].
- **Same:** a tiny trained net's accuracy estimated from a mechanism with declared approximations (ARC, M_4,3);
  Gross et al. derive lower bounds instead.
- **New:** the M_16,10 configuration, which ARC calls open ("not much closer", August 2026): F 2.63 [1.93, 3.39] on
  a sealed panel (lenient pass, strict fail; strict dead (verified) for this paradigm, H-S1). ARC's M_4,3 estimators
  are ~2x worse in RMSE (F ~4), which ARC calls "roughly match". Caveats: approximations validated by MC on
  development retrains (not strictly deductive); retrains, not noisy copies; private ARC work possible (H-E5).
- **Verdict:** known approach + novel aspect. **Confidence:** medium-low.
- **Phrase:** "A mechanism-based accuracy estimate in the style of ARC's M_4,3 estimators, applied to the M_16,10
  configuration. It is sampling-free at run time and has 2.6x the MSE of sampling (lenient pass, strict fail). To our
  knowledge it is the first published estimate of this kind for M_16,10-class models."

## Summary

| # | Result | Verdict | Confidence |
|---|---|---|---|
| 1 | Walk quadrature, H16 T5/T10 | known + novel aspect | medium |
| 2 | Great circles, 1-layer MLP | known, applied | medium-high |
| 3 | Deterministic rare-event walks to 1e-10 | known + novel aspect | medium |
| 4 | Rare-failure gradient, repair, conversion | known + novel aspect (conversion: likely new, low) | medium |
| 5 | Safe-use rating | known framing, applied | medium-high |
| 6 | Certified bound components | known + novel aspect (rigorous Gaussian polytope mass) | medium-low |
| 7 | Sampling-free M_16,10 accuracy estimate | known approach + novel aspect | medium-low |

## References
- [L1] Koutsourelakis, Pradlwarter, Schuëller 2004, Prob. Eng. Mech. 19(4):409-417. https://researchportal.hw.ac.uk/en/publications/reliability-of-structures-in-high-dimensions-part-i-algorithms-an/
- [L2] Nie, Ellingwood 2000, Struct. Safety 22:233-249, doi:10.1016/S0167-4730(00)00014-X; 2005, J. Eng. Mech. 131(3):259-267, doi:10.1061/(ASCE)0733-9399(2005)131:3(259)
- [L3] Griewank, Kuo, Leövey, Sloan 2018, J. Comput. Appl. Math. 344:259-274. https://arxiv.org/abs/1712.00920
- [L4] Brauchart, Saff, Sloan, Womersley 2014, Math. Comp. 83:2821-2851. https://arxiv.org/abs/1208.3267
- [N1] Sotoudeh, Thakur 2019, NeurIPS (ExactLine). https://arxiv.org/abs/1908.06214
- [N2] Sotoudeh, Thakur 2021, TACAS (SyReNN). https://arxiv.org/abs/2101.03263
- [A1] jamesrahenry 2026-08-18, Phase 1 technique census, ARC White-Box Estimation Challenge. https://discourse.aicrowd.com/t/18157
- [A2] AIcrowd submissions 325170 (2026-08-15) and 327651. https://discourse.aicrowd.com/uploads/short-url/bIyC1NJefuKuVEAZY8jk1PIKn8L.pdf , https://discourse.aicrowd.com/uploads/short-url/fKTnYwvOCAYWzpeh6w8uSUuKomJ.pdf
- [R1] Webb, Rainforth, Teh, Kumar 2019, ICLR. https://arxiv.org/abs/1811.07209
- [R2] Wu, Hilton 2025, ICLR. https://arxiv.org/abs/2410.13211
- [R3] Xu et al. 2024-09-13, ARC blog. https://www.alignment.org/blog/estimating-tail-risk-in-neural-networks/
- [R4] Fu, Hu 1997, *Conditional Monte Carlo: Gradient Estimation and Optimization Applications*, Kluwer. https://link.springer.com/book/9780792398738
- [S1] Uryasev 1994, J. Comput. Appl. Math. 56:197-223. https://pure.iiasa.ac.at/id/eprint/12769/
- [S2] Lu, Song, Yue, Wang 2008, Struct. Safety 30(6). https://pure.nwpu.edu.cn/en/publications/reliability-sensitivity-method-by-line-sampling
- [S3] Papaioannou, Breitung, Straub 2018, Struct. Safety 75:24-34. https://portal.fis.tum.de/en/publications/reliability-sensitivity-estimation-with-sequential-importance-sam/
- [P1] Sotoudeh, Thakur 2021, PLDI. https://arxiv.org/abs/2104.04413
- [P2] Usman, Gopinath, Sun, Noller, Păsăreanu 2021, CAV. https://arxiv.org/abs/2103.12535
- [P3] Sohn, Kang, Yoo 2023, ACM TOSEM 32(4), doi:10.1145/3563210
- [P4] Goldberger, Adi, Keshet, Katz 2020, LPAR-23. https://easychair.org/publications/paper/CWhF
- [P5] Meng, Bau, Andonian, Belinkov 2022, NeurIPS (ROME). https://arxiv.org/abs/2202.05262 ; Meng et al. 2023, ICLR (MEMIT). https://memit.baulab.info/
- [F1] Shinozuka, Feng, Lee, Naganuma 2000, J. Eng. Mech. 126(12):1224-1231. https://ascelibrary.com/doi/10.1061/(ASCE)0733-9399(2000)126:12(1224)
- [F2] Legay, Delahaye, Bensalem 2010, RV, LNCS 6418. https://arxiv.org/abs/1005.1327
- [V1] Boetius, Leue, Sutter 2025, ICML, PMLR 267:4660-4699. https://arxiv.org/abs/2405.17556
- [V2] Tran, Choi, Okamoto, Hoxha, Fainekos, Prokhorov 2023, HSCC, doi:10.1145/3575870.3587112
- [V3] Marzari, Corsi, Cicalese, Farinelli 2023, IJCAI. https://www.ijcai.org/proceedings/2023/25
- [V4] Baluta, Shen, Shinde, Meel, Saxena 2019, CCS (NPAQ). https://arxiv.org/abs/1906.10395
- [V5] Corsi, Marchesini, Farinelli 2021, UAI, PMLR 161:333-343 (ProVe). https://proceedings.mlr.press/v161/corsi21a.html
- [V6] Weng et al. 2019, ICML, PMLR 97:6727-6736 (PROVEN). https://proceedings.mlr.press/v97/weng19a.html
- [V7] Zhang, Wang, Kwiatkowska, Zhang 2025, JMLR 26 (PREMAP). https://arxiv.org/abs/2408.09262
- [V8] Kofnov, Kapla, Bartocci, Bura 2025, ICML. https://arxiv.org/abs/2502.11672
- [B1] Ehlers 2017 (Planet). https://arxiv.org/abs/1705.01320
- [B2] Bunel, Lu, Turkaslan, Torr, Kohli, Kumar 2020, JMLR 21(42). https://jmlr.org/papers/v21/19-468.html
- [B3] Zhang, Weng, Chen, Hsieh, Daniel 2018, NeurIPS (CROWN). https://arxiv.org/abs/1811.00866
- [B4] Wang, Zhang, Xu, Lin, Jana, Hsieh, Kolter 2021, NeurIPS (β-CROWN). https://arxiv.org/abs/2103.06624
- [B5] Neumaier, Shcherbina 2004, Math. Program. 99(2):283-296. https://optimization-online.org/2002/06/494/
- [B6] Baldoni, Berline, De Loera, Köppe, Vergne 2011, Math. Comp. 80:297-325. https://arxiv.org/abs/0809.2083
- [B7] Genz, Bretz 2009, Lecture Notes in Statistics 195, Springer, doi:10.1007/978-3-642-01689-9
- [M1] Hilton 2026-01-26, ARC blog (AlgZoo). https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/
- [M2] Christiano, Neyman, Xu 2022. https://arxiv.org/abs/2211.06738
- [M3] Hilton 2024-06-25. https://www.lesswrong.com/posts/SyeQjjBoEC48MvnQC
- [M4] Gross, Agrawal, Kwa, Ong, Yip, Gibson, Noubir, Chan 2024, NeurIPS. https://arxiv.org/abs/2406.11779
- [M5] Neyman et al. 2025-11-18, ARC blog. https://www.alignment.org/blog/competing-with-sampling/
