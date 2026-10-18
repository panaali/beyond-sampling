# Web search, ARC side (T22a)

↑ [lit/ index](../README.md) · [web log](../web_log.md) · [TASKS](../../TASKS.md) · [INDEX](../../INDEX.md)

**Question.** What has been published (ARC and third parties, up to 2026-10-17) that helps build a *mechanistic*
estimate of M_16,10's accuracy that matches sampling in MSE vs compute? What exactly is ARC's protocol?

**Method.** Web search plus direct reads: the alignment.org blog index, LessWrong GraphQL (post bodies with footnotes,
all comments, Jacob Hilton's 2026 comment history), fxtwitter (Hilton's launch thread and its replies), git
fetches of every public branch of `alignment-research-center/alg-zoo` and `mlp_cumulant_propagation`, all three alg-zoo
forks, the Google Drive notes linked from the AlgZoo post, ARC's May 2026 PDFs, and the AIcrowd challenge pages. The
GitHub REST API is blocked in this session (repo-scoped), so issues and PRs were read from the github.com HTML pages.
Every post-January-2026 ARC/AlgZoo source is logged in [../web_log.md](../web_log.md). I ran no estimator code. The
only computation was two accuracy checks (about 10 CPU-seconds), in [noise_check.py](noise_check.py).

**Status.** Done (2026-10-17).

## Headline

1. **ARC has published no solution, and in August 2026 said it was "not much closer".** Jacob Hilton, 2026-08-17,
   replying on the AlgZoo LessWrong post to "How is the progress six months later?":
   > "I don't recall seeing any specific effort on these models in the last six months outside of the comments on this
   > post. To me, that progress is fairly marginal and we are not much closer to succeeding at the challenge given at
   > the end of the post. (I'm excluding other research at ARC that doesn't pertain to these models specifically, but
   > may be on the path to success.)"

   This is the latest public statement. It is evidence against ledger row H-E5 ("someone at ARC has cracked M_16,10"),
   although it leaves room for private work after August or work "not pertaining to these models".
2. **The most detailed public ARC estimator is for M_4,3, not M_16,10.** It is in Hilton's notes "Heuristic
   explanations for 2nd argmax models" (dated December 21, 2024, linked from the AlgZoo post). It gives the exact
   protocol ARC used for its own claim that it "roughly match[es]" sampling: Gaussian weight noise with s.d. 0.05,
   256 noisy models, and compute counted in elementary operations. Both estimators actually *lose* to sampling by
   about 2x in RMSE (4x in compute), and ARC still describes that as "roughly match". See section 2.
3. **ARC's 2026 progress is on random instances, not trained models.** This covers wide random MLPs (cumulant
   propagation, arXiv 2605.05179), random halfspace intersections / #3-SAT / permanents (deduction-projection with
   "mechanistic L² sketching"), and the WhestBench contest on random MLPs. ARC calls random RNNs "much more difficult".
   The only 2026 commits to alg-zoo add MLP versions of the tasks (branch `mlp`, 2026-08-19), including a handcrafted
   k-th-argmax MLP. That suggests ARC is moving the AlgZoo tasks to MLPs, where its random-MLP machinery applies.
4. **Third-party attempts are qualitative.** Martin Keane's 49-document fork, Nicholas Goldowsky-Dill's notebook and
   Chamod Kalupahana's exploration produce no accuracy estimate that is competitive with sampling, and nothing on noisy
   copies.

## 1. Sources

All dates are publication dates. "Post-Jan" means the source is logged in [../web_log.md](../web_log.md).

| # | URL | Date | Author | What is relevant |
|---|---|---|---|---|
| S1 | https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/ (LW: https://www.lesswrong.com/posts/x8BbjZqooS4LFXS8Z) | 2026-01-26 | J. Hilton | Challenge statement and footnotes (quoted in section 4). M_2,2: 40 bits of surprise, exact. M_4,3: "roughly match the computational efficiency of sampling". M_16,10: features (running max, leave-one-out maxima of neurons 1, 2, 4, 6, 7), "incomplete", needs "feature analysis" combined with "brute force after exploiting symmetries ... in a primarily algorithmic way". |
| S2 | LW comments on S1 (GraphQL, 10 comments) | 2026-01-28 to 2026-08-17 | Goldowsky-Dill, Keane, Hilton, others | Goldowsky-Dill: neurons {1,6,7,8} ≈ a(δ)·M_t + b(δ)·max(S_t,0), R² 0.992. Keane: phase wheel, offset f(m)−f(s), gate sequence D[t] stable after an impulse, an effective linear operator per (gap, direction). Hilton (2026-02-04): "You'd need to do things like deduce correlations from the weights, rather than just observe them empirically". Hilton (2026-08-17): "not much closer". |
| S3 | https://x.com/JacobHHilton/status/2015856367429698013 (via api.fxtwitter.com/2/conversation) | 2026-01-26 | J. Hilton | Launch thread: "We propose a minimum bar to meet: a mechanistic estimate for the model's loss that is competitive with sampling (in MSE vs compute)." Credits Zihao Chen, George Robinson, David Matolcsi, Jacob Stavrianos, Jiawei Li and Michael Sklar for work on 2nd-argmax models. The replies hold nothing technical. |
| S4 | https://drive.google.com/file/d/1t2rXhDQqWrOXRwjqstbrBPWiCmYlMts_/view | 2024-12-21 (linked 2026-01) | J. Hilton | **The key methods document.** Two mechanistic estimators for M_4,3 with error and compute numbers. Details in section 2. |
| S5 | https://drive.google.com/file/d/1qH61_WshTrgYh0YLj5vkONzk8x00zImN/view | 2024-12 | G. Robinson | "Symmetric RNNs": the approximate symmetry W_hh ≈ Uᵀ W'_hh U (U = [[1,0,0,−1],[0,1,−1,0]]) reduces a width-4, length-3 RNN to a 3-2-3 network with absolute-value activations. Accuracy is then read off per sign quadrant in closed form. |
| S6 | https://drive.google.com/file/d/1USCD1LPa6hXMTj7PAO6Jmc4dVdxemZY3/view (+ colab) | 2024-04 | Zihao Chen | Original analysis of the example model. **It was trained on Unif[−1,1]**: "a 16-neuron RNN that achieves 94% accuracy on the uniform distribution" (hence the file name `one_layer_16_hidden_94_acc_2nd_argmax.pth`). N2 = running max, N4 = ReLU(x_t − max), N7 = ReLU(max − x_t), N8 = ReLU(2nd max − x_t). A distilled algorithm reaches 99.8%. Proposes "bin propagation" and DFA extraction as untried estimator routes. |
| S7 | https://www.alignment.org/blog/competing-with-sampling/ (LW XdQd9gELHakd5pzJA) | 2025-11-18 | Neyman, Lecomte, Wu, Winer, Hilton, Robinson | Formal matching sampling principle (MSP). Defines "mechanistic" by the heuristic "avoids any random or pseudorandom sampling". Random RNNs are "much more difficult than the MLP setting, because of the weight sharing". |
| S8 | https://www.alignment.org/blog/mechanistic-estimation-for-wide-random-mlps/ ; arXiv 2605.05179 ; https://github.com/alignment-research-center/mlp_cumulant_propagation | 2026-05-06/07 | Wu, Lecomte, Winer, Robinson, Hilton, Christiano | Cumulant propagation ("kprop") for random ReLU MLPs: MSE O(ε²) at O(n/ε²) FLOPs vs sampling's O(n²/ε²). Beats MC "at FLOP budgets spanning 7 orders of magnitude" (width 256, depth 4), but "the dependence of our algorithms on depth is worse" and the methods "rely on the networks being randomly initialized". |
| S9 | `mlp_cumulant_propagation` branches `experiment/max-endpoint`, `max`, `edgeworth-terms` (git) | 2026-07-12 to 2026-08-04 | Kunal Chawla, Max Misterka, Wilson Wu, Kiril Bangachev | **Argmax endpoint**: q_i = P(i = argmax_j Y_j) computed as ∂/∂a_i E[max_j(Y_j + a_j)] from an Edgeworth expansion of propagated cumulants. Unbiased Brier-MSE U-statistics. FLOP rules: "MC FLOPs = m * (one instrumented MLP forward + max reduction ...); random-number generation is excluded"; "Reference (ground-truth) computation is evaluation overhead and is never charged to any estimator." |
| S10 | https://www.alignment.org/blog/mechanistic-estimation-for-expectations-of-random-products/ | 2026-05-15 | Hilton, Robinson, Neyman, Christiano, Winer, Lecomte, W. Wu, G. Wu | Deduction-projection for products F(c_1,x)⋯F(c_k,x). Projection = "mechanistic L² sketching" onto the top eigenfunctions of E_f[f fᵀ] for the *remaining random factors*. They "roughly match or outperform random sampling up to logarithmic factors". |
| S11 | https://www.alignment.org/content/files/2026/05/deductive_estimation_for_random_halfspace_intersections.pdf | written 2025-05-02, posted 2026-05 | J. Hilton | **Directly relevant geometry**: P_x(x·v_i ≥ 0 ∀i) (cones through the origin, the same objects as M_16,10's linear regions). Propagates a degree-D polynomial density on the sphere, one halfspace at a time. Each step projects onto harmonic polynomials of degree ≤ D using the Funk-Hecke / Gegenbauer formula, applied via Givens rotations. Runtime O(k(log 1/ε)²/ε²), matching MC up to logs, for *random* v_i. The code lives in the private repo `alignment-research-center/deduct-est` (`halfspaces/harmonic_prop.py`). |
| S12 | https://www.alignment.org/blog/announcing-the-arc-white-box-estimation-challenge/ ; https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026 ; https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197 | 2026-06-02 to 2026-10 | Hilton, Christiano, W. Wu; AIcrowd | WhestBench: estimate E[M_θ(X)] for random MLPs under a FLOP budget measured by `flopscope` (Phase 2: width 1024, depth 16, 2^41 FLOPs per MLP; closes 2026-10-17). Hilton, 2026-06-23: warm-up winners were "variants on the factorized 3rd cumulant propagation ... combined with learned networks that consume the cumulant estimates as features". AIcrowd: "The best-performing methods in Phase 1 tended to be based on variations of sampling." |
| S13 | https://www.alignment.org/blog/a-mikes-eye-view-of-arcs-research/ | 2026-06-09 | M. Winer | Agenda overview. No AlgZoo content. "we don't have a single non-trivial problem of this type we fully understand". |
| S14 | https://github.com/alignment-research-center/alg-zoo (git: `main` 4433f50 2026-01-19 = our local clone; `mlp` 6e92689 2026-08-19); issues and PRs: 0 open, 0 closed; no Discussions | 2025-12 to 2026-08 | J. Hilton | **New since our clone: branch `mlp`**, with `MLP`, `DistMLP`, `ScalarMLP`, `load_dist_mlp`, `train(..., n_mlp_hidden_layers=...)` and `handcrafted_kth_argmax_mlp(seq_len, k)`: hidden width n(n−1), one ReLU unit per ordered pair ReLU(x_i − x_j), and output weights −(k−1) and −(n−k). Nothing changes for the RNN models. |
| S15 | https://github.com/mrsirrisrm/alg-zoo (fork; 2026-02-01 to 2026-03-24; also gist d6850ff8647d1ed2f67cc92d5bce3ed0) | 2026 | Martin Keane | 49 docs and many scripts: clipping, phase wheel, tropical geometry, Koopman lifting, Schur delay line, a surprise-accounting attempt. Doc 42 "Mechanistic Accuracy Estimate" claims "100%", but only for the 90 two-impulse (M, S position) pairs with all other inputs zero. **It is not an accuracy estimate of the model.** |
| S16 | https://gist.github.com/NixGD/2651be79bfd22bcb13d1f609f23439a4 | 2026-01-28 | N. Goldowsky-Dill | Fits for neurons N1, N6, N7, N8 (Unif[−1,1] inputs). Transition matrix W_sub (4×5) maps the δ-coefficients from one step to the next; the "S < x_t < M" case is marked TODO. |
| S17 | https://github.com/ChamodKalupahana/alg-zoo, https://github.com/poonai/alg-zoo (forks) | 2026-02 | third parties | Small explorations (median M_2,2 handcrafted; a manual RNN class). Nothing on estimation. |
| S18 | https://arxiv.org/abs/2211.06738 (Formalizing the presumption of independence, App. D cumulant propagation); https://www.alignment.org/blog/estimating-tail-risk-in-neural-networks/ ; G. Wu thesis https://gabrieldwu.com/assets/thesis.pdf | 2022-2025 | ARC | Background for S7-S11 (pre-2026, not logged). |

Not found: any talk, podcast or paper by ARC staff describing an M_16,10 estimator; any arXiv paper citing AlgZoo (the
Semantic Scholar API was rate-limited; web search found none); any MATS or other third-party estimator for AlgZoo.

## 2. What ARC has actually done (most concrete reconstruction)

**M_2,2 (10 parameters, error rate about 1/13,000).** Exhaustive linear-region analysis. The error is "the fraction of
the unit disk lying between the model's decision boundary and the line x_0 = x_1", computed in "a handful of
computational operations". It is "brute force": "we essentially checked each linear region of Δ one by one".

**M_4,3 (32 parameters, 98.5%). Two estimators (S4):**

- *Original version.* The sign pattern of W^hi splits x_0 ≥ 0 from x_0 ≤ 0. Approximate near-symmetric coefficient
  pairs by their average (a_3 ≈ −a_0 and so on), which gives 4 regions for h_2. Split by the 6 orderings of
  (x_0, x_1, x_2), giving "14 likely subregions plus 4 unlikely subregions". In each likely subregion, apply a
  presumption of independence to the last ReLU: replace each gate 1[z_3,j > 0] by a random variable independent of z_3
  but correlated with the other gates through a latent Z ~ Bernoulli(q), "where q = 0.2 is a hyperparameter", matching
  the gate's mean and variance. The accuracy of each subregion is then the probability that the right logit is
  largest. Unlikely subregions are given accuracy 1/3. All volumes are solid angles of intersections of halfspaces
  through the origin, computed with "the polygonal generalization of Girard's theorem": 18 volumes with 4 halfspaces,
  56 with 5 and 28 with 6.
- *George's version.* Write the weights as a symmetric reduced model plus an error ε (W_hh = Uᵀ W'_hh U + ε_hh and so
  on). Propagate the reduced model exactly. Propagate the mean and covariance of the error terms with "a version of
  covariance propagation" under a presumption of independence between ε_h and h'. The resulting 24 six-dimensional
  volumes are estimated by sampling: "We avoid the problem of figuring out how to do this exactly for now and just use
  sampling."
- *Evaluation, quoted:* "we added zero-mean Gaussian noise with a standard deviation of 0.05 to the weights of network
  and ran the two estimators on 256 such models. The original version achieves a RMS error of around 3%, and the
  version based on George's approach achieves a RMS error of around 5%, although it appears to have lower bias."
- *Compute, quoted:* "In 3 dimensions, let us crudely estimate that it takes around 5k³ operations to do this for k
  half-spaces ... the original version takes around 5(18·4³ + 56·5³ + 28·6³) = 71000 operations ... estimating the
  accuracy using k samples takes around (3·4·(4+1) + 4·3)k = 72k operations ... sampling should achieve an RMS error of
  around 0.8% and 2.3% respectively. So we are underperforming sampling by around a factor of 2 in terms of error (i.e,
  a factor of 4 in terms of compute). But the half-space algorithm could almost certainly be done more efficiently."
- *Surprise accounting:* "This all seems confusing enough that we should perhaps just stick to quality versus compute
  for evaluating explanations".

**M_16,10 (432 parameters, 95.3%).** Only the feature analysis of S1: neuron 2 ≈ max(0, x_0..x_{t−2}), neuron 4 ≈ the
increment of the running max, neurons 6 and 7 ≈ leave-one-out maxima minus x_{t−1}, neuron 1 ≈ another
leave-one-out maximum. This led to a 22-neuron handcrafted model with 99% accuracy (`handcrafted_2nd_argmax(10)`). The
stated gaps: "most of the remaining 11 hidden neurons are more densely connected"; "we have not produced a
correspondence between most of hidden neurons of the trained model and the hidden neurons of the handcrafted model";
"we have not dealt with the approximation error". ARC's own plan, quoted: "combine this kind of feature analysis with
elements of the 'brute force after exploiting symmetries' approach ... in a primarily algorithmic way". No number, no
estimator and no noise level has been published for M_16,10.

**What ARC's general toolkit is (S7-S11), and where it stands on our problem.** Their working mechanistic estimators
are deduction-projection estimators: propagate a representation of the input distribution layer by layer, or factor
by factor, and after each step project it onto a tractable family. The families are Gaussian (covariance
propagation), cumulants up to order K with Hermite/harmonic tensor decompositions (kprop), or low-degree harmonic
polynomials on the sphere (halfspace estimation). The projection is chosen to preserve inner products with the
*remaining* random computation (the sketching theorem). Every proof and every success is for **random** weights.
Trained structure is supposed to enter as "advice" π, and the advice mechanism is "not yet working" (S8). Their own
contest results (S12) show that at practical sizes, sampling variants or kprop combined with learned correctors win.
In short, ARC's public toolkit is not yet sufficient for M_16,10, which matches the August 2026 comment.

## 3. Ideas for our estimator (added to TASKS.md Backlog as T36-T41)

| ID | Goal | Done when | Pointers |
|---|---|---|---|
| T36 | **Reproduce ARC's two M_4,3 estimators** (Girard-volume subregions with a latent-Bernoulli gate correlation q = 0.2, and symmetry reduction with ε covariance propagation) on `zoo_2nd_argmax(4,3)`. This calibrates what ARC itself calls "roughly match", and checks our op accounting against theirs (72 ops per sample). | We obtain RMS errors close to 3% and 5% at absolute σ = 0.05 over 256 noisy models, with the ARC-style op counts, and we report our best M_4,3 deductive method under the same protocol. | S4, S5; [../web_arc/README.md](README.md) section 2 |
| T37 | **Reference model plus error propagation for M_16,10** ("George's approach" scaled up). Build a reference RNN from our mechanism (FINDINGS) or `handcrafted_2nd_argmax(10)` whose accuracy and region structure are computable deductively. Write trained = reference + ε, and propagate the mean and covariance of the ε terms through the 10 steps under a presumption of independence, as in S4 (this is not E13's propagation of the full distribution). | We measure bias and RMSE on the leaderboard models (original, seeds 0-4, noisy copies). We also measure the ceiling: the error of the reference-only estimate before any ε correction. | S4 ("Version based on George's approach"), S1 (22-neuron handcrafted model), [m16/FINDINGS.md](../../m16/FINDINGS.md) |
| T38 | **Harmonic-polynomial deduction-projection on S⁹** (Hilton's halfspace algorithm adapted to the RNN). The model has no biases, so accuracy is a spherical measure. Within a gate history, each new gate and each logit comparison is a halfspace through the origin. Propagate g̃ (degree ≤ D on S⁹) through the halfspaces step by step, projecting by Gegenbauer kernels (Funk-Hecke). Or, following the sketching theorem, choose the projection basis from the top eigenfunctions of E_f[f fᵀ] over an ensemble of "future" functions (for example the noisy-copy ensemble), so the sketch keeps exactly the inner products later steps need. Start at T = 4-5 (the T10 minimal pair). | We have a bias/RMSE-vs-FLOPs curve at T = 4, 5 and 6 (and 10 if it holds) against sampling. A negative result counts if it shows which degree D the rare wide bands (H-E12) need. | S11 (algorithm + runtime), S10 (sketching), ledger H-E11/H-E12, T10 |
| T39 | **Independent check of our mechanism claims against Martin Keane's analysis** (rule 4 needs independent checks). Compare his phase-wheel period (complex W_hh eigenvalue, "period = 5.4 steps"), his forward/reverse offset f(m)+g(s) with g = −f, and his D[t] stability after an impulse with our H-M claims (42.7 deg/step phase, about 8.4 steps). | A table lists each H-M claim as agreeing, conflicting or not covered, with the discrepancy (5.4 vs 8.4 steps) explained. | S15 (docs 32-49, `src/`), S2, S16 |
| T40 | **Argmax endpoint from cumulants (ARC max-endpoint branch).** q_i = ∂_a E[max_j(l_j + a_j)] via an Edgeworth expansion gives smooth argmax probabilities from propagated cumulants. Test it as the final-step endpoint only, on top of exact gate enumeration for the early steps, where E13's cumulants were reasonable. Low priority: E13 shows the full-depth cumulants are far from Gaussian. | We measure the bias of the endpoint alone at T = 3-5, given exact cumulants of the final logits, jointly with the target index. | S9 (`src/mlp_kprop/max_endpoint/argmax.py`), E13 |
| T41 | **Ask ARC for the protocol and whether a solution exists.** Use a comment on the LW AlgZoo post, or contact Hilton: noise σ for M_16,10, FLOPs or forward passes, whether fixed QMC nets count as "mechanistic", and whether advice may be computed per noisy copy. Report our negative map (H-E11/H-E12). | We have an answer (or no answer after 2 weeks), recorded in H-E4 and H-E5. *Blocked: needs user approval* for external communication. | S1 footnotes, S2, S4; ledger H-E4/H-E5; T9 |

T40 extends T35 (ARC's cumulant method, already reserved by another agent): T35 implements kprop, and T40 adds only the argmax endpoint.

## 4. Points that contradict or refine the project's assumptions

1. **The noise level for M_16,10 is unspecified, and ARC's only worked example used absolute σ = 0.05.** The challenge
   text, quoted: "A cheap way to measure mean squared error is to add noise to the model's weights (enough to
   significantly alter the model's accuracy) and check the squared error of the method on average over the choice of
   noisy model." Footnote 9: "A better but more expensive way to measure mean squared error is to instead average over
   random seeds used to train the model." Footnote 3: "the method should match sampling on average over random seeds."
   Footnote 8: models from `zoo_2nd_argmax(16, 10, seed=0..4)` are ones "which a good method should also work well
   on". For M_4,3 ARC used absolute σ = 0.05 (S4). M_16,10 is far more sensitive, so σ = 0.05 does not transfer. I
   measured (16 models × 100k Gaussian inputs each): absolute σ = 0.01 gives mean accuracy 0.62 (s.d. 0.095);
   absolute σ = 0.05 gives 0.19, near chance. Our relative levels (s.d. = σ × matrix RMS; RMS = 6.44, 0.56 and 2.35 for
   W_hi, W_hh, W_oh) give: 0.0025 → 0.931 ± 0.008; 0.005 → 0.881 ± 0.023; 0.01 → 0.747 ± 0.053; 0.02 → 0.551 ± 0.068
   (12 models each). All of these "significantly alter" the accuracy, so ARC's wording is satisfied. The seed-averaged
   protocol (footnote 9) is the one ARC calls "better", which supports giving the seeds 0-4 column weight in the
   leaderboard.
   Reproduce: `source .venv/bin/activate && python lit/web_arc/noise_check.py` (about 10 CPU-seconds; prints every
   number in this point and in point 7).
2. **ARC's own bar for "roughly match" is looser than ours.** In S4, ARC's M_4,3 estimators are 2x worse in RMSE (4x
   in compute) and S1 still says "We find that we are able to roughly match the computational efficiency of
   sampling". Its George-style estimator also used *sampling* for a sub-integral. Our success criterion 3 ("MSE at most
   that of sampling") is stricter than ARC's self-assessment. That is fine as a target, but our "every pure calculation
   loses by 8x to 300x" should be read against ARC's own 4x. If the 8-300x figures are compute (MSE) ratios, then our
   deductive methods are roughly 2-75x behind ARC's M_4,3-level bar, not 8-300x.
3. **The estimand for the sampling benchmark (T22).** The MSP (S7) requires
   E_c[(G − E_x M)²] ≤ ε² E_c[Var_x M], and the halfspace note (S11) states the baseline as "a mean squared error of
   E[p(1−p)]ε²". So ARC's benchmark is the **mean of the per-model variances p_i(1−p_i)** divided by the budget, not
   mean(p)(1 − mean(p)). This agrees with how T22 was resolved (v3 baseline = mean of p_m(1−p_m)/cost; commit 65db758). It now has ARC's own
   wording as support.
4. **Compute accounting.** ARC's accounting is in elementary operations or FLOPs, never wall-clock. In S4,
   sampling = (ops of one forward pass) × k. In S8/S9, "MC FLOPs = m * (one instrumented MLP forward + max reduction
   ...); random-number generation is excluded", and ground-truth computation "is never charged to any estimator". In
   the WhestBench contest, FLOPs are counted by `flopscope` with symmetric-kernel adjustments, and Phase 2 forbids
   compiled kernels. Our leaderboard's forward-pass-equivalent units are consistent with this if (a) estimator
   overheads are converted to FLOPs honestly (T20) and (b) RNG cost is excluded for sampling, as ARC does.
5. **"Mechanistic" is defined by a heuristic, quoted (S7):** "A simple, though not entirely correct, heuristic for
   whether an estimation algorithm is deductive, is whether it avoids any random or pseudorandom sampling." On this
   reading, our deterministic quadratures that evaluate the model at fixed Sobol / QMC nodes (sci_feynman, sci_tao,
   tube) are *pseudorandom sampling in spirit*. ARC would probably not count them as mechanistic even though they use
   no RNG. Our "deductive" label for them should be qualified.
6. **ARC allows advice; our criterion 2 forbids per-model fitted tables.** The MSP allows an explanation π of size
   O(|θ|), produced in the train-and-explain variant by an algorithm with Time(E) ≤ O(Time(T)). ARC's own M_4,3
   estimators hard-code model-specific structure (which coefficient pairs to average, the hyperparameter q = 0.2) and
   reapply it to noisy copies. So per-model-*family* hand analysis is within ARC's practice. Per-*copy* fitted tables
   are a grey area that only T37 can settle.
7. **Input distribution.** ARC's 95.3% is Gaussian accuracy. Measured with 400k inputs: N(0,1) 0.9531 and Unif[−1,1]
   0.9394. The file name's "94_acc" refers to the uniform distribution the model was trained on (Zihao Chen, S6).
   Both the 2026 alg-zoo code (`th.randn`) and ARC's random-instance work use Gaussian inputs. Our assumption of
   x ~ N(0, I_10) is correct, but the example model was *not* trained on it, unlike `zoo_2nd_argmax(16,10,seed)`.
   That may be one reason the seed models behave differently.
8. **H-E5.** Hilton's 2026-08-17 comment is the strongest public evidence on whether ARC has a solution. It says no,
   for these models specifically, as of mid-August 2026. The private claim relayed to the user may refer to later work
   or to work on other problems.

## Claims

- [measured] alg-zoo has two public branches. `mlp` (2026-08-19) is newer than our clone and only adds MLP
  architectures and a handcrafted k-th-argmax MLP. There are 0 issues and 0 PRs. There are three forks; only
  mrsirrisrm's contains substantial analysis.
- [measured] Accuracy of the example model: 0.9531 under Gaussian inputs vs 0.9394 under Unif[−1,1] (400k each). Noisy
  accuracies as in section 4, point 1. Command: `python lit/web_arc/noise_check.py`.
- [measured, quoted] ARC's M_4,3 protocol and numbers in section 2. Hilton's "not much closer" comment of 2026-08-17.
- [inferred] ARC's public methods do not yet handle M_16,10. Their successes are on random instances, and their own
  contest is won by sampling variants.
- [speculative] The `mlp` branch indicates ARC moving AlgZoo tasks toward MLPs, where kprop applies.

## Alternatives not ruled out

- ARC may hold unpublished internal results (the deduct-est repository is private). Hilton's comment excludes "other
  research at ARC".
- AF-only comments, X replies outside the launch thread, and talks without transcripts were not covered. The
  Semantic Scholar citation search failed (HTTP 429).
- The Google Drive notes may have newer versions than the PDFs fetched here. The PDF creation dates are 2024-2025.

## Failure modes of this review

Web summaries were paraphrased by a fetch model. Every protocol quote above was checked against raw text: the LW
GraphQL markdown/HTML, pdftotext output of the PDFs, or git objects. The AIcrowd and WhestBench details come only from
fetched summaries and are less certain.

## Compute accounting

Web and git reads only. Two accuracy checks of about 2.8M model evaluations in total (CPU seconds). No estimator was run.
