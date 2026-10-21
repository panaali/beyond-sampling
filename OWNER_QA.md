# Owner questions and answers (DRI session, 2026-10-17/18)

This file collects every substantive question the project owner asked the DRI between 2026-10-17 22:55 and 2026-10-18 13:24 UTC, with the DRI's answer to each. It was built from the session transcript. The owner's words are quoted verbatim, with light typo fixes only. The answers are condensed paraphrases, not quotes, but they keep every number and caveat. **Each answer is as of its timestamp**, and most results quoted in it were unverified when given. Where a later result in the same session, or the registry in [INDEX.md](INDEX.md) (shared branch `ccr-6a61f25c-wn80ii`, commits `60ace4bd`/`f2b23d41`, 13:14–13:24 UTC), changed an answer, a **Later updates** line follows it, naming the INDEX rows that hold the current numbers; the update text itself is kept verbatim in [docs/archive/OWNER_QA_later_updates.md](docs/archive/OWNER_QA_later_updates.md). Task IDs (T-numbers) point to rows in INDEX.md and [TASKS.md](TASKS.md). The file is organised by topic (sections 1-6); the owner's directives are collected word for word, in time order, in [section 8](#directives). **This is the only place directives and owner Q&A are recorded**; other files link here, and the rules that bind agents are one-liners in [AGENTS.md "Standing rules"](AGENTS.md#standing-rules).

**Terms used below**

- **M_16,10** is ARC's challenge network: 16 ReLU neurons, 10 inputs, 432 parameters, no biases. Its task is to output the position of the 2nd-largest input. Its true accuracy is 95.31175% ± 0.00004 pp (T192, 300 billion samples). **(H,T)** or **M_H,T** means width H and list length T.
- **pp** means percentage points. **FPE** means forward-pass equivalents, also called "model runs".
- **F** = mean squared error × compute ÷ (variance of plain sampling at equal compute). Plain sampling scores F = 1, and F = 5 means sampling has 5× lower squared error at the same compute.
  - **Strict bar:** F ≤ 1 (ARC's "match").
  - **Lenient bar:** F ≤ 10, and the upper end of the 95% CI must also be ≤ 10 for a "candidate". This is a project convention (Directive 8), taken from the 2.4–10× of ARC's own estimators on M_4,3. It is not ARC's rule.
- **HP1–HP5** are sealed held-out panels, scored only by the scorer session (T281). **DEV2 and DEV3** are open development panels of retrained models.
- **Deductive vs inductive** is Hilton's distinction. Running constructed inputs through the network is allowed; extrapolating from observed behaviour, or fitting coefficients to samples, is not.

## Contents

**1. The challenge and ARC's bar**
- [1.1 Is there any hope for the methods we are testing? (00:18)](#q1-1)
- [1.2 Owner report: a new ARC member has a solution (00:39)](#q1-2)
- [1.3 Could we prove infeasibility for random weights? (03:17)](#q1-3)
- [1.4 Would a 100%-accurate network be a solution ARC accepts? (03:35)](#q1-4)
- [1.5 If ARC only wanted two steps, have we solved it? (05:06)](#q1-5)
- [1.6 If ARC asked for 16 neurons and 7 steps, would we win? (05:13)](#q1-6)
- [1.7 The Hilton talk transcript (05:42, 05:44)](#q1-7)

**2. The network, its neurons, and the owner's intuition**
- [2.1 Maximum accuracy of this architecture (03:22)](#q2-1)
- [2.2 Accuracy for 2 to 10 numbers (03:32)](#q2-2)
- [2.3 Did we find what each of the 16 neurons does? The owner's intuition (03:41)](#q2-3)
- [2.4 Experiments to prove or disprove the intuition (03:50)](#q2-4)
- [2.5 Experiments to find each neuron's role quantitatively (03:52)](#q2-5)
- [2.6 The right intuition, first version (04:04)](#q2-6)
- [2.7 The right intuition, in simple terms (06:18)](#q2-7)
- See also [4.1](#q4-1) and [4.2](#q4-2) on why the neuron roles did not help the estimator.

**3. Our methods and results (strict and lenient bars, approximations, noise)**
- [3.1 "Better than sampling only on the clean model" (04:38)](#q3-1)
- [3.2 The fixed set of test points (04:39)](#q3-2)
- [3.3 Our solution for 4 numbers at 16 neurons (05:08)](#q3-3)
- [3.4 "This solution doesn't look like an understanding" (05:09)](#q3-4)
- [3.5 A formula over the top values and their gaps (05:11)](#q3-5)
- [3.6 The mechanism-based method we hoped would pass the lenient bar (05:22)](#q3-6)
- [3.7 Where is our hope now? (05:35)](#q3-7)
- [3.8 Do we pass the lenient bar? (12:22)](#q3-8)
- [3.9 What are the "80 approximations"? Would ARC accept them? (12:28)](#q3-9)
- [3.10 Would 5× more compute match sampling? ARC's M_4,3 factor (12:36)](#q3-10)
- [3.11 Did our reproduction of ARC use noisy copies? How much noise breaks our models? (12:41)](#q3-11)
- [3.12 How much noise breaks our solution? (12:44)](#q3-12)

**4. ARC's long-term goal (rare events, what mechanistic approaches answer better)**
- [4.1 ARC's goal, our mechanistic progress, and whether we used the neuron roles (12:44)](#q4-1)
- [4.2 Why the neuron information was not that helpful (13:00)](#q4-2)
- [4.3 Are we closer to estimating rare bad behaviour? (12:54)](#q4-3)
- [4.4 Questions our approach can answer better than sampling (13:06)](#q4-4)
- [4.5 When will T340 be resolved? (13:12)](#q4-5)
- [4.6 Rare events: the first results (13:43)](#q4-6)
- [4.7 Is the walk method that ties random testing at 10 steps acceptable to ARC? (10-19 05:2x)](#q4-7)
- [4.8 Do we have several solutions for the lenient bar and the main challenge? (10-19 06:2x)](#q4-8)

**5. Compute, GPU and process**
- [5.1 Which task needs a GPU? (2026-10-17 23:01)](#q5-1)
- [5.2 Questions for the previous DRI (23:12, 00:31)](#q5-2)
- [5.3 What the Codex managers wrote in DRI_HANDOFF.md (00:26)](#q5-3)
- [5.4 Why does Google reclaim the Colab node? (03:28)](#q5-4)
- [5.5 When will all these experiments be done? (04:46)](#q5-5)
- [5.6 Why is the training-ceiling study slow? Better GPU, CUDA or C? (04:48)](#q5-6)
- [5.7 Can one fused kernel do data generation, forward and backward? (04:53)](#q5-7)
- [5.8 Could one fused kernel run all the experiments? (04:54)](#q5-8)
- [5.9 Have we published all results, and do the reviewers have them? (05:04)](#q5-9)

**6. Architecture and the exact route**
- [6.1 Would a different architecture (all 10 inputs at once, like a Transformer) make it easier? (13:19)](#q6-1)
- [6.2 Why are exact and region-counting methods dead for this problem? (13:24)](#q6-2)
- [6.3 How much compute would exact enumeration of M_16,10 need? (13:31)](#q6-3)
- [6.4 GPU brute-force counts: how long would they take? (13:36)](#q6-4)
- [6.5 Would quantising the model let us solve the challenge? (13:50)](#q6-5)

**7. [Status timeline](#status)**

**8. [Owner directives (verbatim)](#directives)**

---

## 1. The challenge and ARC's bar

<a id="q1-1"></a>
### 1.1 Is there any hope for the methods we are testing?

**Q** (2026-10-18 00:18 UTC): "Is there any hope for any of the methods that we are testing for solving the challenge?"

**A:** Yes, some, but not equally for both routes (all odds are the DRI's judgment).
- **The rule cuts both ways.** ARC asks for an estimate "at least as precise as random sampling for the same compute". Sampling always has some error, so an exact answer beats it at whatever cost, as long as it can finish. ARC's notes say M_16,10 needs "brute force after exploiting symmetries … in a primarily algorithmic way", and their only working estimator (on the 32-parameter model) checked each linear region one by one.
- **Route 1: exact structured enumeration (T192, T5). Odds about 1 in 3.**
  - At (16,4) the recipe's cells roughly equal the linear regions, at 0.7× the cost of exact enumeration.
  - The old projection spans 1e17–1e23 operations.
  - The low end is plausible because the network has an estimated 1e7–1e8 regions (2.7e4 at (16,4)). Growth would have to slow from 34×/step to 3–4×/step.
  - A lead: the recipe starts from 4·T! ordering cells (about 14.5M at T=10), while the answer only needs 90 cones.
- **Route 2: a cheap mechanism-based estimator. Odds about 1 in 10.**
  - About 30 cheap deterministic methods lose to sampling by 37× or more.
  - We understand about 80% of the mechanism, but the errors sit in rare, irregular near-tie bands.
  - Live bets: error-budget pruning (T195; 80% of splits change the result by < 1e-6, but the pruned cells may be exactly the error bands) and order statistics plus a bound (T189).

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u1-1). Current numbers and status: [INDEX](INDEX.md) rows T192, H-E13, T270, T316, T195, T317c, T317e, HP3, H-L1, T317f, HP4.

<a id="q1-2"></a>
### 1.2 Owner report: a new ARC member has a solution

**Q** (2026-10-18 00:39 UTC), an owner statement: "Regarding that ARC that has solved this problem, I think the person who has solved it is a new joiner to ARC, and they have tried new things. I'm not sure exactly what they did, but the ARC contact told me that they have a solution and they're waiting for us for comparison."

**A:** This changes the assessment. A solution reported by a newly joined member is plausible even though we cannot verify it. Hilton's public "not much closer" (August 17) would predate their work. The ledger now says "reported, unverified" (H-E5), and a comparison-prep task was added (T203).
- **What it means scientifically:**
  - The problem is probably solvable, which keeps the cheap bets (T195, T189) alive alongside the exact route.
  - Their method could be quite different from ours: they "tried new things".
- **How to make the comparison worth having:**
  1. Freeze our results before learning anything about their method, so our results act as an independent check.
  2. Score both sides identically: same models, noise copies, compute accounting and budgets.
  3. Offer an exact answer key if T192 finds one affordable.
- **What to ask ARC about the protocol:** the noise levels and models in their panel, how they count compute and at which budgets, whether their method uses any sampling, and the expected comparison format and timeline. A draft message was offered for the owner to send.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u1-2). Current numbers and status: [INDEX](INDEX.md) rows T41, T203.

<a id="q1-3"></a>
### 1.3 Could we prove infeasibility for random weights?

**Q** (2026-10-18 03:17 UTC): "If someone resets all the weights to random, can we theoretically prove that it would not be feasible to have an answer to this problem or not?"

**A:** No, not as a proof.
- **What can be proven.** A deterministic method that only evaluates the network on inputs it chooses, without reading the weights, cannot beat random sampling in general (Elekes; Bárány–Füredi): an adversary can hide the answer between the chosen points. For methods that read the weights, which is what ARC wants, a proof either way would settle open problems. Deterministic estimation of a circuit's acceptance rate (CAPP) is a known hard problem, and Ryan Williams showed that even modest algorithms for it would imply new circuit lower bounds.
- **Two loopholes block a clean statement.**
  1. Hard-coding: for one fixed network, a method that prints the right answer exists, so impossibility can only be stated for families of networks.
  2. "Mechanistic" and "pseudorandom" are not formally defined, and pseudorandom derandomisation is believed possible.
- **Expectation for random weights (judgment, not proof).** Accuracy would be near chance and rough averages easier, but beating sampling harder. Random networks should have near-maximal region counts, while trained ones reuse gate patterns. The closest theory is the no-coincidence conjecture (Christiano, Neyman, Xu), which is still a conjecture.
- **Proposal:** measure the contrast with random-weight versions of M_16,10. The owner approved it, and it ran as T232.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u1-3). Current numbers and status: [INDEX](INDEX.md) rows T232, T271, T280, H-E14, E35.

<a id="q1-4"></a>
### 1.4 Would a 100%-accurate network be a solution ARC accepts?

**Q** (2026-10-18 03:35 UTC): "If the accuracy is 100% for a network, do you think we have a solution that Arc would accept?"

**A:** No, for three reasons.
1. **It would be a different network.** The challenge is to estimate M_16,10's accuracy (95.31%), not to build a better network.
2. **At 100% the challenge is trivial.** Sampling would also be exactly right, so a calculation could only tie. The challenge is meaningful only for accuracies strictly between 0% and 100%.
3. **ARC also tests noisy copies.** Noise lowers M_16,10's accuracy to roughly 45–94%, so even a perfect network would stop being exact once noised.

100% is probably not reachable with this design: trained networks top out around 99.95% even for 3 numbers. The idea does connect to understanding. A network provably 100% correct would be strong understanding, and "ideal network plus deviation" was already tried (T37, a handcrafted reference plus residual). It missed by 22–48 pp, because the deviation is the messy part.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u1-4). Current numbers and status: [INDEX](INDEX.md) rows T234.

<a id="q1-5"></a>
### 1.5 If ARC only wanted two steps, have we solved it?

**Q** (2026-10-18 05:06 UTC): "Does this mean that for two steps we are understanding what the network does, and if what ARC wanted was just two steps, then we have solved the challenge?"

**A:** No. In T241's test, the formulas replaced the state after steps 1–2 (100% agreement), but the real network still computed the hard last steps where near-ties are decided. After 2 steps there are only 34 linear pieces.
- **Short networks as challenges of their own:**
  - 2 numbers: about 99.99% accuracy, essentially trivial.
  - 3 numbers on M_4,3, the 32-parameter model ARC studied: yes. Our mechanism-derived estimator (E36) beats sampling about 50×, independently confirmed, while ARC's own estimators were 2.4–10× worse than sampling.
  - 3–4 numbers at 16 neurons: the recipe beats sampling on all seeds and noisy copies (not yet independently verified).
  - 5 numbers and up: it starts losing.
- **ARC asked for M_16,10 with 10 numbers.** The pieces near ties multiply about 25–30× per step, so being exact for 2 or 4 steps does not carry over. We have solved short versions, which is a real result, but not the challenge.

<a id="q1-6"></a>
### 1.6 If ARC asked for 16 neurons and 7 steps, would we win?

**Q** (2026-10-18 05:13 UTC): "If what Arc was asking was 16 neurons and 7 steps, can we say that the mechanism that we have found are able to beat the sampling? And win the challenge, just assuming that that's what they have asked for. 16 neurons, 7 steps."

**A:** No. "Faithful through step 7" means that, inside the 10-step network, our formulas reproduce the internal state up to step 7 and the real network then makes the final decisions. A 7-number network makes its near-tie decision at step 7, and our formulas break at the final near-tie decision, whichever step that is. The test computes no accuracy.

Measured at 16 neurons:

| Numbers | Mechanism-based recipe (no random inputs) | Fixed-grid tie-slicing (quadrature) |
|---|---|---|
| 3–4 | beats sampling (unverified) | beats sampling 56–1,900× |
| 5 | loses, about 35× worse | beats 5–15× |
| 6 | not run | 3–5× better |
| 7 | no zoo model yet | probably 2–4× better (interpolated) |
| 8 | not run | 1.5–4× better |
| 10 | infeasible | 0.7–2.3× |

So for 7 numbers the honest answer is "not won by their rules". Tie-slicing would likely win 2–4×, but ARC probably counts it as disguised sampling.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u1-6). Current numbers and status: [INDEX](INDEX.md) rows T277.

<a id="q1-7"></a>
### 1.7 The Hilton talk transcript

**Q** (2026-10-18 05:42 UTC, with the .srt transcript of Jacob Hilton's talk "Miniature mathematical mysteries: uninterpreted models with fewer than 1500 parameters"): "This is a talk from about a month ago, I think." Then at 05:44, with the .txt version: "Read this in full yourself."

**A:** The DRI read the whole transcript.
1. **The challenge is "completely open".** As of the talk, ARC calls it that: estimate M_16,10's accuracy mechanistically, matching random sampling in mean squared error for the compute. It is measured over a family of models "by randomizing the random seed or maybe adding noise to the weights".
2. **ARC's working definition of mechanistic is "no sampling".** Hilton: "even on some subset of the network, you're not allowed to just sample some inputs, see empirically how the network behaves, and assume other things behave similarly." Running inputs is allowed; what matters is reasoning deductively from them rather than extrapolating inductively. Fitting a regression on samples is "a little bit cheating". Applied to our methods:
   - exact enumeration and the recipe: allowed;
   - fixed-grid tie-slicing: not allowed;
   - the probe estimator: borderline;
   - T241's fitted neuron formulas: inductive.
3. **ARC understands less of M_16,10 than we do.** They have "okay-ish understanding" of about 4 neurons (running max, max excluding the last input, latest input, leave-one-out max) and "pretty much no understanding of the rest".
4. **On accuracy:** a hand-built network of about 21 neurons gets 100%, and with 16 neurons "I expect you can do better".
5. **Other points:** near-zero weights matter (assuming insensitivity "is just like false"); M_16,10 has no training checkpoints; ARC has 5 full-time staff, with about 10% of their work on tiny models.
6. **The talk does not support the lenient bar.** It says "match", so the 10× bar rests only on ARC's earlier small-model write-up.

Logged in [lit/web_log.md](lit/web_log.md), with notes at [lit/web_arc/hilton_talk_2026-09_notes.md](lit/web_arc/hilton_talk_2026-09_notes.md). H-E4 and H-E5 were updated.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u1-7). Current numbers and status: [INDEX](INDEX.md) rows HP2, HP4.

---

## 2. The network, its neurons, and the owner's intuition

<a id="q2-1"></a>
### 2.1 Maximum accuracy of this architecture

**Q** (2026-10-18 03:22 UTC): "What is the maximum accuracy that a network trained using these neurons would be able to achieve?"

**A:** Nobody knows. There is no proof of a ceiling, and no construction reaching 100%. Measured accuracies:

| Network | Accuracy |
|---|---|
| M_16,10 (ARC's network) | 95.31% |
| 16 neurons, 5 other zoo seeds | 88.0–94.9% |
| 32 neurons, 5 zoo seeds | 99.1–99.4% |
| 16 neurons with 8 / 6 / 5 / 4 numbers | 95.8–97.7% / 97.0–98.5% / 98.7–99.3% / 99.7–99.8% |

- **Why 100% is probably out of reach for 16 neurons (reasoning, not proof).** Exactly updating "second largest" needs a comparison nested inside another comparison. One ReLU layer per step plus a linear readout cannot do that exactly, so near-ties get misjudged.
- **Part of the error is likely a training outcome.** About 0.83 pp comes from the blind region (at most one positive input), which nothing in the architecture forces. A better-trained 16-neuron network might reach 96% or more (a guess).
- **How to find out:** train longer with more data. This does not affect the ARC challenge itself.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u2-1). Current numbers and status: [INDEX](INDEX.md) rows T234. The final training-ceiling result (18:24, T234) is filed under [4.6](docs/archive/OWNER_QA_later_updates.md#u4-6).

<a id="q2-2"></a>
### 2.2 Accuracy for 2 to 10 numbers

**Q** (2026-10-18 03:32 UTC): "Do we know what would be the accuracy if there are two numbers, three numbers, up to, you know, 10 numbers? I want to know the accuracy for each of these settings."

**A (03:33):** From the AlgZoo models (`m16/approaches/ht_seeds/README.md`):
- 16 neurons:
  - 3 numbers: 99.94–99.95%
  - 4: 99.73–99.83%
  - 5: 98.73–99.26%
  - 6: 97.0–98.5%
  - 8: 95.8–97.7%
  - 10: 88.0–94.9% (M_16,10: 95.31%)
- 8 neurons collapse past 4–5 numbers (35–43% at 10). 32 neurons stay near 99% (99.08–99.44% at 10).
- 2, 7 and 9 numbers were not measured: the zoo has no 7- or 9-number models.
- So capacity, not only training, sets most of the drop, and the spread between seeds widens with T.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u2-2). Current numbers and status: [INDEX](INDEX.md) rows T234.

<a id="q2-3"></a>
### 2.3 Did we find what each of the 16 neurons does? The owner's intuition

**Q** (2026-10-18 03:41 UTC): "The reason that I asked about a network that is fully random and a network that is 100% was that I have this intuition that because the number of neurons in this network is small, it can just capture some structure of the problem. And we already found what each of those neurons are doing. I guess, did we find what each of the 16 neurons is doing or not? First answer me that. But going back to the idea, it seems that the number of neurons is not enough for the network to get to 100% accuracy, which means that for part of the input space, the network might behave just like a network that is random, which has a very large number of regions, as you mentioned, and counting all those regions seems to be computationally more expensive than just sampling. I don't know if we can say it's a hard problem to prove or not, but that's my understanding. And then for the other part that the network was able to solve, I mean the 95%, for those we already understood what could be understood. And definitely we can say that we have a mechanistic answer to that and we know how the network is behaving. And I think it was an early result that we got. So kind of I'm saying for the part that we could provide a mechanistic answer, we already did. And for the error part, the network is not able to create any structure for those regions, and that's why the number of regions is just too many, so we might not be able to find any algorithm to count those regions with lower compute than testing. I hope that I was able to convey my intuition. I guess you might be able to elaborate it better than me. And please give me your thinking about it."

**A:** **Partly.** Most neurons have roles, but there is no complete, verified account of all 16.
- n2: running max (strong from its dynamics, never ablated).
- n4: new-record detector (ablation supports it).
- n1, n6, n7, n8: compare the last input with the stored top values (strong: switching them off destroys the answer).
- n0, n10, n11, n12, n14, n15: "clocks" whose angle records how long ago the max arrived (41.5–42.7° per step vs 42.8° predicted).
- n5, n9: fire after a near-record late input (never tested).
- n3, n13: "harmonics", withdrawn as causal.

The earlier "all 16 confirmed" claim was corrected to 13, in groups. Labels with a few constants reproduce behaviour only 54% of the time, so the understanding is qualitative, about 80%.

**The intuition, sharpened:** too few neurons give a good approximation, structured where the top values are separated and generic where they nearly tie. The error part may have no shortcut better than sampling. The DRI agreed with much of it:
- errors sit at 2nd-vs-3rd ties (2.86%) and in the blind region (0.84%);
- regions grow 23–30× per step, and covering nearly all the probability still needs 15–19× more per step;
- the error tail is a continuum;
- summaries were biased (60 of 60 models overestimated).

The DRI refined it in four ways:
1. Even the 95% is not explained to the 0.01–0.05 pp precision needed: the best compact description gives 98% against the true 95.3%.
2. The region explosion is everywhere, not only in the error part.
3. The error part is "irregular and correlated", which is worse than random for estimation.
4. Nothing can be proven with today's tools, only an empirical obstruction.

If ARC's member solved it, they either found structure in the error part or count compute differently.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u2-3). Current numbers and status: [INDEX](INDEX.md) rows T241, T232, T239.

<a id="q2-4"></a>
### 2.4 Experiments to prove or disprove the intuition

**Q** (2026-10-18 03:50 UTC): "Any other experiments that we can run to prove or disprove my intuition?"

**A:** Yes, five checkable predictions:
1. **Correct-mass vs error-mass concentration** on saved exact enumerations at (8,4), (16,4) and (16,5).
2. **Local region density** near ties vs far from ties. This also settles the DRI's refinement 2.
3. **Comparison with random weights** (T232 and scrambled controls).
4. **Certify the easy part:** how much of the 95% can be proven correct cheaply.
5. **Capacity comparison** at 8, 16 and 32 neurons and on T234's models.

Experiments 1–3 ran as T239 (child, pre-registered), 4 as T240 (Codex lane 78), and 5 was to wait for T234. Even if all came out as predicted, that would be strong empirical evidence, not a proof.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u2-4). Current numbers and status: [INDEX](INDEX.md) rows T232, T239, H-E15, T280, T240, T279.

<a id="q2-5"></a>
### 2.5 Experiments to find each neuron's role quantitatively

**Q** (2026-10-18 03:52 UTC): "Any experiments you suggest we run for finding what each of the 16 neurons does? I see lots of 'never tested', 'role unclear', and things like that. And you're saying that our understanding is qualitative. Why can't we make it quantitative? Let's try to do that."

**A:** Yes. The account is qualitative for three reasons:
1. Roles are spread across neurons (the clock is a 6-neuron rotation).
2. A neuron's role changes with which neurons are on.
3. The final margin is a small difference of large numbers (typical size 2.8 against terms of 19–39), so 99%-accurate descriptions still flip answers.

The honest test is to replace neurons with formulas and check that the answers stay the same. Launched as **T241**, in four phases:
- (A) a causal map: each neuron × step removed three ways;
- (B) formulas for each neuron from meaningful quantities, held-out R²;
- (C) a replacement test with pre-registered targets of ≥ 99% agreement and accuracy within 0.1 pp;
- (D) locating the leftovers.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u2-5). Current numbers and status: [INDEX](INDEX.md) rows T241, T260.

<a id="q2-6"></a>
### 2.6 The right intuition, first version

**Q** (2026-10-18 04:04 UTC): "What's the right intuition to have based on the results of all experiments?"

**A:** Most of this was unverified at the time; the confirmed parts are noted.
1. **The pieces come with the architecture; training decides only what they output.** The trained network has as many pieces as a random one, 25–30× more per step. Training turns about 10% accuracy into 95%. (Counts confirmed to 4 steps; the random comparison unverified.)
2. **Almost all pieces agree, so everything happens at one boundary.** Accuracy is the probability between the network's decision boundary and the true one ("two inputs equal"). That band is the 2nd-vs-3rd ties (2.86%), the almost-all-negative region (0.84%), the 1st-vs-2nd ties (0.72%) and a little else.
3. **The band is frayed, irregular and correlated, not random.** It forms a continuum, not a few families. Covering all but 1e-5 of the probability takes about 16× more pieces per step (confirmed to 4 steps).
4. **The obstacle is precision, not understanding.** Sampling's error at budget C is about 16/√C, around 0.01–0.05 pp. Understanding wins up to about 4 numbers and loses from about 5.
5. **The explanation itself is long.** Disabling any one neuron costs 21–67 pp, and roles change over time and gate pattern.
6. **The owner's intuition, corrected.** Limited capacity does give a structured approximation plus an irregular residual, but the shredding is everywhere and matters only at the boundary.

In practice, deterministic methods win only while the band is simple, and at 10 numbers everything that works is sampling plus small tricks (1.1–2×). It would be overturned by a method that describes the band's probability compactly, or by T234 showing better training makes the band much simpler.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u2-6). Current numbers and status: [INDEX](INDEX.md) rows T239, T280.

<a id="q2-7"></a>
### 2.7 The right intuition, in simple terms

**Q** (2026-10-18 06:18 UTC): "You mentioned that my intuition was kind of correct and kind of wrong. What's the right intuition right now based on everything that we have now? In simple terms."

**A:** The network has a simple story with a detailed fingerprint near the ties.
- **What the owner had right.**
  - There is a clean correct part: the network tracks "biggest so far" and "second biggest so far", and simple formulas describe it through about step 7.
  - About 80% of the error comes from near-ties, and most of the rest from almost-all-negative inputs.
- **What needs correcting.**
  - The correct part is simple in what it does, not in how it is built: it is thousands of linear pieces, about as many as a random network with the same activity.
  - The error part is not noise but a fingerprint. Training packs the fine pieces at the tie boundaries, finer than a random network there and coarser away from them.
  - The fingerprint is model-specific. Retrains and noisy copies share the story but differ in the detail. On noisy copies up to two-thirds of the error sits in a wider band, and a formula fitted to one model is off by up to 20 pp on others.
- **Why this makes the challenge hard.** Accuracy depends on the fingerprint. Pieces multiply 15–19× per step, and every shortcut tried so far misses detail that matters.
- **In one line:** the network's rule is easy to explain, but its accuracy is decided by a fine, model-specific fingerprint near the ties, and no cheap description of it has been found. (Measured, mostly not yet independently checked.)

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u2-7). Current numbers and status: [INDEX](INDEX.md) rows T280, T292, T337.

---

## 3. Our methods and results

<a id="q3-1"></a>
### 3.1 "Better than sampling only on the clean model"

**Q** (2026-10-18 04:38 UTC): "In number one, when we say it's better than sampling only on clean model, could you elaborate a bit more? Be super concise and talk in simple English, simple terms."

**A:** ARC tests a method on the clean model (M_16,10 as trained) and on noisy copies (the same weights plus small random changes), and it has to beat random testing on both. Our best calculation method (T223) had 10–25% smaller error than random testing on the clean model, but was no better on the noisy copies, and 3–6× worse on the full set of 30 models.

It isn't a real solution, for two reasons:
1. It only works on the one network it was tuned around.
2. It relies on a fixed set of test points, which ARC would likely count as disguised testing.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u3-1). Current numbers and status: [INDEX](INDEX.md) rows HP1, HP2, T271.

<a id="q3-2"></a>
### 3.2 The fixed set of test points

**Q** (2026-10-18 04:39 UTC): "Concisely elaborate the second point about fixed set of test points."

**A:** Plain testing runs the network on random inputs and counts how often it is right. Our method runs it on a fixed, evenly spread list (a Sobol grid) and averages. It is smarter about where the points go, but it still checks one input at a time and never uses the weights to reason about the answer. ARC's rule forbids random or pseudorandom inputs; a fixed grid isn't random, but it does the same job, so ARC would likely see it as testing in disguise. A real solution computes accuracy from how the network works.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u3-2). Current numbers and status: [INDEX](INDEX.md) rows T263.

<a id="q3-3"></a>
### 3.3 Our solution for 4 numbers at 16 neurons

**Q** (2026-10-18 05:08 UTC): "Remind me, what was our solution for 4 numbers at 16 neurons? Be very concise."

**A:** It is "recipe108", the same recipe used on the smaller models:
1. Split the input space into cells by input order.
2. Follow each neuron's on/off switch step by step, splitting only when a weight-derived bound says a switch could change the answer.
3. Compute each cell's error exactly as a Gaussian cone mass, with no random inputs.
4. Add up the errors.

At (16,4) it beat sampling on all 5 seeds and 4 noisy copies: worst MSE ratio 0.035, about 30× better, at about 3 billion operations per model. This is unverified, it costs nearly as much as exact enumeration, and from 5 numbers on it loses.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u3-3). Current numbers and status: [INDEX](INDEX.md) rows T194, T196.

<a id="q3-4"></a>
### 3.4 "This solution doesn't look like an understanding"

**Q** (2026-10-18 05:09 UTC): "This solution doesn't look like an understanding to me."

**A:** Right. It is closer to careful brute force.
- At (16,4) it visits about as many cells as the network has linear regions, at about 0.7× the cost of full exact enumeration. It uses a little insight (the true tie boundaries and the weight-based bounds), but it is mostly bookkeeping.
- The clearest sign: its cost tracks the number of pieces, not the mechanism, which is why it breaks at 5 numbers.
- It passed because ARC's written test only asks for no random inputs plus beating sampling at equal compute. That meets the letter of the challenge on small models, not the spirit.
- The closest we came to understanding was E36 on M_4,3: built from the mechanism, about 50× better than sampling, but it did not carry over to the other seeds.
- Our understanding of the running memory is fairly precise (T241: about 99% faithful through step 7) and fails at the final near-tie decision. A real solution would make that decision a compact calculation, for example a formula over the top values and their gaps.

<a id="q3-5"></a>
### 3.5 A formula over the top values and their gaps

**Q** (2026-10-18 05:11 UTC): "Try a formula over the top values and their gaps"

**A:** Accuracy = 1 − P(error), and errors happen almost only near ties. So write a compact formula for P(error | V), where V is the gaps between the top three values, their positions and ages, and the last input's case, then integrate it exactly over the known distribution of the top values of 10 Gaussian inputs. That costs almost nothing, so if it is accurate it beats sampling outright. The bar is about 0.05 pp; earlier "typical value" shortcuts were off by 0.1–1.4 pp (H-E10), and T241's direct fit agreed only 96.6%.

Ran as **T269**, pre-registered:
- Phase 1, a sampled diagnostic: a ladder of richer V sets, integrated exactly and scored on the target, 5 seeds and 2 noisy copies, with no fitting on the scored model. Pass: < 0.05 pp with ≤ about 100 parameters.
- Phase 2, only if phase 1 passed: derive the coefficients from the weights.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u3-5). Current numbers and status: [INDEX](INDEX.md) rows T269, T292, T300, HP2, T302, T304, T317, T317b, T317c, HP3, T317f, HP4.

<a id="q3-6"></a>
### 3.6 The mechanism-based method we hoped would pass the lenient bar

**Q** (2026-10-18 05:22 UTC): "Concisely and in simple terms, describe our mechanism-based method that we hope to give us a solution for lenient bar."

**A:** It is the "probe" estimator (FINDINGS §5):
1. It splits the errors into the blind region (whose probability is known exactly), 2nd-vs-3rd ties and 1st-vs-2nd ties.
2. For each of 720 top-3 arrangements it builds one typical input and bisects for the gap at which the network flips.
3. It converts each flip point into an error probability with the exact gap distribution.
4. It corrects for atypical other inputs: first-order in v3, second-order in v4.
5. It adds everything up.

On the clean model its error went 0.74 → 0.59 → 0.25 pp across v2 → v4, about 23× worse than sampling and the closest any mechanism-based method had come. On one noisy copy it was off by about 8 pp. T274 adds v5 and robustness.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u3-6). Current numbers and status: [INDEX](INDEX.md) rows E7, T271, T274a.

<a id="q3-7"></a>
### 3.7 Where is our hope now?

**Q** (2026-10-18 05:35 UTC): "Where is that our hope now?"

**A:** Three running tests plus one unknown:
1. **Lenient bar with a mechanism-based method, maybe 25–40%.** Either the probe (about 23× on the clean model; if its rate is real (T271), v5 (T274) might reach about 6×, and it must stop breaking on noisy copies) or the gap formula (T269).
2. **Strict bar by exact counting, about 5–10%.** T270 computes the step-6 masses: growth below about 7 per step could make it feasible, while about 12 or more kills it.
3. **Smoothing the tie band (T272), a long shot.** If freezing switches inside the band changes the answer by < 0.02 pp, the band becomes smooth.
4. **The unknown: ARC's rules.** At ARC's small-model noise level, M_16,10's accuracy drops to 12–28%, which is a different problem. We don't know their noise level, budgets, or whether fixed grids count.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u3-7). Current numbers and status: [INDEX](INDEX.md) rows T269, T317, HP3, HP4, T270, T272, HP1, T326, T333, T339, T343.

<a id="q3-8"></a>
### 3.8 Do we pass the lenient bar?

**Q** (2026-10-18 12:22 UTC): "Do we have a solution that passes the lenient bar?"

**A:** Yes, under our own lenient bar (F ≤ 10 with the upper CI ≤ 10 on a sealed panel). The passing estimator is **T317e**, the final cleanup of T317c/T317d with identical outputs.

| Panel | F | 95% CI |
|---|---|---|
| HP3 (100 fresh retrains, scored once) | 5.07 | 3.41–7.03 |
| HP2 (100 retrains, used earlier in the lineage) | 6.75 | 4.86–8.78 |

It costs about 5,400 FPE, with an error of 0.80 pp and a bias of +0.41. The independent scorer ran both panels.
- **What the pass does not mean:**
  - The lenient bar is our convention; ARC asks to match (F ≤ 1), and at F ≈ 5 sampling has about 5× lower error.
  - It covers retrained models only: noisy copies score F 39–206.
  - It is not strictly deductive: there are no runtime random draws, but the approximations were chosen by Monte Carlo on 80 DEV models.
  - It depends on how compute is counted: under the most generous FLOP count, T317c's HP2 interval goes over 10 (upper 11.25) while T317d stays at 9.95. HP3 passes under every count.
  - It is partly self-checked: the same agent did the re-implementation checks and the dossier.
- A newer early-stop rule gets F ≈ 3.1 on the open models and was to be scored on reserved HP4.
- It is a real lenient pass, written up as a progress note, not a solution.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u3-8). Current numbers and status: [INDEX](INDEX.md) rows T317f, T317e, HP4, HP5, H-L1.

<a id="q3-9"></a>
### 3.9 What are the "80 approximations"? Would ARC accept them?

**Q** (2026-10-18 12:28 UTC): "What is this 80 approximations? Is it necessary? Do you think that Arc would accept it? As a non-sampling based method that does not use random or pseudo-random sampling."

**A:** There aren't 80 approximations. **80 is the number of open development models** (DEV2: 80 retrains) used to design and test the method. The method walks the network exactly, gate by gate, along the directions where the top values get close, for each of 720 top-3 placements. It has six declared approximations, seven in T317f:
1. **Closure:** the 7 smaller inputs are replaced by one representative value.
2. **One point for the 4th-largest value.**
3. **Gap cutoff:** cases with gap > 0.5 are assumed correct.
4. **One test point per region:** its verdict is applied to the whole region.
5. **Product rule:** the two directions' answers are multiplied, as if independent.
6. **Blind direction:** one barely used direction is assumed correct.
7. **Early stop (T317f only):** a walk stops once the answer looks settled.

- **Are they necessary?** Some approximation is unavoidable, because exact enumeration is infeasible (≥ 24× growth per step). These particular ones are not sacred. The error budget (second agent) attributes about 40% of the error to the product rule and about 20% to the gap cutoff, but removing either costs as much as it saves, which is why we are stuck at F ≈ 3–5.
- **Would ARC accept it as non-sampling?**
  - At runtime, yes: no random draws (the scorer ran it with numpy's generators disabled), and the fixed normal-distribution tables are model-independent.
  - In design, it is a grey zone: the choices were picked against sampled truths on DEV2. They are the same for every model, nothing is fitted per model, and the scored model was never sampled, so ARC would likely accept that as method design. It is still not deductive in Hilton's strict sense, and the dossier says so (T324 audit).
- **The bigger problem is accuracy:** at F ≈ 5 it is not a solution ARC would accept, so the DRI recommends not contacting ARC yet.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u3-9). Current numbers and status: [INDEX](INDEX.md) rows T333, T327, T339.

<a id="q3-10"></a>
### 3.10 Would 5× more compute match sampling? ARC's M_4,3 factor

**Q** (2026-10-18 12:36 UTC): "Does this mean that if we spend 5x more compute, the error rate would be the same as sampling method? I recall you mentioned that even Arc, when they said for the smaller network, there it almost match or it's very close or something like that, their method was 10x worse. I just want to know that if we are passing that definition."

**A:** **No.** F = 2.63 means that at the *same* compute (about 3,300 FPE) sampling's squared error is 2.6× smaller, and its typical error about 1.6× smaller. Sampling reaches our accuracy with about 1,255 samples, roughly 40% of our compute.
- **More compute doesn't close the gap.** Our error levels off (the closure step alone contributes about 0.46 pp) while sampling keeps improving: T331 spent about 8,400 FPE and reached 0.67 pp, where sampling would be at about 0.30. Matching needs 2.6× cheaper at the same error, or 1.6× more accurate at the same cost.
- **What ARC said about M_4,3:** they "roughly match the computational efficiency of sampling", and were "underperforming sampling by around a factor of 2 in terms of error (i.e., a factor of 4 in terms of compute)". Our reproduction under their protocol gave 2.4–10× in our MSE ratio; the "10×" is the top of that range and the origin of our lenient bar.
- **Do we pass that definition?**
  - Against ARC's M_4,3 "roughly match" level: yes, with room, on retrains (1.6× in error and 2.6× in compute, against ARC's 2× and 4×).
  - Against the M_16,10 challenge as stated: no. Hilton says "match", and calls it "completely open".
  - Test conditions differ: ARC's M_4,3 claim was measured on noisy copies, where ours fails (F 39–206). Ours passes on retrains, which Hilton named as equally valid.
  - Its deductive status is weaker (approximations tuned on sampled truths); ARC's M_4,3 estimator also used a hand-set constant.

It is a judgment call for the owner. If the lenient bar counts, this is a credible candidate, best presented as a progress note with caveats.

<a id="q3-11"></a>
### 3.11 Did our reproduction of ARC use noisy copies? How much noise breaks our models?

**Q** (2026-10-18 12:41 UTC): "When you say that ARC's claim was measured on copies with noise added to the weights, is that what our reproduction of their work also able to show? And how much noise causes our models to break down?"

**A:** **Yes.** For M_4,3 we re-ran ARC's two estimators under their protocol (256 models × 4 seeds, Gaussian noise σ = 0.05 on every weight) and got their error levels back: 3.3% and 4.8%, against ARC's about 3% and 5%. In F terms that is about 10.4 and 2.4, so ARC's "roughly match" was measured on noisy copies and means 2.4–10×. We never tested their estimators on retrains.

T317c on the noisy panel (relative σ = a fraction of each matrix's typical weight size):

| Weights | F |
|---|---|
| Clean | 3.8 |
| Relative 0.5% | 39 |
| Relative 1% | 55 |
| Absolute 0.05 (ARC's M_4,3 level) | 206 (overestimates by 6.6 pp) |

It already fails at 0.5%, the smallest level that panel tests. M_16,10 itself is very noise-sensitive: copies drop to 91.5–94.6% at 0.25% relative noise and 82.3–93.1% at 0.5%. Our approximations rely on regularities that training creates and noise destroys, so the estimator keeps predicting trained-like accuracy and overestimates. T337 was started to measure the breakdown curve.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u3-11). Current numbers and status: [INDEX](INDEX.md) rows (none cited).

<a id="q3-12"></a>
### 3.12 How much noise breaks our solution?

**Q** (2026-10-18 12:44 UTC): "I mean, how much noise causes our solution to break down?" (The second half of this message is in 4.1.)

**A (12:45):** Not known yet. It already fails at 0.5% relative noise (F ≈ 39 against the lenient 10), the smallest level tested. T337 is testing 0.05%, 0.1%, 0.25% and 0.5% on open models.

**A (13:05, T337 result, unverified):** **It breaks between 0.1% and 0.25% relative weight noise.**

| Relative noise | F (T317f) | Overestimate (pp) |
|---|---|---|
| none | 3.3 | +0.58 |
| 0.05% | 3.1 | — |
| 0.1% | 4.0 | +0.75 |
| 0.25% | 12.0 (fails lenient) | +1.63 |
| 0.5% | 32 | +2.94 |

- **Why:** noise creates mistakes where the method assumes it is always right.
  - Clear cases (both gaps ≥ 0.5): 0.09 → 0.76 pp at 0.25%, about half the bias.
  - Answers of rank 4 or lower: about 1 → 3.2 pp.
- Removing the gap cutoff fixes about 40% of the error on one retrain, so that shortcut breaks first.
- Fragility depends strongly on the model: at 0.25% the main model is at +0.28 pp, while two retrains are at +3.3.
- These far-from-tie mistakes are the same rare islands T340 targets.

---

## 4. ARC's long-term goal

<a id="q4-1"></a>
### 4.1 ARC's goal, our mechanistic progress, and whether we used the neuron roles

**Q** (2026-10-18 12:44 UTC): "What was the ARC's goal of defining this problem, this challenge? They wanted a mechanistic solution. Do you think we have made progress in that regard? For example, early on, we found the role of each of the neurons. Have we used any of that information in this solution?"

**A:**
- **ARC's goal.** The challenge tests understanding: if you understand how a network computes, you can work out its behaviour from its weights at least as well as sampling. Accuracy is the test case; the long-term goal is estimating the probability of rare bad behaviour, which sampling cannot find. That is why sampling is forbidden and fitted coefficients are "a little bit cheating". Hilton said matching sampling needs "much much more understanding than we currently have".
- **Progress on understanding is real**, but the estimator barely uses the neuron level. We understand:
  - what each neuron does (T241, checked by a second agent);
  - where errors happen (thin 2nd/3rd near-tie regions, broken into many pieces);
  - what separates good and bad retrains (a diffuse widening of the error regions);
  - why cheap methods fail.
- **What the estimator uses:** the error geometry (organised around the near-tie regions, exact walks along them, a recency rule for the 4th input).
- **What it doesn't use:** the per-neuron roles. It follows the on/off switching exactly, which works for any retrain without knowing its neurons.
- **Why not:** reproducing accuracy within about 1 pp needs formulas accurate to about one part in 100,000, and each retrain implements the roles with different neurons. So the estimator is "white-box" (it reads only the weights and never samples) but not explanation-based in ARC's sense: partial progress.
- **Where understanding could pay off:** the closure step causes about a third of the error. T338 picks from each model's weights which low inputs the network remembers, treats those exactly and lumps the rest. Go line: closure floor ≤ 0.40 pp at about the same cost.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u4-1). Current numbers and status: [INDEX](INDEX.md) rows T327, T338, T341.

<a id="q4-2"></a>
### 4.2 Why the neuron information was not that helpful

**Q** (2026-10-18 13:00 UTC): "Why? The information about what each of the neurons is doing was not that much helpful. This is my understanding based on your description. Maybe I'm wrong."

**A:** Mostly right. The neuron-by-neuron story did little for the number itself, but the higher-level picture it led to shaped the method.
- **Why the roles didn't help the estimate:**
  1. **They describe what the model gets right.** The 92–95% of correct cases are explained, but accuracy is decided by the 5–8% of wrong ones, in thin near-tie regions where tiny weights matter (Hilton agrees).
  2. **They are not precise enough.** The formula-swapped network disagrees on about 3–4% of inputs, and its accuracy moves +0.1 to +0.8 pp. About 1 part in 100,000 would be needed.
  3. **Each retrain is different.** ARC scores over many retrains, and the formula fitted on the main model failed to transfer.
  4. **The method doesn't need them.** Along any line, a ReLU network is exactly piecewise linear, and the walk just follows it.
- **Where understanding did help:**
  - Where to look: the 2nd/3rd near-ties and the nearly blind direction.
  - The recency role informed how the 4th input is placed.
  - Changing roles and switching explained early why freezing or bell-curve shortcuts break after 7–9 steps.
- **The bigger point:** a human-level explanation falls well short of predicting behaviour to half a point, which is the gap ARC's challenge exposes. T338 was the one place the neuron roles might pay off directly.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u4-2). Current numbers and status: [INDEX](INDEX.md) rows T338, T341.

<a id="q4-3"></a>
### 4.3 Are we closer to estimating rare bad behaviour?

**Q** (2026-10-18 12:54 UTC): "Are we closer to help Arc with saying that we can estimate the probability of rare bad behavior? The one that you mentioned is their long-term goal."

**A:** Not directly yet, but this is where our kind of method should have its biggest edge.
- **Not yet, because:**
  - M_16,10 errs 5–8% of the time, which is not rare.
  - The estimator's per-case estimates (each top-3 placement) are off by several points, and the total is right partly through cancellation.
  - The gap > 0.5 shortcut assumes the model is always right far from ties, exactly where rare mistakes would be.
- **Why it could help:** to see a 1-in-100,000 event, sampling needs on the order of 100,000 runs, so at about 3,300 runs it usually sees zero. Our method computes region probabilities from the weights, and its cost depends on how many regions it walks, not on how rare they are. So an estimate within a factor of 2 would already beat sampling by a wide margin.
- **T340 started:** 2–3 rare error events with p between 1e-4 and 1e-6, defined by the inputs (for example "wrong even though the top values are far apart"), on the target and two retrains. It uses the exact walks without the gap cutoff and checks against a 1-billion-sample truth, comparing with sampling at equal compute. If it works, that is a concrete point for ARC; if it fails, we learn where.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u4-3). Current numbers and status: [INDEX](INDEX.md) rows T337, T340, T344.

<a id="q4-4"></a>
### 4.4 Questions our approach can answer better than sampling

**Q** (2026-10-18 13:06 UTC): "Are there any questions regarding the network that our mechanistic approach would be able to better answer than the sampling method? I'm referring to ARC's long-term goal."

**A:**
- **Already shown:**
  1. **Statements that hold for sure.** Exact walks follow the network's piecewise-linear computation along a line and can prove "correct everywhere on this segment" or "first goes wrong exactly here". The rigorous screen built in T330 does this with zero error. Sampling can only say "no errors in N tries"; it can never rule out a small region of failures.
  2. **Exact answers when the model is small enough.** For M_4,3, enumerating its regions gives the accuracy exactly in about 17,500 operations, about 420× cheaper than sampling to 0.1% precision. This route is dead for M_16,10 (≥ 24× growth per step).
  3. **Why and where it fails.** It says which confusion caused each error, at what gap, and which neuron's flip triggered it. Weaker retrains fail through a diffuse widening of the near-tie error regions (60% of the variation), and noise creates errors far from ties.
- **Plausible, being tested:**
  4. **Rare-failure probabilities (T340),** ARC's main long-term target. At one-in-a-million rarity, sampling with a few thousand runs sees nothing. The test drops the gap shortcut, since rare errors sit far from ties.
- **Not yet reliable:**
  5. **How accuracy depends on the weights** (which weights matter, and what small changes do). The method is smooth and noise-free but biased under small noise (breaks at about 0.1–0.25% relative), and its predicted changes track the truth only at about 0.8 correlation.
- **Bottom line:** for the headline accuracy number, sampling is still better at equal compute. The mechanistic approach already wins on exact local guarantees, exact answers on small models and explanations of failures. A working T340 would be the strongest result to bring ARC.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u4-4). Current numbers and status: [INDEX](INDEX.md) rows T344, T340.

<a id="q4-5"></a>
### 4.5 When will T340 be resolved?

**Q** (2026-10-18 13:12 UTC): "When should we have resolved T340? I think this is the part that we should focus. I mean, your answer to my last question, about questions regarding the network that our approach would be able to better answer than the sampling method. …" (The rest of the message asks for this document; see [Owner directives](#directives).)

**A (13:14):**
- T340 was assigned at about 12:55 with a 2-hour budget, so the first result should land around **15:00 UTC**. The DRI asked for an exact ETA, and the GPU agent can do the billion-sample truth run in minutes if that is the bottleneck.
- T340 is now the top priority on the board, with a second lane:
  - **T340 (estimate):** how accurately the exact walks estimate error probabilities of 1e-4 to 1e-6, against sampling at equal compute.
  - **T344 (proof):** a guaranteed upper bound on the same probabilities, using the rigorous check built earlier, on T340's event definitions.
- This Q&A document was commissioned at the same time.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u4-5). Current numbers and status: [INDEX](INDEX.md) rows T340, T344.

<a id="q4-6"></a>
### 4.6 Rare events: the first results (update to 4.3-4.5)

T340 (13:43): on 3 models, rare error probabilities of 2e-7 to 4e-4. A cheap variant (11-19k FPE) estimates the "wrong with both gaps large" events within 3x in 6/6 cases; a careful variant (0.2-0.3M FPE) estimates the far-gap events at .64-.89 of the truth on the retrains and .20-.29 on the main model. Equal-compute comparison with executed costs (post hoc, ties separated): each fixed variant beats sampling in 5/15 cases; using each event family's better variant gives 10 wins, 3 ties, 2 losses. The DRI's first "12 of 15" counted ties as wins and was corrected. No single estimator wins a majority. An independent re-check on fresh DEV3 models (T350) fixes that family rule in advance. T344 (certified bounds): rigorous volume certificates fail for this architecture (certified boxes <= 1e-4 wide even in 3-D; sub-cone enumeration grows to ~2e8 regions), so only the trivial bound is rigorous (8,200-27,500x loose).

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u4-6). Current numbers and status: [INDEX](INDEX.md) rows T350, T340, T340c, T354, T317f, H-S1, T317, T356, T357, T358, T361, T359, T346, T362 ….

<a id="q4-7"></a>
### 4.7 Is the walk method that ties random testing at 10 steps acceptable to ARC? (owner, 10-19 05:2x)

**Q (owner, verbatim):** "This method ties with random testing. You said it's not mechanistic. Are you sure that Arc is not accepting it based on what criteria you're saying? And you're saying that it's a form of structured sampling, but do we sample by running any inputs or looking at outputs and similar? If not, then why you think it's not going to be accepted by Arc?"

**A (DRI, corrected):** Not sure; earlier wording ('not mechanistic', 'not a calculation from the weights') overstated it. The method (T10/H-E16) uses a fixed set of lines through input space, chosen in advance and identical for every model, with no random numbers; along each line it computes from the weights exactly where every neuron switches and exactly which stretches are wrong, and integrates their exact probability; the estimate is a weighted average over the lines. It never draws random inputs or checks outputs on sampled inputs. ARC's criteria: S7 (Competing with sampling, 2025): 'A simple, though not entirely correct, heuristic for whether an estimation algorithm is deductive, is whether it avoids any random or pseudorandom sampling' (the method meets this letter). Hilton's talk (Sept 2026): 'deductive rather than inductive', 'just like no sampling', and 'even on some subset of the network, you're not allowed to just like sample some inputs ... and assume that like other things will behave similarly'; 'Mechanistic isn't really formally defined'. ARC has not addressed fixed node sets. The doubt is the cross-line average: the lines stand in for the whole input space without a weight-derived error bound (the error is judged from alternative fixed sets), which resembles the inductive step Hilton excludes; the line set is also built to spread like random points. For it: no randomness, exact per-line computation from the weights, ARC's own M_4,3 method is deterministic integration, and ARC calls its rule informal. Verdict: grey area, ARC acceptance unknown; only ARC can rule (contact is the owner's call). Making it clearly deductive needs a weight-derived bound on between-line variation, or structure-chosen lines with the error regions handled exactly (T46 hybrid, T389).

<a id="q4-8"></a>
### 4.8 Do we have several solutions for the lenient bar and the main challenge, with only ARC acceptance unknown? (owner, 10-19 06:2x)

**Q (owner, verbatim):** "Does this mean that we have multiple solutions for the lineate bar and also for the main challenge? The only thing is we are not sure if Arc accepts them or not. Is that accurate? Can you name each of them, please?"

**A (DRI):** Partly. Several methods pass our lenient bar (F <= 10, a project convention, not ARC's); for ARC's main challenge (MSE <= sampling at equal compute over a distribution of models) nothing has a clean, sealed, pre-registered pass yet; and ARC acceptance is unknown for all, for different reasons. On the 10-step target class (MSE ratio vs sampling, < 1 wins):

| Method | Kind | Best result | Lenient | Main challenge | ARC acceptance |
|---|---|---|---|---|---|
| T317f exact walks with declared approximations | no sampling, stated approximations | F 2.63 sealed HP4, 2.50 DEV3 | pass | fail; H-S1 shows the family cannot reach 1 (verified) | unknown; closest to ARC's wording (approximations allowed) |
| E33 tie-slicing (H-E11) | deterministic quadrature over fixed directions | ~1.6x better than sampling (ratio ~.6) on the 5 zoo H16T10 seeds; confirmed 10-17 | pass | possibly, on those 5 models only; never sealed, no noisy copies, budget-dependent | unknown (cross-line average, 4.7) |
| T10 recipe (H-E16) at T = 10 (T380) | same kind | .97 (64k) / 1.15 (16k) on 100 DEV3 retrains | pass | a tie, not a pass | unknown (4.7) |
| T389 pre-set allocation | same kind | .74 [.51, 1.01] (16k) / 1.22 (64k) on DEV3 | pass | near-miss; its pre-registered line (64k) failed | unknown (4.7) |

Off-target: H-E16 (16 neurons, 5 steps, confirmed) and T387 (all-at-once network, F .15-.30) beat sampling, but not on ARC's model. Not yet solutions because: (1) no sealed pre-registered pass on untouched models (HP5) at a fixed budget with noisy copies; (2) the quadrature methods lost on heavily noised copies at T = 5 and ARC's noise level is unknown; (3) ARC acceptance (T317f's error is too high; the quadratures rely on the cross-line average). Plan: after T391, one pre-registered sealed HP5 scoring of all frozen candidates (E33, T10 recipe, T389, T391) at 16k and 64k, clean and noisy, by the independent scorer, unless the owner wants HP5 kept unspent.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u4-8). Current numbers and status: [INDEX](INDEX.md) rows T24, T377.

---

<a id="q4-9"></a>
**Update (10-19 18:2x):** the strict bar is now met on a sealed panel by the frozen walk quadrature at T = 10 at budgets >= 1e6 FPE ([INDEX H-E17](INDEX.md#4-hypothesis-ledger), T431); it is a deterministic quadrature-class (inductive under ARC's definition) estimator, so the ARC-acceptance question of 4.7 still applies.

### 4.9 What is ARC's ultimate goal, and how does our work relate to it? (owner, 10-19 06:5x)

**Q (owner, verbatim):** "What do you think is the ultimate goal of Arc?"

**A (DRI; a reading of ARC's published work and the sources logged in [lit/web_log.md](lit/web_log.md), not a statement from ARC):**

*Short version.* ARC wants to make powerful AI safe in situations we cannot test: to know from a model's weights what it will do, including rare catastrophic behaviour that never shows up in testing.

*The problem.* Testing (sampling) only checks behaviour you can trigger. The failures that matter most for safety are the ones you won't see in testing: extremely rare, or appearing only in situations that have not happened yet (a model deceiving its overseers, or taking a harmful action when it "thinks" it won't be caught).

*Their approach.* "Mechanistic explanations": structured arguments built from the weights that predict a model's behaviour without running it on samples, found automatically and general, not hand-written analyses a human reads. They want these explanations for three things:
1. **Low-probability estimation**: estimate the chance of rare bad outputs (even one in a billion), so models can be trained or certified against them before deployment.
2. **Mechanistic anomaly detection**: notice when a model gives its normal-looking output for an unusual internal reason (the camera shows the diamond because the sensor was tampered with, not because the diamond is there); their route to Eliciting Latent Knowledge.
3. **Arguments that hold even when humans cannot follow them**: their theory work (heuristic arguments, the no-coincidence conjecture, the matching-sampling principle).

*Where AlgZoo fits.* A minimum bar, not the goal: if explanations cannot match random testing at estimating a 432-parameter network's accuracy, they will not work on large language models. Hilton: tiny models are "maybe like 10%" of ARC's work; they want "general schemes" rather than one-off analyses.

*How our work maps onto this.*
- Low-probability estimation: our closest match. We estimate rare-mistake probabilities from the weights far beyond what testing can see, for one kind of mistake (H-R1, confirmed, scoped).
- Anomaly detection: a small step. We can name which internal mechanism drives a rare failure (the n6/n7 comparator balance, T372; check T385 running), the "why", not just "how often".
- The minimum bar (matching random testing on accuracy): not cleared; our best methods tie it at 10 steps (T380, T389, T391, T397).

*Judgment.* ARC would likely value a general, automatic method that works passably on many models more than a perfect hand analysis of M_16,10, so future work should favour methods that transfer (the rare-event walk, the comparator analysis) over more tuning on this one network.

<a id="q4-10"></a>
### 4.10 The engineering view: a safe-use rating and repairs for failures too rare to test (owner, 10-19 07:1x)

**Owner (verbatim):** "I recall one time you mentioned that an engineer, when they build a bridge, they don't test it to know when it breaks in what like rare circumstance, but they know the physics and they can simulate and they can figure where that can happen and they can patch it basically. Or they can say that we should not allow cars to with I don't know this weight to pass this bridge under these conditions or things like that. So basically either they they would say that when it's safe to use it and when it's not safe to use it, or they would try to make it more safe. So I think maybe based on the experiments that we have done, we can say that something similar could be done using our methods, especially for the cases that are too rare that the sampling won't be able to catch it or find it."

**A (DRI).** Yes. A bridge engineer does three things without testing the bridge to failure, and our methods already do a small version of each for M_16,10's rare mistakes:

| Engineer | Our analogue | What we have |
|---|---|---|
| **Locate** where the structure can fail (stress analysis) | Find where in input space the network errs, from the weights | Errors live at near-ties of the 2nd and 3rd largest inputs and in mostly-negative inputs (H-E15, T218); the rarest far-gap errors sit on ~6 of 720 position triples (T373); three-way near-tie slivers are the failure mode of the estimator itself (T364, T367) |
| **Rate** the safe load ("no trucks over 10 t when icy") | A safe-use rating: the error probability under stated input conditions, including conditions too rare to test | The estimator gives P(error and g23 >= t) within ~2x deep into the tail (H-R1), i.e. "when the 2nd and 3rd largest differ by at least t, the error chance is about p"; it flags where it cannot be trusted (three-way near-ties). A guaranteed version is in progress (T398-T400) |
| **Reinforce** the weak member | A targeted repair: change the few weights that drive the rare failure and check the failure becomes rarer without hurting accuracy | We know which weights drive the rarest far-gap error (n6/n7 balance, T372; intervention check T385). The repair itself is the next experiment (T402) |

The point, as with the bridge: sampling cannot find or rate a one-in-a-billion failure, but a calculation from the weights can, because its cost does not depend on how rare the failure is. Limits: one network, one kind of rare mistake, estimates (not yet guarantees), and the three-way near-tie blind spot. Two demonstrations are set up to make the analogy concrete: **T401** (a safe-use rating chart computed from the weights) and **T402** (a targeted repair verified by independent truths).

<a id="q4-11"></a>
### 4.11 Are our results all negative, or are some positive? (owner, 10-19 13:5x)

**Answer (DRI):** not all negative. Ledger status is in [INDEX](INDEX.md#4-hypothesis-ledger); in short:
- *Positive, independently confirmed:* locating the weights behind a rare failure (H-R2) and repairing it ~17x with a
  12-weight edit while random edits fail (H-R3); rare-error estimates within ~2x deep into the tail on ordinary networks
  (H-R1); walk quadrature beating sampling at T = 5 (H-E16) and on the all-at-once network (H-A3); the fast rigorous mass
  bound (H-C1).
- *Positive, not yet independent:* the graded rule's blind check (T403), noisy-copy rare events (T388, T404), safe-use
  ratings on two retrains (T401), every pruning certificate re-verifying exactly (T400), and early scale-study data where
  walks beat sampling on wider networks (T423, interim).
- *Negatives that still teach something:* repairing all far-gap errors is blocked because training has optimised the
  common error (T411); a certificate over all inputs is blocked by 'late' orderings (T398, T414).
- *Bottom line:* ARC's main challenge (deductive accuracy at T = 10 beating sampling) is not solved; on ARC's long-term goal
  (rare bad behaviour) we can estimate, locate and repair a rare failure from the weights, but not yet guarantee a bound
  over all inputs.

<a id="q4-12"></a>
### 4.12 What do T, Q, E, H and the other letters mean? (owner, 10-19 14:0x)

**Answer:** see [README.md "ID prefixes and names"](README.md#id-prefixes-and-names) (the one place they are defined).

<a id="q4-13"></a>
### 4.13 Do we have a solution ARC might accept? Must it beat sampling? (owner, 10-19 19:4x)

**Q (owner, verbatim):** "So that we have a solution that Arc might accept for the challenge. If so, what's the solution? And I don't think that they said we have to beat sampling. I think they said it should almost match."

**A (DRI):** The owner is right about the bar. ARC asks for an estimate "competitive with sampling (in MSE vs compute)" (launch thread, [lit/web_log.md](lit/web_log.md)) and called its own M_4,3 estimators, about 2x worse in error (4x in compute), a rough match ([3.10](#q3-10)). Our strict bar (F <= 1) is stricter than ARC's; the lenient bar (F <= 10) is the top of what we measured for ARC's own estimators. The open question is ARC's other condition, "mechanistic" (no sampling), not the bar. Two candidates, numbers in INDEX:
- **Walk quadrature** (H-E16, H-E17, T429, T431): a fixed set of evenly spread lines through input space, the same for every model. Along each line the exact points where neurons switch are computed from the weights, the exact probability of the correct stretches is added up, and the lines are averaged. At T = 10 it roughly matches sampling at small budgets and beats it from about 1e6 FPE (sealed panel). Doubt: the average over lines stands in for the whole input space without a weight-derived error bound, close to the step Hilton excludes ([4.7](#q4-7)). Noisy copies and other line sets are being checked (T434, T442).
- **Gap-direction walks with five declared approximations** (H-L1, T317f): about 2.6x sampling's squared error on a sealed panel, inside ARC's own "rough match" example. Doubt: the approximations were chosen against sampled truths on development models, and weight-noised copies are a weak spot ([3.9](#q3-9)).

Neither is certain to count; only ARC can rule ([4.7](#q4-7)). A weight-derived bound on the variation between lines would make the first one clearly deductive.

<a id="q4-14"></a>
### 4.14 Is anyone working on the weight-derived bound between lines? (owner, 10-19 20:4x)

**Q (owner, verbatim):** "Are we working on any solution to bound from the weights how much the answer varies from line to line to make method one clearly reasoning from the weights? Or we have tried and failed and we are not working on it anymore?"

**A (DRI):** Neither. It was never tried directly; the literature review lists it as open. Related rigorous attempts point to difficulty: interval and Lipschitz-type bounds blow up over 10 steps (E19, T344), and the certified bound works for one top-3 ordering but not the whole input space (T398, T416). Tested as a pre-registered kill test: **NO-GO**, confirmed by an independent re-check for budgets up to 1e10 FPE (INDEX T449, T451). Every bound tried is infinite or hundreds of times looser than the actual error, because the per-line answer has creases and jumps where lines graze region boundaries; closing it would need cells at the scale of the exact regions (~1e13-1e14), far beyond any budget comparable to sampling. So the walk method stays quadrature-class unless ARC accepts it as is.

<a id="q4-15"></a>
### 4.15 Which part of our work is the most novel, and which is the biggest? (owner, 10-19 22:3x)

**Q (owner, verbatim):** "In terms of novelty, which part of our work is the novelist of all? And which part you think is the biggest."

**A (DRI, judged against [lit/novelty_review.md](lit/novelty_review.md) v2; no outside review):**
- **Most novel:** the floating-point-rigorous Gaussian mass of 10-D polytope cells (outward-rounded M5t and its fast version, H-C1). It turns the known probabilistic branch-and-bound pipeline into a certificate near 1e-11 (T416). The closest prior work, Kofnov et al. 2025, computes such masses in low dimension without rigour. Runner-up: sampling-free rare-failure estimates near 1e-10 from exact line walks, without enumerating regions (H-R1).
- **Biggest:** the sealed strict pass at ARC's full size. Fixed-node line walks have about half the error of random testing at equal compute on 100 never-used retrains (H-E17, T431). It is new as a result, not as a method (line sampling + QMC + ExactLine), and it stays quadrature-class (no weight-derived bound: T449, T451). For ARC's long-term goal, the biggest open bet is a guarantee over all inputs, which T418 reopened (T455).

<a id="q4-16"></a>
### 4.16 Which part is the biggest contribution to AI safety? (owner, 10-19 22:4x)

**Q (owner, verbatim):** "I was about to ask which part of our work is the biggest contribution to the AI safety."

**A (DRI):** The rare-failure work, end to end, on a real trained network: estimate failures too rare for testing (within about 2x near 1e-10, H-R1), locate and repair them with a small targeted edit (H-R3), and prove an upper limit from the weights (T416 for one input section; all inputs may be feasible, T418 / T455). The guarantee is the part a safety case most needs, because testing can never give it. The most transferable lesson is T411: training sits at a balance point, so repairing rare failures by small edits turns some into common ones, a trade-off we measured rather than assumed.

<a id="q4-17"></a>
### 4.17 How large a network can our methods handle, and what compute do they need? (owner, 10-19 22:5x)

**Q (owner, verbatim):** "Give me an estimate of how large a network can be such that we would be able to apply our most useful and interesting methods. Also, give me an estimate of how much compute (maybe in terms of the cost and time) we would need for running our methods and providing what we were able to find."

**A (DRI; educated estimates, not measurements, except where an INDEX row is named).** FPE = one forward pass of the network; M_16,10's pass is ~6e3 FLOPs. CPU priced at $0.01-0.05 per core-hour.

*How large a network (my estimate, confidence in brackets):*

| Method | Needs | Practical up to about | Main limit |
|---|---|---|---|
| Focused-sampling rare-failure estimates (T436-T456) | forward passes and gradients; a known input distribution | 1e8-1e9 parameters (medium-low) | finding every failure region when inputs are high-dimensional; honest error bars were the hard part even at 10-D |
| Locate and repair (H-R3) | the rare-failure estimate plus curvature | 1e6-1e8 parameters with Hessian approximations (medium-low) | rare-to-common conversion (T411) |
| Line walks for accuracy (H-E17) | piecewise-linear units (ReLU, hard or piecewise-linear attention, T444) | 1e5-1e6 parameters, inputs up to ~20-50 dimensions (low) | pieces per line grow with size; the gain over sampling shrinks as input dimension grows; wider nets helped so far (T423) |
| Safe-use rating (T401) | as the rare-failure estimate | as that row (low) | per-network trust flag |
| Certified upper bound (T416, T418) | piecewise-linear units, low input dimension | ~10-20 inputs and a few hundred to ~1e3 unrolled ReLUs (medium); M_16,10 has 160, so (32,10) may need 10-100x more compute | cells grow exponentially with neurons and input dimension |

*Compute to reproduce what we found on M_16,10 (from INDEX where named):*

| Result | Compute | Cost today | Time |
|---|---|---|---|
| Accuracy estimate beating sampling, one model (T431) | ~4e6 FPE, ~2e10 FLOPs | < $0.001 | under a minute on one core |
| Same on a 100-model panel, with 1e8-sample answer keys | a few CPU-h | < $1 | an hour on a laptop |
| One rare-failure estimate near 1e-10 (T456) | ~3-5e6 FPE; its answer key 1-10 CPU-h | < $1 | minutes, plus hours for the answer key |
| Targeted repair and its check (T402, T410) | edit in minutes; 1e9 paired inputs ~1 CPU-h | < $1 | about an hour |
| Certified bound, one easy input section (T416) | 130-260 CPU-h (COMPUTE_FRONTIER 4a) | $3-13 | a day on a few machines |
| Certified bound, the section with the mistakes, (9,3,2) (T398, T400) | ~250 CPU-h to enumerate + ~3.7e3 to charge | $40-200 | ~5 days on 32 processes (less with T452) |
| Certified bound over all inputs (if T455 works) | ~1e4-1e5 CPU-h (low confidence) | $100-5,000 | 2-8 weeks on ~64 cores |

*At larger sizes (estimates):*
- Focused sampling on a 1e6-parameter net costs ~1e7 passes, ~2e13 FLOPs: about a CPU-hour.
- The same on a 1e9-parameter language model with ~100-token inputs is ~2e18 FLOPs: a few GPU-hours, ~$5-20.
- Line walks on a 1e6-parameter net: CPU-minutes to hours per model, but whether they still beat sampling there is untested.
- Certified bounds stop being affordable well before 1e4 unrolled neurons.
- In this project the main cost was agent time (tokens), not compute.

<a id="q4-18"></a>
### 4.18 Can we apply our methods to a billion-parameter model, and at what cost? (owner, 10-19 22:5x)

**Q (owner, verbatim):** "Can we apply our methods to a billion-parameter model? How much would it cost? You can give me dollars or gigawatts of compute for a month."

**A (DRI; educated estimates).** Assumptions:
- 1 GW for a month is about 1e6 H100-class GPUs, ~1e27 FLOPs, ~$1-1.5B at ~$2 per GPU-hour.
- A 1e9-parameter model with ~100-token inputs costs ~2e11 FLOPs per forward pass.

| Method | Applies to a 1e9 model? | Compute per use | Cost |
|---|---|---|---|
| Focused-sampling rare-failure estimate with error bars (T456) | yes, if the input distribution is defined (e.g. a prompt generator) | ~1e7 passes + ~1e6 gradient steps, ~3e18 FLOPs | ~$5-30 per failure type (a few GPU-hours) |
| Checking that estimate against brute-force sampling | only for rates >= ~1e-6 | 1e8-1e9 samples per event; ~1e12 at 1e-10 | $50-500 per event; ~$100k per event at 1e-10 |
| Safe-use rating | yes, as above | a few estimates per condition | $100s-1,000s |
| Locate and repair | yes, with Hessian approximations (as in OBS-style LLM pruning) | GPU-hours to GPU-days | $100-10,000 |
| Line walks for accuracy | no as is: softmax, GELU and LayerNorm are not piecewise linear, token inputs are not lines, and pieces per line explode | - | - |
| Certified upper bound | no | cells grow ~5-10x per 16-neuron step (T398) | 1 GW-month (~1e8-1e9x our full M_16,10 certificate, ~1e18-1e19 FLOPs) buys ~150-200 more unrolled neurons: a network about twice M_16,10, not 1e9 parameters |

So the sampling-class safety tools cost about 1e-8 to 1e-6 of a GW-month: one 8-GPU node for hours to days. The bottleneck at that scale is method, not compute: defining the input distribution, and finding every failure region in a huge discrete input space (honest error bars took three iterations even in 10-D). Exact and certified methods do not reach a billion parameters at any budget we can foresee.

<a id="q4-19"></a>
### 4.19 Our methods on a 1e9-parameter transformer language model built to suit them (owner, 10-19 23:0x)

**Q (owner, verbatim):** "It would be very interesting if we could apply our methods to a billion-parameter model that is built for language modeling using a transformer and self-attention, which we think our method would be able to handle. I would like to hear your numbers."

**A (DRI; educated estimates; nothing here is measured at this scale).** The model is the scaled-up T444 design: decoder-only, ReLU MLPs, piecewise-linear routing (sparsemax or positional / hard attention), no LayerNorm. It is piecewise linear in its embeddings, so the exact machinery applies in principle. Its quality at 1e9 parameters is untested; T444 is a 1e4-parameter result.
- Assumptions: H100-class GPU at ~4e14 effective FLOP/s and $2-5 per hour; 100-token inputs, ~2e11 FLOPs per forward pass.
- 1 GW-month is ~1e27 FLOPs, ~$1-1.5B, ~7e5 MWh.

| Step | Compute | Cost | Time on one 8-GPU node |
|---|---|---|---|
| Train it (20e9 tokens, 6ND, 2-3x slowdown for sparsemax kernels) | ~3e20 FLOPs | $1,000-5,000 | 1-3 days |
| Rare-failure estimate with error bars, per failure type (T456 recipe; discrete search over token sequences instead of MIQP) | ~3e18 | $5-50 | under an hour |
| Brute-force check of that estimate (rates >= 1e-6) | 2e19-2e20 | $50-500 | hours |
| Safe-use rating, ~6 conditions | ~2e19 | $50-300 | hours |
| Locate and repair (rare-event gradient from IS samples; Hessian-vector products on ~1e4 candidate weights; paired check) | ~1e19 | $100-2,000 | hours to a day |
| Exact prefix-tree head (the S2 / T445 analogue of exact walks): depth 2 (1e9 prefixes) | ~2e20 | $300-700 | ~1 day |
| Same at depth 3 (3e13 prefixes) | ~6e24 | $2-5M (~0.005 GW-month) | months on a large cluster |
| Local certificates (one prompt, a few token edits; IBP / CROWN, known methods) | GPU-minutes to hours each | $1-10 per prompt | - |
| Global certified bound over all prompts | V^L inputs; cells grow per layer as in T398 | not reachable even at 1 GW-month | - |

**Totals:**
- A full study (train, 100 failure types estimated and checked, 10 repairs, ratings, a depth-2 exact head) is ~$5,000-50,000. That is ~25 MWh, about 3e-5 of a GW-month.
- The method risks matter more than the money:
  - defining the prompt distribution and what counts as a bad output;
  - finding every failure region in a discrete input space, which honest error bars depend on (it took three versions at 10-D);
  - the routing variant's quality at scale.
- A staged path: the S5 ladder (T448) first, then a 1e7-1e8 synthetic-language rung for ~$10-100, then 1e9 on real text (backlog T457).

<a id="q4-20"></a>
### 4.20 How novel was our work, in simple terms? (owner, 10-19 23:0x)

**Q (owner, verbatim):** "simply and concisely. Describe what were the novel contribution of our work and what we found related to it in the literature. So I want to know how much novel it was. be concise and talk in simple terms"

**A (DRI):** The plain-language summary of [lit/novelty_review.md](lit/novelty_review.md) section 3 (the verdict table is the source). Overall: mostly known ideas applied to a new target. Likely new are the rigorous 10-D region volumes, the sampling-free rare-failure estimates near 1e-10, and the sealed result itself ([4.15](#q4-15)). The focused-sampling estimator (T436-T456) applies Bai et al. 2020 and Hesterberg 1995. No outside reviewer has checked these verdicts.

<a id="q4-21"></a>
### 4.21 Honestly, would ARC accept our results? Judged by why they set the challenge (owner, 10-19 23:0x)

**Q (owner, verbatim):** "I would like your honest response regarding whether Arc accepts our solutions as valid for their challenge. You can respond by thinking about what the goal of Arc and Hilton was in proposing this challenge. What did they want to get out of it? Why did they talk about mechanistic, and what did they want from mechanistic? Then look at our method and think about whether it matches that. If you think about the why, then you would be able to make a proper judgment. There was a reason that they designed this challenge. They wanted to use it for some purpose, and we need to think about that. What's your probability of their accepting each of these results and things that we propose and find?"

**A (DRI; subjective probabilities; ARC has not been contacted).** Why ARC set it ([4.10](#q4-10), [lit/web_arc/hilton_talk_2026-09_notes.md](lit/web_arc/hilton_talk_2026-09_notes.md)):
- They want methods that estimate what a network does by reasoning about how it computes. Sampling cannot see rare catastrophic behaviour, nor tell when a model behaves well for the wrong internal reason (mechanistic anomaly detection).
- Accuracy is the calibration test: sampling works there, so a mechanism-based estimate can be checked against it. Matching sampling is evidence that the understanding is real and efficient.
- "Mechanistic" (deductive, "no sampling even on some subset", fitted coefficients "a little bit cheating") rules out the step that fails on rare events: looking at some inputs and assuming the rest behave alike. They expect general schemes, not one-off hand work.

Four tests follow: (1) no inductive step; (2) the estimate comes from an account of how the network computes; (3) the account carries over to rare events and anomalies; (4) it could scale.

| Result | (1) | (2) | (3) | (4) | P(ARC counts it as a valid solution) |
|---|---|---|---|---|---|
| Line-walk quadrature, strict pass (H-E17) | partly: no random draws, but the average over fixed lines is the "assume the rest behave alike" step | no: generic, explains nothing | partly | weak | ~10% |
| Gap-direction walks + five declared approximations, F 2.6 (H-L1) | mostly at run time; approximations chosen against sampled truths | partly: built on top-3 order, gaps, independence assumptions | weak: misses triple-tie tails | unclear | ~15% |
| Any of our results accepted as solving M_16,10 | | | | | ~20% |
| Certified bound (T416, T418) | yes | no: exhaustive splitting, which ARC sees as not scaling | yes, rigorous | exponential | ~1% as a solution; valued as an independent check |
| Deterministic rare-event walks, Q16 (H-R1, T370) | mostly (QMC-class) | partly (tie structure) | yes | unclear | outside the challenge; ~30% they find it useful |
| Focused-sampling IS (T436-T456) | no: sampling | no | yes | yes | ~0% as mechanistic; a strong baseline of the kind ARC's own rare-output work uses |
| Targeted repair (H-R3, T411) | - | - | - | - | outside the challenge; modest interest |

Most likely reaction: strong baselines that raise the bar for "matching sampling", plus useful rare-event and certification checks, but not the explanation-driven estimate the challenge exists to elicit. What would change that: an estimator derived from a written account of how M_16,10 computes the 2nd argmax, with no constants tuned on sampled truths, and holding on weight-noised copies at F <= ~4.

## 5. Compute, GPU and process

<a id="q5-1"></a>

**Correction (10-19 06:4x):** the E33 row is withdrawn: at executed cost it loses on 100 M_16,10 retrains (T397 in INDEX); no method has a DEV3 confidence bound below 1, so no sealed HP5 run yet.

### 5.1 Which task needs a GPU?

**Q** (2026-10-17 23:01 UTC): "Which task required GPU? I have a Google Colab with 300 compute units. I want to set that up first. …" (The rest is in [Owner directives](#directives).)

**A:** **T192**, the compiled/GPU exact reference and cost law. It moves exact calculation and the recipe onto GPU code, measures cost growth at (8,5) and (8,6), and decides whether an exact answer key for the full network is affordable. That last part is T5, which was blocked on compute approval and is included. Child session `session_016BGedpds19gq95yjWmDMn5`, branch `claude/t192-gpu-cost-law`. Its rules:
- talk to the owner before coding;
- reproduce the CPU results first;
- ask before spending more than about 30 units on any one stage;
- report its spending.

At 23:10 its budget became 150 of 300 units without asking, with a stop if any stage passes 40.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u5-1). Current numbers and status: [INDEX](INDEX.md) rows T192, T270, HP1, HP5, H-E13.

<a id="q5-2"></a>
### 5.2 Questions for the previous DRI

**Q** (2026-10-17 23:12 UTC): "Make sure you have a good understanding of the problem, what we have tried, what was successful and what wasn't, and any context that you need. If you have any questions for the previous DRI, gather all your questions, and at the end of your current work, tell me those questions all in one message, and I will ask the old DRI." Then at 00:31: "Did you have any questions for the old DRI? I forgot to send them."

**A (23:34, then 00:31):** Sixteen questions at first. Four were resolved by 00:31: where the Codex work lives, how to reach Codex root, the E46/E48 cost units (ruled in T196) and where `leaf_mass_review` is. That left 12 plus 2 new ones, of which #1, #5 and #6 mattered most:
1. Why recipe108 starts from 4·T! half-chambers.
2. The T170 certificate fix.
3. The `evalH.py` double count.
4. Whether "≥ 1e17" was a bound or an extrapolation, and T133's 2.76e10.
5. Whether the (8,5) failure was expected, and why exact enumeration was never scaled.
6. Exact masses sitting low against sampling.
7. Which protocol is canonical.
8. The leaderboard taking the minimum over budgets.
9. arc_cumulants and srb.
10. E44's 0.59 vs 0.765.
11. The 216-state lookup.
12. The error breakdowns.
13. The blind region and clock slope.
14. The source of the ARC claim.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u5-2). Current numbers and status: [INDEX](INDEX.md) rows (none cited).

<a id="q5-3"></a>
### 5.3 What the Codex managers wrote in DRI_HANDOFF.md

**Q** (2026-10-18 00:26 UTC): "m16/approaches/crossover_compute/DRI_HANDOFF.md — check what the Codex agent managers have written for you and make sure they're under your guidance."

**A:** The DRI read Codex root's two sections, replied at the top of the file, and posted DRI directive 6.
- Codex confirmed the owner's publication authorisation and integrated 6,030 paths onto the shared branch.
- They proposed that Codex root dispatches and integrates while the DRI keeps scientific priorities, shared code and confirmation. The DRI accepted.
- Their lane plan predated the (8,5) failure, so the DRI sent the new main route, exact enumeration made cheaper by structure.
- New lanes:
  - 34: T193 as an independent exact-mass calculation;
  - 56: T188, losslessly merging cells;
  - 78: T189, the bound's undecided mass vs sampling;
  - h4: auditing T195;
  - 910: T197, the `evalH.py` fix;
  - T55: reviewing T192's port and op count;
  - root: keeps the cost-recount lanes frozen.
- The file became the standing channel.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u5-3). Current numbers and status: [INDEX](INDEX.md) rows T240, T260, T261.

<a id="q5-4"></a>
### 5.4 Why does Google reclaim the Colab node?

**Q** (2026-10-18 03:28 UTC): "Ask a subagent to figure out why Google reclaims the node and if we can prevent that."

**A (03:40; an inference from public documentation and code, untested):**
- **Most likely cause:** our runs looked idle.
  - On September 25, Google removed the Colab CLI's keep-alive ping, so a runtime stays alive only while its kernel is busy or a connection is open.
  - Our jobs were launched with `nohup`, so the GPU looked idle between polls.
  - Google's code reclaims idle runtimes within about 30 minutes, matching the 15–60 minutes observed.
  - Premium-GPU capacity pressure is also possible.
- **Other candidate cause:** CLI versions before 0.7.4 falsely report sessions as lost after about an hour.
- **Fixes sent to T192:**
  1. CLI ≥ 0.7.4, and check whether the machine is really gone.
  2. Hold a connection open while jobs run, and read logs by download.
  3. Checkpoint every 5–10 minutes.
- **If reclaims continue:**
  - Pro+ background execution (undocumented for CLI use);
  - Colab Enterprise (about $0.78/h L4, $4.11/h A100);
  - a plain Google Cloud GPU VM (about $0.85/h L4, $3.67/h A100).
  - Both alternatives need billing set up by the owner.
- Memo: [review/colab_ops/RECLAIM_MEMO.md](review/colab_ops/RECLAIM_MEMO.md).

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u5-4). Current numbers and status: [INDEX](INDEX.md) rows HP5.

<a id="q5-5"></a>
### 5.5 When will all these experiments be done?

**Q** (2026-10-18 04:46 UTC): "When do you think we're done with all of these experiments?"

**A:** The main scientific question was settled and the report published. Estimates for the rest:

| Experiment | Estimated finish |
|---|---|
| 2–10 accuracy table | about 05:30 |
| Neuron-study review | about 05:45 |
| Training ceiling (T234) | 6–12 h, depending on when accuracy saturates (around midday) |
| T240 (Codex) | several hours |
| Codex reviews | ongoing |

The training study could be capped at 2× and 4× data to finish in about 3 hours.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u5-5). Current numbers and status: [INDEX](INDEX.md) rows T240.

<a id="q5-6"></a>
### 5.6 Why is the training-ceiling study slow? Better GPU, CUDA or C?

**Q** (2026-10-18 04:48 UTC): "Why the training ceiling is so slow? Can't we use a better GPU and optimize our code? Maybe write a CUDA kernel for it? Something that runs it way faster than what we have, or write it in C."

**A:** It can be much faster, but GPU power is not the problem.
- The network is tiny: each step is a few million operations, done in microseconds.
- Each step still takes about 1.4 ms because of launch overhead. Steps are sequential: batch 1,024 means 4M steps for 2^32 sequences (about 1.6 h) and 16M for 2^34 (about 6.5 h).
- A faster GPU barely helps.
- Options:
  - bigger batches (8–16× fewer steps, minutes of work; a recipe change, so keep standard-batch controls);
  - a fused CUDA/Triton kernel (probably 10–30× per step, a few hours of work, must be validated);
  - C on CPU (about the same, not worth it).
- Packing many models is nearly free: 20 models take 2.5 ms against 1.4 ms for 5.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u5-6). Current numbers and status: [INDEX](INDEX.md) rows T192, T261.

<a id="q5-7"></a>
### 5.7 Can one fused kernel do data generation, forward and backward?

**Q** (2026-10-18 04:53 UTC): "Can the fused GPU kernel do the whole data generation, forward, backward, everything?"

**A:** Yes, and that is the fastest design.
- Inputs come from in-kernel counter-based Philox random numbers: reproducible, and resume exactly.
- Labels take a few comparisons.
- The forward pass, loss, backward pass and AdamW all run in-kernel, with the 16×16 weights in on-chip memory.
- With one model's batch of 1,024 per thread block, a launch can run thousands of steps with no host involvement, which removes the launch-overhead bottleneck.
- Caveat: the inputs are not PyTorch's numbers, so the result is statistically equivalent but not bit-identical. Validation is "reproduce the existing 5-seed accuracy range".

The design was passed to both kernel writers.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u5-7). Current numbers and status: [INDEX](INDEX.md) rows T192.

<a id="q5-8"></a>
### 5.8 Could one fused kernel run all the experiments?

**Q** (2026-10-18 04:54 UTC): "And if you're able to fuse the whole training and everything, would we be able to run all experiments that we need in just one fused kernel? I mean, we just launch one fused kernel and it runs all the experiments that we want on GPU."

**A:** Yes, almost exactly.
- **How it works:**
  - Each thread block trains one model from its own settings row: seed, data size, learning rate and schedule, batch size, hard-case share.
  - Runs of different lengths can share a launch.
  - Bigger batches are handled by summing gradients over sub-batches.
  - Hard-case data and the final testing are made in-kernel.
- **Caveats:** run in 5–10-minute chunks with saves between them, and pass correctness checks first.
- **Rough estimate (not measured):**
  - about 17M operations per model-step, about 0.1–0.3 ms on one core cluster;
  - about 100 models at once on an A100;
  - a 2^30 run about 5 minutes, 2^34 about 1–2 hours;
  - the whole 60–100-run study in about 1–2 hours, against about 12 hours.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u5-8). Current numbers and status: [INDEX](INDEX.md) rows HP2.

<a id="q5-9"></a>
### 5.9 Have we published all results, and do the reviewers have them?

**Q** (2026-10-18 05:04 UTC): "Have we published the above results, all of them? And do you know if the Fable and Codex agents know about them? I think we might need to tell them to fetch again to get all the results."

**A:** Not all.
- The ledger (INDEX.md) had every result, but the detailed write-ups sat on each child's branch.
- The published report lacked a T241 section.
- The Fable reviewer had started at about 04:57, before the T241 corrections.

Fixed at 05:05:
- Section 8 on the neuron study was added to [m16/SCALING_OBSTRUCTIONS.md](m16/SCALING_OBSTRUCTIONS.md).
- Fable was told to refetch everything and note which versions it read.
- Codex Astra's task got a "fetch everything fresh first" note with every branch's current commit.

---

## 6. Architecture and the exact route

<a id="q6-1"></a>
### 6.1 Would a different architecture make it easier?

**Q** (2026-10-18 13:19 UTC): "Do you think if the model was taking all 10 steps at once, similar to how Transformers models take the whole sequence at once, would it have been easier for us? Basically, I'm asking if Arc designed a different architecture for their model, would it make it significantly different for us or not?"

**A:** Probably a lot easier, but because of depth more than "all at once", and ARC likely chose a deep architecture on purpose.
- **What makes this model hard.** A 10-step RNN is a 10-layer ReLU network that reuses its weights, and every major obstacle grows per step:
  - exact enumeration: linear regions grow 24–33× per step;
  - rigorous bounds: uncertainty is amplified about 50× per step, so proofs only work on tiny boxes (T330, T344);
  - Gaussian (bell-curve) state approximations break by steps 7–9;
  - M_4,3 (3 steps) is solved exactly and cheaply, and the length scans show difficulty climbing with T.

  So depth, not width, is the main enemy.
- **A shallow network seeing all 10 inputs at once (1–2 layers)** has no compounding:
  - its regions are cut directly by hyperplanes in input space, so exact enumeration and integration are far more tractable;
  - bounds don't compound;
  - pairwise comparisons would be explicit.

  A mechanistic estimate matching sampling would likely be much more reachable. This is an inference, not tested.
- **A Transformer is mixed.** It has few layers, and attention compares pairs, which fits the near-tie structure. But softmax and normalisation are smooth, not piecewise linear, so the exact-walk machinery would need different math.
- **What wouldn't change:**
  - errors still come from the task's 2nd/3rd near-ties;
  - a shallow network is also cheaper to sample, so the bar moves too;
  - a 432-parameter feed-forward network might learn the task worse, since the RNN gets its efficiency from weight reuse.
- **Why ARC probably chose this:** their target is deep Transformers. The compounding here is a miniature of the depth problem, so the 10-step recurrence is the point of the challenge. A cheap test would train a small feed-forward network with the same budget and run our exact tools on it.

**Later updates:** moved to the archive, [full text](docs/archive/OWNER_QA_later_updates.md#u6-1). Current numbers and status: [INDEX](INDEX.md) rows T345.

<a id="q6-2"></a>
### 6.2 Why are exact and region-counting methods dead for this problem?

**Q** (2026-10-18 13:24 UTC): "You told me that exact methods or pseudo exact methods that count regions or similar were dead for this problem. Elaborate a bit on that."

**A (13:30):**
- **The idea.** The model is a ReLU network with no bias terms. Fix every neuron's on/off state at all 10 steps (a gate pattern), and the network becomes a linear function of the 10 inputs. The inputs that produce one pattern form a cone; inside it, "is the answer correct?" is a few linear inequalities, and a cone's Gaussian probability is a solid angle, computable exactly. So accuracy = the sum over cones of the probability of each cone's correct part: exact, no sampling. This works on M_4,3 (~88 cones, zero error, ~420x cheaper than sampling to 0.1%).
- **Why it dies on M_16,10.** The number of cones explodes per step: ×23, ×27, ×30, ×31-33 for k = 2..6; 632,394 at k = 5; ≥ 15.5M at k = 6 (exact lower bound, still undercounted because of the T316 geometry bug). Extrapolated to 10 steps: on the order of 1e13 regions, each needing its own linear-algebra and volume work, against ~74,000 forward passes for sampling to reach 0.1 pp. Roughly a billion times more expensive.
- **The pseudo-exact versions die too.**
  - Cover most of the probability and bound the rest: the regions needed to leave a fraction δ uncovered still grow ≥ 7.6×/step (δ = 1e-3) to 14×/step (δ = 1e-7) at k = 6; the pre-registered kill line was 12×.
  - The probability-weighted effective count grows ~5×/step.
  - Interval/relaxation bounds widen ~50×/step (the recurrent weights amplify uncertainty).
  - Branch and bound grows 15-30×/step.
- **The underlying reason.** Errors live where the 2nd- and 3rd-largest inputs are nearly tied; near those walls the regions are ~40× finer, and the error probability is spread thinly over very many of them. No small set of big regions carries the answer.
- **Scope of "dead".** Measured, not proven: region-by-region enumeration or coverage of M_16,10 at 10 steps is infeasible at any compute where comparison with sampling makes sense. Counts are independently confirmed to k = 4, and the 10-step figure is extrapolated. A cleverer exact method (e.g. merging regions with the same decision) is not excluded, and exact computation as a component is alive: T317f computes exactly along chosen 1-D lines and approximates only how those lines combine.

**Records:** H-E13 in INDEX.md (confirmed counts 34 / 792 / 21,032 for k = 2..4); T270 and T316; T222 (mass coverage); T279/T289 (bounds); 1.1 *Later updates* above.

<a id="q6-3"></a>
### 6.3 How much compute would exact enumeration of M_16,10 need? Could a 1 GW data center do it in 6 months?

**Q** (13:31 UTC): "You keep telling me that the count for M1610 would be around 10 to the power of 13 regions ... How much is that work? And how much compute would be needed for the whole thing? ... Assume that we have a very large data center, like 1 gigawatt ... Would we be able to compute it in less than 6 months?"

**A:** Probably yes; plausibly in days to weeks. Order-of-magnitude estimate:
- Per region: an LP to find it, a split into correct/incorrect parts, and an exact Gaussian volume in 10-D (the expensive part). Measured at k = 6 (T316, subtree 125): ~6.5 ms/region/core with estT, ~0.2 s with the bug-fixed geometry; assume 10-100x more at 10-D.
- Regions at T = 10: ~2e13 (growth stays x31) to ~4e13 (rising to x39) to ~1e14 (rising fast).
- Total: ~3e12 to 8e14 core-seconds (~1e21-1e24 operations).
- A 1 GW GPU site has ~5-7e5 GPUs and ~1e7 host CPU cores; a CPU-heavy site ~1e8 cores; 6 months is ~1.6e7 s. Central cases take hours to ~3 months; the worst corner (1e14 regions, robust geometry, 100x) ~6 years on 1e7 cores or ~8 months on 1e8 cores, unless the volume work moves to the GPUs. Embarrassingly parallel over subtrees, little memory.
- Comparison: sampling reaches 0.1 pp with ~74,000 forward passes (~4e8 operations); exact needs ~1e12-1e16x more. A 1e12-sample Monte Carlo on a few GPUs already gives ~±3e-5 pp. Exact would add only a proof.
- Correction (given at 13:52): strictly, ARC compares error at equal compute, and an exact answer has zero error, so brute force meets the letter of the bar at its own huge budget; it does not meet its point (a cheap mechanistic estimate).

<a id="q6-4"></a>
### 6.4 GPU brute-force counts: how long would they take?

**Q** (13:36 UTC): "Yes, the GPU brute force counts. That is your idea. How long it would take."

**A:** Exact counts at 7 steps with today's CPU-bound code would cost ~30x the 6-step run: ~500-700 A100-hours, ~3,000 Colab units (~10x the budget), or about a day on 30 VMs. A sampling census instead (>= 1e9 inputs, distinct gate patterns, mass coverage, restricted to near-tie and error inputs, k = 4..10) takes ~15-30 min of GPU and ~0 extra units. Queued as **T348**; validated at k = 4 against exact counts (394 / 1,699 vs 396 / 1,737). Caveat: at k >= 8 most patterns are singletons, so fine-delta counts are lower bounds with a Good-Turing missing-mass estimate. (The GPU census result, T348 at 16:36, is filed under [4.6 later updates](docs/archive/OWNER_QA_later_updates.md#u4-6).)

<a id="q6-5"></a>
### 6.5 Would quantising the model (binary, FP4, FP8, int4, int8) let us solve the challenge?

**Q** (13:50 UTC): "What do you think if we quantize this model? Would it help? ... binary model or FP4 or FP8 or int4 or int8 ... Assuming that ARC proposed such architectures, we want to know if we can solve the challenge."

**A** [speculative, untested]:
- **Weights only** (int8/FP8/int4/FP4/binary weights, real activations): little help. The network stays continuous piecewise-linear on Gaussian inputs, and the x23-33/step region growth comes from composing 10 steps. Integer weights make exact geometry cheap and robust. Accuracy cost: the model is very sensitive (0.25% relative noise costs points), so int8/FP8 post-training rounding costs accuracy, and int4/FP4/binary weights need quantisation-aware training (QAT), likely with lower accuracy at 432 parameters.
- **Activations quantised** (finitely many hidden states): the big change. The RNN becomes a finite state machine; histories reaching the same state merge automatically (the "merge identical futures" idea built in). An exact forward dynamic programme over (state, input interval, label summary) costs about reachable states x intervals x steps. Binary activations: <= 65,536 states, ~17 input intervals each: likely solvable exactly at roughly the cost of sampling to 0.1 pp. int4: depends on how many states are reachable (unknown). int8/FP8: probably too many states unless the dynamics are very low-dimensional. Discrete inputs (int4/int8) make the whole problem finite and simpler.
- **Would ARC accept it?** By the letter yes (an exact answer has zero error); by the spirit it is debatable: deductive and arguably mechanistic, but it sidesteps the hard part (real-valued states composing over many steps), which is what real models do.
- Follow-up **T352** (owner, 13:58): QAT retraining with binary/int4/int8 activations, reachable-state census, exact DP for the binary model, and discrete-input (int4/int8) variants.

<a id="status"></a>
<a id="q6-6"></a>
### 6.6 Which of our positive results would transfer to the same architecture trained on a different task? (owner, 10-19 14:1x)

**Question (owner, condensed):** which positive or near-positive methods (lenient bar passed, or matching sampling) would
apply to a network of the same size and input trained for another task? Find about 5 tasks (some completely different from
"n-th largest") that the exact M_16,10 architecture learns to 90-95%, then test which methods carry over; one session screens
the tasks, then one session per task. The owner also passed on another agent's list (2nd-smallest, 3rd-largest, largest adjacent
|difference|, 2nd-largest |x|, largest adjacent pair sum).

**Answer (DRI, inferred until T419 and its follow-ups run):**
- *Likely to transfer (generic, no task-specific mechanism):* exact 1-D walk quadrature (H-E16: beats sampling at T = 5, ties
  at T = 10), weight-gradient locate and Hessian-guided repair of a rare failure (H-R2, H-R3), and the certified pruning + rigorous
  mass machinery (T398, T400, T412). Whether each wins depends on the task: how fragmented the regions are along lines, and
  whether the task's rare events can be pruned early (T398's late-triple obstruction is specific to rank tasks).
- *Partly transferable:* the rare-event exact walks (H-R1) are built around rank events (the line along x_p3); rank tasks reuse
  them directly, other tasks need the event and the walk line redefined.
- *Not transferable as is:* T317f (the lenient-bar pass) is derived from M_16,10's specific mechanism, so each new task needs its
  mechanism worked out again; that is itself the most interesting test.
- *On the proposed list:* keep the architecture exact, so every task must output a POSITION (one of 10 classes) from N(0,1)
  inputs; value-output tasks need a different head. '2nd-smallest' is 2nd-largest on -x for symmetric inputs, so it is a free
  control, not a new algorithm. Ties have probability 0 for continuous inputs. Longer-sequence generalisation is not ARC's
  question (accuracy on the fixed input distribution is).
- *Plan:* T419 screens candidate tasks on the exact architecture and recipe; then one session per selected task applies the
  transferable methods against both bars.
- *Follow-up (owner, 14:2x): tasks that return a PAIR of positions, trained with a multi-label loss.* DRI: yes, as a second
  tier. Keep the same 10 output units (no extra parameters), train with sigmoid/BCE or a listwise top-2 loss, read out the
  top-2 logits, and score exact set match. The walk code adapts, because the decision boundary becomes the 2nd-vs-3rd logit
  crossing (still piecewise linear). Prefer genuinely 2-D pairs (two largest, closest pair |x_i - x_j|); 'adjacent pair' is one
  position. At most 2 of T419's 5 picks may be pair tasks, so that at least 3 keep M_16,10's single-position readout and ARC's
  accuracy metric.
- *Follow-up (owner, 14:3x): 10 tasks instead of 5, one session each.* T419's pick rule is now 10 tasks >= .90, at least 6
  outside the k-th-largest family, at most 3 pair-output.

<a id="q6-7"></a>
### 6.7 Do our positive results hold for larger networks (more neurons H, more steps T, or both)? (owner, 10-19 14:4x)

**Request (owner, condensed):** most results are for ARC's M_16,10 (16 neurons, 10 steps). Scale up H, T or both, and see how far
the positive and near-positive results keep holding.

**Answer (DRI):** one lead session (T421) trains a grid of 2nd-largest RNNs with the zoo recipe beyond (16,10) (H up to 64 and
T up to 20, several seeds per point, reusing T392's H = 24 / 32 models where they match); then one session per method measures
where it stops holding, each against a pre-registered criterion: walk quadrature (H-E16: F <= 1 / <= 10), the rare-event
estimators (H-R1 and the graded rule: within 2x), locate and repair (H-R2, H-R3: a >= 10x cut at <= .05 pp), certified pruning
(survivor growth per step), and the safe-use rating (T401). Prediction [inferred]: the generic methods degrade with T faster
than with H, because each extra step multiplies the gate flips along a line; T392's chunked RNNs already show walks losing
at k = 1-2 inputs per step.

<a id="q6-8"></a>
### 6.8 How novel are our positive results, and how should we describe them? Are we rediscovering adversarial-robustness work? (owner, 10-19 14:5x; follow-ups 18:3x, 18:5x)

**Answer (DRI, from two literature passes; full verdicts, citations, phrasing and a ranked "methods to try next" list are
in [lit/novelty_review.md](lit/novelty_review.md), the one place they are kept):** partly yes. The verification
literature already has the main ingredients, so most of our results are known methods combined or applied; few parts are
new. Specifically: the certified-bound pipeline is an application of probabilistic branch and bound (Boetius, Leue &
Sutter, ICML 2025) and region enumeration with probabilities (ProbStar / StarV, Tran et al.; Kofnov et al., ICML 2025, the
closest prior to our rigorous mass bound, in 1-3 dimensions without floating-point rigour); the likely-new part is the
outward-rounded Gaussian mass of 10-D polytope cells. The walks are line sampling / preintegration made exact per line
(ExactLine, Sotoudeh & Thakur); the great-circle step is the same primitive as LIN-ESS (Gessner et al. 2020). The rare-event
walks are new mainly in being sampling-free near 1e-10 without enumerating all regions. The rare-to-common conversion of
repairs is a known kind of trade-off, newly measured here. The T431 strict pass is likely new as a result, not as a method,
and is QMC-class. Nothing found makes ARC's accuracy estimate deductive. Do not claim a "first certified rare-event
probability for a neural network". The ranked untried methods became backlog rows T436-T442.

<a id="q6-9"></a>
### 6.9 Educated guess: which positive results will hold for larger networks and other problems? (owner, 10-19 15:0x)

**Answer (DRI, a guess, before the scale and transfer data; ranked most to least likely):** (1) locate and repair
(H-R2, H-R3), which is generic, and the finding that training optimises common errors (T411); (2) rare-event estimation
within ~2x for errors far from decision boundaries (H-R1), worse where errors sit in thin near-tie slivers; (3) the
safe-use rating (T401), which inherits (2); (4) walk quadrature beating sampling, expected to hold for shallow networks and
to fade with more steps and neurons; (5) full certificates only for small networks or 'early' slices (T398); (6) the
mechanism-derived calculation (T317f) needs a new mechanism for each network; (7) the sampling-class bound always holds but
costs grow as 1/p. **Early correction:** the first scale-study points contradict guess (4) for H: at T = 10 the frozen walk
beats sampling at H = 24 and 32 (T423, interim), so the guess is kept here only as the record of what was predicted.

<a id="q6-10"></a>
### 6.10 Do our methods apply to more modern architectures? Which class of architectures keeps at least one of them? (owner, 10-19 17:1x)

**Request (owner, condensed):** change the network structure to something more modern (ideally more accurate) while
keeping weights and neurons affordable, and find the class of architectures for which at least one positive or
near-positive method still applies (softmax or other neuron types may break some of them).

**Answer (DRI, [inferred] until T432 runs):** each method needs a different property. *Exact line walks and the
certified bound* need the network to be piecewise linear along any line (ReLU, leaky ReLU, maxout, hard-tanh /
hard-sigmoid gates, max-pooling, residual connections, linear state-space recurrences); products such as attention scores
make it piecewise polynomial, so line walks stay exact with polynomial roots but the polytope mass machinery does not;
softmax, sigmoid, tanh, GELU and layer norm need numerical integration along lines (no longer exact) and CROWN-style
linear bounds for certificates. The exact radial integral also needs no biases (positive homogeneity). *Locating and
repairing a rare failure* only needs gradients, so it should apply to any differentiable architecture. *The rare-event
estimators* are tied to the task's event, not the architecture. Plan: T432 maps methods x architecture features, trains
4-6 affordable modern candidates on the same task (bias ReLU RNN, stacked ReLU RNN, gated RNN with hard and smooth gates,
a linear state-space model with ReLU readout, a tiny transformer with ReLU vs softmax attention, a 1-D conv net) and tests
which methods carry over against both bars.

<a id="q6-11"></a>
### 6.11 Would our methods apply to a mini language model (next-token prediction), possibly with a modified mini self-attention? (owner, 10-19 17:2x)

**Request (owner, condensed):** train a small language model on a mini language and test whether any of our methods
apply; it may have a mini self-attention layer, changed if needed (from the literature or our own ideas) so that at
least one, ideally many, of our positive methods apply; split the work into parallel streams via a lead.

**Answer (DRI, [inferred] until T433 runs):** two changes matter. (1) *Inputs become discrete tokens*, so the input
distribution is over sequences, not Gaussian vectors; exact enumeration is possible for short contexts (T352 found exact
dynamic programming beats sampling by orders of magnitude on discrete inputs), so the language must make V^L large enough
that sampling is the real baseline. (2) *Softmax attention* is not piecewise linear; variants that keep our exact tools are
hardmax (argmax) attention, ReLU or linear attention (piecewise polynomial), and position-only attention (linear mixing).
Locate and repair (gradients) and the safe-use rating should transfer directly to a rare bad output such as an
ungrammatical token. Plan: a lead (T433) builds the shared language, data and training harness and proposes parallel
streams; the DRI spawns one session per stream (attention variants, estimators for discrete inputs, rare bad outputs,
certified bounds).

## 7. Status timeline

| Time (UTC) | Owner asked | Snapshot |
|---|---|---|
| 10-17 22:55 | "…tell me about the next three tasks that are at the highest priority." | Unsolved: deterministic methods lose by 37× to 4000× on M_16,10; the cell recipe wins only on small models, with cost exploding along T (1.7e5 → 8.7e8 → 2.2e11 at H=8, T=3..5). Top three: T187/T188 (fading memory, same-future merging), T193 (independent check of E46–E48), T189 (order statistics plus a bound). |
| 10-18 02:22 | "Status in simple terms." | Not solved; tonight's bet (exact counting) is a dead end: pieces multiply 25–30× per step, about 10 trillion pieces, about 1,000 years. The recipe fails at 5 numbers; merging, pruning, smarter starts and simplified weights failed. Odds < 10% (30–40% for a fixed-point version). About 16 of 305 Colab units used. |
| 10-18 03:11 | "Status in simple terms from the last time that you described." | Region counts confirmed independently. Coverage still needs 15–19× more pieces per step, and near-tie families form a continuum (T218). Pruning with a bound and the weight bound failed. The best deterministic method beats sampling only on the clean model (1.1–1.3×). Odds ≤ 5%; report drafted; about 20 units used. |
| 10-18 03:14 | "Elaborate in simple terms the last three that is still finishing." | (1) 300B-sample answer key, then 95.3117%, ±0.004 → ±0.00004 pp; (2) step-6 coverage growth (18.6, 16.1, 14.9 so far); (3) independent repeat of the coverage measurement. Later: the key landed at 95.31175% ± 0.00004; coverage confirmed (about 16×/step to 4 steps); the step-6 estimate (about 30×) was found censored, and T270 gave ≥ 24.5×. |
| 10-18 12:13 | "Status in simple terms." | Strict (F ≤ 1) unsolved; lenient passed by T317e, F ≈ 5 on fresh HP3, retrains only, not for ARC yet. Strict needs about 5× less error or cost: the cost floor of about 2.2k FPE gives F ≈ 1.9 (T326 corrected), and the bias is about +0.4 pp. Fable round 5: tops out at F ≈ 1.2–1.7, about 5%. Five cheap tests running. Later: T317f F 2.63 on HP4. |

---

<a id="directives"></a>
## 8. Owner directives (verbatim)

All times are UTC. Each directive is followed by a one-line note on what was done.

- **2026-10-17 22:55:** "I'm assigning you as the new DRI of this project. Please read the handoff document, the index, and all the relevant files, like the backlog, because you're also the product manager and project manager. You're also the scientist. So you need to decide what task we want to run next. For now, just learn and tell me about the next three tasks that are at the highest priority."
- **23:01:** "Which task required GPU? I have a Google Colab with 300 compute units. I want to set that up first. Create a new child session that takes the tasks. The new child session should have its own Claude Code session and its own sandbox. I believe you have the tooling to do that. Then I'll work with that child session to set up the Google Colab, because Google Colab has a CLI access. They should just tell me what they want and I will help them with that." Done: T192 child.
- <a id="d-2307"></a> **23:07:** "I'm helping the child session that works on the GPU to set up the Google Colab CLI and log in. After that, you are the DRI and you should manage it. It should not wait on me, and you should never wait on me anymore. Start child sessions instead of sub agents if you're asking them to do any experiment, so they have their own compute. If the task does not require any compute, for example, for researching about the method or such, you can spawn your own subagents. Try to keep your context window clean so you don't do the tasks and you delegate as much as possible. But you need to be a really good scientist that understands what's going on so you can prioritize the tasks and manage the sessions."
- **23:12:** "Make sure you have a good understanding of the problem, what we have tried, what was successful and what wasn't, and any context that you need. If you have any questions for the previous DRI, gather all your questions, and at the end of your current work, tell me those questions all in one message, and I will ask the old DRI." See 5.2.
- **2026-10-18 00:26:** "m16/approaches/crossover_compute/DRI_HANDOFF.md — check what the Codex agent managers have written for you and make sure they're under your guidance." See 5.3.
- **00:37 (owner relays information):** the previous DRI's answers, saved at [review/dri_handover/QA_2026-10-18.md](review/dri_handover/QA_2026-10-18.md).
- **00:39 (owner reports information):** a newly joined ARC member reportedly has a solution and ARC awaits a comparison. See 1.2.
- **00:42:** "Unless we have a candidate solution, we should not contact Arc." Recorded in T41/T203. The DRI still recommends no contact: the lenient result is a progress note, not a candidate solution.
- **03:22:** "Yes, go ahead." This approved the random-weight comparison (T232; 1.3).
- **03:27:** "I'd like to run the training that we just discussed. I think it's worth knowing if we run for longer with more data, whether the accuracy saturates and what it is. And if those issues that you mentioned that are training outcomes might be gone if we have the right data for it." Done: T234 (2.1).
- **03:28:** "Ask a subagent to figure out why Google reclaims the node and if we can prevent that." See 5.4.
- **04:51:** "Agent speeds are much faster than human speeds, so I would doubt that writing Triton or CUDA kernels would take that long. I think the agent would be able to do that very quickly. If the subagent [model name omitted] is not doing it, we can ask a Codex agent to do it. Because sometimes [model name omitted] avoids writing GPU kernels or such." The kernel was made mandatory for T192, with Codex backup T261.
- **04:52:** "I think Codex has extra capacity. You can just ask a Codex agent to also write the GPU and Triton kernels." T261 was set Ready, and both kernels were delivered (5.6).
- **04:58:** "Can you start a Fable model as a subagent with the effort X high to review the report and the experiments and think critically about the problem and brainstorm some ideas on how we might be able to solve it? Or some ideas that would help us be more precise and have more data of where to look for." Fable reviews ran in five rounds: [review/fable_review/](review/fable_review/README.md) (05:16), then round 2 at 07:19, round 3 at 09:02, round 4 at 10:38 (which found the walk-cost saving behind T317) and round 5 at 12:13 (pessimistic on strict).
- **05:00:** "Ask the Codex to do the same. First, tell the Codex manager that we need an astro [Astra] model at X high." Done: T262, [review/codex_astro_review/](review/codex_astro_review/README.md). Astra round 2 landed at 07:53; round 3 never arrived (Codex silent from about 08:30).
- **05:11:** "Try a formula over the top values and their gaps." See 3.5 (T269).
- <a id="d-0519"></a> **05:19:** "Okay, then let's also track if we can get up to 10x worse than sampling, because that seemed to be also something that Arc might accept." Done: Directive 8, every result reported against the strict (F ≤ 1) and lenient (F ≤ 10) bars; later amended (fixed budgets, then own-cost scoring with a 1k-FPE floor, and a strict-bar wording fix).
- **05:42:** "[.srt transcript of Jacob Hilton's talk] This is a talk from about a month ago, I think."
- **05:44:** "[.txt transcript] Read this in full yourself." See 1.7.
- **06:56:** "If you're stuck, ask for Fable and Astra help. Do not stop until we have solved the challenge. Think as a scientist." This led to the 07:03 reframing (cheap regime, own-cost scoring, sealed retrain panels), Fable/Astra round 2 and later rounds, and the T286–T317 lineage that reached the lenient bar.
- **13:12:** "When should we have resolved T340? I think this is the part that we should focus. I mean, your answer to my last question, about questions regarding the network that our approach would be able to better answer than the sampling method. I asked many questions and you provided very informative answers. I would like to make sure that these are all written down, so it's easy for future agents to read them and find them. Could you make sure that all these points are captured somewhere, including the questions that I asked?" T340 was made top priority with T344 added (4.5), and this document was written.
- **13:23:** "Add this small feed-forward network idea to the backlog. It's low priority." Done: backlog row **T345**, low priority (6.1).
- **13:40:** "And exploring the ideas about clever exact methods. Or exact computation as a component. We need to discuss that with Fable and Astra. Write down any idea that you have regarding it. The whole thing is as step two, brainstorm. This is the new task at the back of the backlog with low priority." Done: backlog row **T346** (low priority) at the end of the backlog; the DRI's seed ideas A-J with kill tests are in [m16/approaches/exact_brainstorm/](m16/approaches/exact_brainstorm/README.md); step 2 is independent Fable and Astra reviews (6.2).
- **13:58:** "Yes, please add the cheap way to test by retraining M1610 with binary and int for activations using quantization aware training. We can also consider the case that the inputs are also discrete, like int 4, int 8." Done: backlog row **T352** (after the rare-event lane); see 6.5.
- <a id="d-1727"></a> **17:27:** "Let's stop using Fable from now on." Done: no further Fable reviews or sub-agents; recorded in TASKS (owner directive) and DRI_HANDBOOK. Existing Fable reviews (r2-r6, T347 note, T346 brainstorm) stay in the record.
- **23:3x (after the DRI listed options A-D with costs):** "C" — run T370 (fix the rare-event closure; CPU, est. 10-100 CPU-hours plus <= 3 Colab units, with a stop rule after stage 1 on seen models) and T352 (quantised variants; est. 5-20 Colab units, cap 20) in parallel. Done: both reserved and assigned (TASKS In progress). The ARC progress note stays undecided; no contact.
- <a id="d-234x"></a> **23:4x:** "You should not ever contact them. I will contact them when I deem it's the right time. Also in parallel, we need to work on the other ideas that we have in the backlog that were lower priority. For example, the quantization. We have many of them that we can run each in parallel using a separate session. So each have their own sandbox." Done: recorded in TASKS (directives; T41 closed; T203 (d) for the owner only), STATUS, DRI_HANDBOOK. Agents never contact ARC. Four backlog ideas started in their own new sessions: T345 (feed-forward control), T371 (rare events on weight-noised copies), T372 (exact weight-gradient of a rare-event probability), T373 (certified rare-event probability by event-cone enumeration, kill test); T352 (quantisation) was already running in the T192 child's own session.
- **23:5x:** "You also mentioned that if the network receives all 10 numbers at the same time as input, the problem would become much easier because we can use shallower network. So I want a network that has the same accuracy, about 95% for this problem, that takes all the numbers all at once. We should try to keep it as simple as possible, so hopefully we can solve the challenge for that new architecture. We might need to add a few layers until it reaches this accuracy, but hopefully it won't be similar to the RNN, which you mentioned is analogous to 10-layer deep." Done: T345 retargeted (simplest feed-forward network with accuracy >= .95, then the challenge on it); the T345 child (session_01Na5o6K4TXWJmNtj7N5f76X) was told before its pre-registration was pushed.
- **10-19 00:1x:** "Start a sub-agent or a new session that creates a interesting HTML interactive tutorial in the style of distill.pub and the post from Nick about neural networks are clean or similar, or Anthropic interpretability team type of interactive papers, in which we start as the simplest form of this problem, and then build upon it until we get to the results that we got. We should definitely start the article with saying our results and what we have achieved, and also mention about some of the theoretical foundations that you discussed with me, like what the Cristiano conjecture was. Also, regarding, you mentioned about some theoretical foundations like about complexity theory, like this problem is p-hard, or it's hard, or it's in some like complexity theoretical stuff that you mentioned. So we want to mention about those two, and but the main of this tutorial paper should be to start with the simplest form of this problem and talk about different dimensions that it has. For example, we have tried with a less number of neurons, we have tried with less T, we have tried we are trying with using quantization forms like the binary int 4, int 8, FP4 and FP8, and then if there are any other simplification that we can do. Also, we talked about like different other simpler architectures for the same problem of finding the second largest number that we are working on. So we want to build upon all of these and then finally get to the one that that arc is asking, and show how the method worked, and like the number of regions, and you know, the brute force approach, and how the network pushes the regions towards the boundary parts, and how the noise would have effect, and how seeds would have effect. Consider this to be written in a way that is for someone who doesn't have any knowledge in this area but has ML knowledge and deep learning knowledge. It should be very, very easy to understand what the paper and work is. I suggest instead of a subagent, use a new session." Done: task **T374**, run in its own new session. First draft (10-19 00:30): https://claude.ai/artifact/22zmMZh9WU6LAPLWqJmXRX (private; updated in place).
- **10-19 00:2x:** "also ask the session to spawn a subagent with Scratch context to read the tutorial and criticize any part that is not what is asked, and any part that could be improved to be simpler, easier to understand. Keep it pedagogical." Done: T374 runs a fresh-context review sub-agent after every draft and major update.
- **10-19 01:5x:** "Codex Astro Model is a much better writer. Could you add a backlog item specifically for Codex to rewrite the tutorial with better English and better sentences? Make sure that the latest version of the tutorial is inside the repo. Let me know when it's ready, so I will tell the Codex to pick it up." Done: backlog row **T382** (Ready, reserved for Codex); the latest tutorial (all of tutorial/, including its data files) copied onto the shared branch.
- **10-19 05:0x:** "Yes, please resume the ladder's tasks. I don't know why I paused them." Done: T86 + T111 (training missing ladder sizes, then T87's H4T4 failure on better-trained models) and T46 (exact ladder at H = 16: finish T = 5/6, then a hybrid exact + walk-quadrature estimator) resumed in two new sessions.
- <a id="d-1019-060x"></a> **10-19 06:0x:** "Yes, please start it and any other task that you think we can run in parallel. Anything that remained, we should just create sessions for them and let them run and calculate. vCPU times are essentially free for us. We are only paying for the tokens. So, and we also have some tokens left and some GPU time left. So if there are experiments that are helpful for us, let's run them. But prioritize the ones that require vCPU as it's essentially free." Done: T392 (chunked-input RNN, k = 1/2/5/10), T393 (compiled walk, real wall-clock cost), T394 (independent check of the all-at-once results), T386 (de-aliased deterministic rare-event gradient) in new sessions; T395 (GPU census at 1e10) to the T192 child after T352.
- <a id="d-1019-064x"></a> **10-19 06:4x:** "No need to update the tutorial for now. I'll ask you when we have everything we wanted, and we'll do one final update." Done: T374 paused at v7 (pending edits committed to its branch, not published); check-ins no longer send tutorial updates.
- <a id="d-1019-065x"></a> **10-19 06:5x:** "Why anything would I tell you, you update in 3 files? I think you should just update in 1 file and have reference in the others. Do a bit of cleanup. It seems there are many duplicates and the repo is a bit unorganized, or like the docs are not organized as we like." Done: task **T396** (one source of truth per kind of information; plan in [docs/CLEANUP_PLAN.md](docs/CLEANUP_PLAN.md)). From now on a directive is recorded only here, with a one-line standing rule in [AGENTS.md](AGENTS.md#standing-rules) when it binds agents.
- <a id="d-1019-070x"></a> **10-19 07:0x:** "I think guaranteed upper bound on rare mistake is super important. We need to give it more resources. If GPU helps it, let it run its work on GPU. If we need more number of CPUs than 4, I think the collab agent can get more CPUs. I might be wrong, but we can check. And we can also spawn multiple sessions all helping this one, and they can all communicate using the tools that is available. Basically, it would be nested sessions, I guess. So the session can spawn multiple nests at once, so we can go faster, or we can spend more compute. Also, 210 million doesn't seem unreachable. What do you think? Even for that many pieces, if we can give exact or guaranteed results, I think that would be fantastic. Am I missing something?" Done: the certified-bound lane is the top compute priority: T398 (lead: tighter bounds, choice of a rarer threshold, shard plan), T399 (GPU batched bound propagation on Colab, before T395), T400 (rigorous per-region mass bounds and exact-arithmetic verification of pruning certificates); worker sessions to shard the 720 position triples once the per-node cost is measured. Answer and caveats in the DRI reply (missing pieces: the 679 no-hit triples, per-region mass bounds, exact verification, and choosing a rarer event where the certificate matters).
- <a id="d-1019-071x"></a> **10-19 07:1x:** owner asked to document the answer on ARC's ultimate goal and proposed the bridge-engineering framing (verbatim in [4.10](#q4-10)). Done: [4.9](#q4-9) and [4.10](#q4-10); demonstrations **T401** (safe-use rating) and **T402** (targeted repair).
- <a id="d-1019-152x"></a> **10-19 15:2x:** "For some of our work, we abandoned them because we said we don't have enough compute, or it's just too much compute. Keep in mind that this is preliminary work, and we want to be very explicit and do a rough estimation about the compute needed for some of these, even if the method as is would not work and require more agent tokens on it. We can also do a rough estimation of how much token might be required for solving that problem. So at the end of the day, when we are publishing, we want to say, okay, we think that there are these options that we haven't explored because lack of compute. But for example, if we have like these large cluster, or like if we had like, I don't know, $10 billion to buy this much compute or this much tokens, we could have solved it. So we want to write those down because it's a safety case, and sometimes safety problems are so important that it makes sense to pour in a lot of money, and 5 years from now, you know that the compute is gonna be much cheaper, and each chip would have a much higher compute capability, and also in terms of the tokens, the agents are gonna be way smarter than what they are right now. So although the problem might not be solvable just right now, given the resources that we have, it might be solvable using more compute later down the road. So we just want to like write them down if we have some idea of if we are given a lot of more tokens and a lot of compute, are we gonna be able to solve it or not." Done: [docs/COMPUTE_FRONTIER.md](docs/COMPUTE_FRONTIER.md) written (10 approaches; summary table; what $1M / $100M / $10B would settle) (per approach: compute and agent effort needed, cost today and in 5 years, compute- vs method-limited); standing rule added to AGENTS.md.
- <a id="d-1019-144x"></a> **10-19 14:4x:** "It's okay if we spend a little bit of GPU time for it, but it's better to use free sandbox vCPUs, and it's okay if we parallelize." Done: passed to the scale study (T421, GPU cap 15 units) and the task screen (T419, cap 10); consistent with the CPU-first standing rule.
- <a id="d-1019-163x"></a> **10-19 16:3x:** "Great explanations, great findings. Just make sure that everything that we have discussed is properly documented. And as we discussed, no duplication in documentation." Done: Q&A 4.11, 4.12, 6.9; README ID glossary; INDEX rows for the running tasks' interim results (T86, T392, T419, T421, T423, T427) and updates to T370 and T398; STATUS rewritten in place in plain words.
- <a id="d-1019-223x"></a> **10-19 22:3x:** "You are running out of credit. This might be your last message. Please update the docs, update the tutorial, and answer this question." Done: Q&A [4.15](#q4-15); STATUS rewritten; DRI handoff note (DRI_HANDBOOK section 12); final tutorial update sent to T374 (this lifts the 06:4x pause).

- <a id="d-1020-takeover"></a> **10-20, temporary DRI takeover (owner, verbatim):** "How are the agents doing? The Claude DRI ran out of credit and will be off for the next three days. You need to take over and drive the project forward. See where we are standing and see if there are tasks that you can help and you can push forward. If all the tasks are in progress from the Claude side, we need to think if it works to do it here or just wait for the Claude credit to reset on Friday afternoon. It's your call, but we want to push the project forward. We don't want to spend a massive number of tokens, but if there are things that you think are helpful for the project, it would be nice to do. I suggest you to read what the DRI has written in terms of the overall doc, the Q&A that I have with it, the index, the tutorial, latest version of it, latest results that we have, so you have a good holistic understanding of where we are standing right now and see if you can manage our sub-agents to do better." **Action/answer:** Codex root assumes acting DRI coordination, preserves independent-review requirements, audits published progress with two bounded reviewers, and publishes the [takeover decisions](docs/DRI_TAKEOVER_2026-10-20.md). T458–T461 completed; next priorities T462–T464. No assumption that Claude's listed jobs remain alive and no duplicate long runs.

- <a id="d-1020-cpu"></a> **10-20, CPU authorization and literature review (owner, verbatim):** "It's okay if you use the CPUs that we have as much as you can. If you can spawn new sessions, that would be great. Please also read the outcome of the literature review that we just." **Action:** latest available review v2 read; [seven recommendations reconciled](docs/LITERATURE_ACTIONS_2026-10-20.md). Separate compute-session allocation is not exposed here; T462/T463 use explicitly authorized local CPU under existing concurrency/memory limits, with published protocols and independent review.


<a id="d-1020-readable-docs"></a>
### Documentation for new readers (2026-10-20)

**Owner:** "we need to keep the readme and results ledger in simple terms so someone who's not familiar with the project understands it."

**Application:** Lead the README and INDEX with the question, the main results, their practical meaning and remaining uncertainties in plain language. Explain technical terms before using them. Keep measurements, identifiers and historical evidence available in expandable sections. Present the two full-challenge candidates prominently without implying ARC has accepted them.


<a id="d-1021-whitebox-start"></a>
### Start White-Box challenge work (2026-10-21)

**Owner (verbatim):** "checkout whitebox_challenge_assessment.md and start work on a solution"; after interruption, "continue". The owner then supplied public-code leads and authorized review: "you can assign a subagent to check these".

**Action:** Claimed and published T473, built and ran a bounded metered control-variate pilot, and assigned an independent read-only public-source review. Results and limitations live in [INDEX T473](INDEX.md); source pins and implementation caveats in [the review](m16/approaches/whitebox_pilot/public_sources.md). T474 records the public cumulant-baseline follow-up. No organizer contact, registration or challenge submission.


<a id="d-1021-whitebox-continue"></a>
### White-Box baseline continuation and standing (2026-10-21)

**Owner (verbatim):** "ok continue". During T474: "any estimate of where we're standing in https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards?round=phase-2 ?" and "Do we not have enough memory?"

**Answer/action:** T474 claimed and published. Explained that no submission/rank exists and the first pilot was far above leading scores. Explained the active constraint was the standing per-process memory limit, not a demonstrated machine-wide shortage. Implemented an independently source-audited memory-only port of public V25 and measured it against pinned public references; updated comparison and limitations live in [INDEX T474](INDEX.md). The memory limit was not waived. T475 records the stronger-public-method follow-up.

<a id="q-1021-whitebox-transfer"></a>
### Have we applied all the earlier project's knowledge? (2026-10-21)

**Owner (verbatim):** "anything we didn't get to optimize? any ideas missed?" and "Have we used all of our knowledge regarding how we did solve the previous challenge and all the techniques and everything that we learned?"

**Answer/action:** No. T473 tested one analytic-mean control variate; T474 reproduced a public cumulant baseline with a memory-only rewrite. We reused cost accounting, bounded memory, independent checks and limited development/evaluation separation, but have not exhausted our prior methods. The earlier project established scoped successes and full-challenge candidates, not an ARC-accepted general solution. The strongest missing transfers are approximation-by-approximation error attribution with paired ablations (including cancellation), bounded exact-line/circle feasibility, and function-preserving invariance plus final-output-sensitive allocation. These are queued as T476–T478 in [TASKS](TASKS.md), alongside stronger public-baseline work T475. No new experiment or transferred performance claim is made. Task-specific order-statistic formulas, weight repairs and rare-event certificates do not directly estimate random-network activation means. Four public reference rows are not a sealed test; broader evaluation and a genuinely fresh final panel remain necessary before claiming general gains.


<a id="d-1021-submission-sprint"></a>
### First submissions and deadline sprint (2026-10-21)

**Owner (verbatim):** "Let's make our first submission. Could you do that?"; "You can also use my browser."; "Spawn subagents to try other methods that we have."; "We want to get to the best solution in the next 22 minutes and submit them."; "We can make 10 submissions within these 22 minutes."

**Memory authorization (verbatim):** "My machine has a lot of free memory available. Would that help you?" Following a request to raise the local cap to 8 GB/process and 16 GB total: "do your best", then "Use whatever you can use. No limit."

**Application:** Owner authorizes this challenge's submission uploads and a bounded deadline sprint, superseding the general no-contact restriction for these entries only. No forum/email outreach. Owner waives the old local memory ceiling for this sprint; still respect actual available machine memory and the remote grader's 8 GB submission-process cap. T475/T476/T477 run in parallel under published claims; root handles T479 packaging, validation and uploads. Official receipts and results belong in INDEX T479 and its linked artifact manifest.
