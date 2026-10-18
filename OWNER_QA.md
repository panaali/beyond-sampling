# Owner questions and answers (DRI session, 2026-10-17/18)

This file collects every substantive question the project owner asked the DRI between 2026-10-17 22:55 and 2026-10-18 13:24 UTC, with the DRI's answer to each. It was built from the session transcript. The owner's words are quoted verbatim, with light typo fixes only. The answers are condensed paraphrases, not quotes, but they keep every number and caveat. **Each answer is as of its timestamp**, and most results quoted in it were unverified when given. Where a later result in the same session, or the registry in [INDEX.md](INDEX.md) (shared branch `ccr-6a61f25c-wn80ii`, commits `60ace4bd`/`f2b23d41`, 13:14–13:24 UTC), changed an answer, a **Later updates** note follows it. Task IDs (T-numbers) point to rows in INDEX.md and [TASKS.md](TASKS.md). The owner's standing instructions are collected word for word at the end, in [Owner directives](#directives).

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

**Later updates:**
- 00:24: T192 found that the recipe loses at (8,5) seed 2 (MSE ratio 3.4–4.6), and that exact enumeration costs only 1.2–1.5× the recipe from T=4 on.
- 00:37: the 4·T! start is not needed for exact enumeration, but it explains only about 10% of the growth ([review/chamber_split/](review/chamber_split/)).
- **The exact route is dead** (H-E13).
  - Region counts of 34 / 792 / 21,032 for k = 2..4 are confirmed independently.
  - T270 found at least 15.5M regions at 6 steps, so growth from step 5 to 6 is at least 24.5×. That is an exact lower bound. The 31–33× estimate is biased low by the T316 geometry bug.
- T195 pruning is negative: it saves only constant factors.
- **Lenient bar reached on retrained models:**
  - T317c/T317e score F 5.07 [3.41, 7.03] on fresh HP3 (H-L1).
  - T317f scores F 2.63 [1.93, 3.39] on fresh HP4.
  - Both fail on noisy copies.
- **The strict bar is still unsolved.** Fable's round-5 review gives about 5% odds within 3 months.

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

**Later updates:**
- 00:42: the owner ruled "Unless we have a candidate solution, we should not contact Arc." The draft was withdrawn, and T41/T203 record the ruling.
- Hilton's talk from about a month earlier calls the challenge "completely open" (1.7), so any ARC solution would be from the last month.
- Fable round 1 put 25–35% odds on ARC's reported solution meeting the strict bar.

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

**Later updates:**
- T232 (03:56): the trained network has as many linear pieces as random networks of the same size: 21,032 after 4 steps, against 19,000–26,000. The difference lies in what the pieces compute: 95% accuracy against about 10%.
- T271's activity-matched random control found 0.78×. The independent T280 found 0.8–1.0×, now confirmed for k ≤ 4 (H-E14).
- Hilton's talk says ARC (Mike Winer) did match sampling on randomly initialised networks. Our earlier adaptation of ARC's wide-random-MLP method (E35) was off by 41–100 pp on M_16,10.

<a id="q1-4"></a>
### 1.4 Would a 100%-accurate network be a solution ARC accepts?

**Q** (2026-10-18 03:35 UTC): "If the accuracy is 100% for a network, do you think we have a solution that Arc would accept?"

**A:** No, for three reasons.
1. **It would be a different network.** The challenge is to estimate M_16,10's accuracy (95.31%), not to build a better network.
2. **At 100% the challenge is trivial.** Sampling would also be exactly right, so a calculation could only tie. The challenge is meaningful only for accuracies strictly between 0% and 100%.
3. **ARC also tests noisy copies.** Noise lowers M_16,10's accuracy to roughly 45–94%, so even a perfect network would stop being exact once noised.

100% is probably not reachable with this design: trained networks top out around 99.95% even for 3 numbers. The idea does connect to understanding. A network provably 100% correct would be strong understanding, and "ideal network plus deviation" was already tried (T37, a handcrafted reference plus residual). It missed by 22–48 pp, because the deviation is the messy part.

**Later updates:**
- Hilton's talk: a hand-built network with about 21 neurons gets 100% (or 100 minus ε). With 16 neurons "I expect you can do better", because training gets stuck in local minima.
- The training study (T234) found trained 16-neuron networks above 95.3%; see 2.1.

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

**Later updates:** T277 found that the deterministic methods' rate of improvement with compute stays about at sampling's rate from 3 to 10 steps. What changes is the starting error, which grows about 1000×: the method's error is 1/1000 of sampling's at 3 steps and 1–2.5× at 10.

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

**Later updates:**
- 07:03 reframing: per the talk, ARC scores the average error over many retrained models (or noisy copies) and names no budget, so cheap budgets count. That led to the 1k-FPE floor, the "own-cost" scoring rule (Directive 8, amendment 2) and the sealed retrain panels HP2–HP4.
- 11:26: the DRI confirmed that "roughly match" does not appear in the talk. The lenient result meets our 10× bar, not ARC's stated goal.

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

**Later updates:**
- The owner asked for the training study, which ran as T234 on Colab.
- A measuring tool split M_16,10's 4.69% error into blind region 0.84%, 2nd-vs-3rd near-ties 2.86%, 1st-vs-2nd near-ties 0.72% and other 0.27%.
- Interim results at 07:07:
  - With zoo settings, mean accuracy rises with data: 92.7% at 2^30 sequences, 93.8% at 2^31, 95.5% at 2^32 (best run 97.1%).
  - Large-batch runs reached 96.9% (batch 8,192, 2^33).
  - One linear learning-rate schedule run collapsed to 51%.
- Fused-kernel validation (40 vs 40 seeds, means 91.9% vs 92.1%) showed that retrain outcomes are chaotic: same-init correlation 0.11, spread about 3 pp. M_16,10 sits near the top of 40 retrains, whose best was about 96%.
- The 1,050-model grid (about 100 runs per setting) was still running at 13:07; its final stage ends about 17:05 UTC. Hilton's talk independently expects 16 neurons can beat 95.3%.

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

**Later updates:** the complete table, unverified, was measured with 100M samples per model at 05:22 (T234):

| Numbers | Accuracy per seed | Mean |
|---|---|---|
| 2 | 99.988–99.997% | 99.991% |
| 3 | 99.939–99.951% | 99.945% |
| 4 | 99.73–99.83% | 99.78% |
| 5 | 98.72–99.25% | 99.02% |
| 6 | 97.04–98.46% | 98.00% |
| 7 (newly trained, 3 seeds) | 97.45–97.86% | 97.65% |
| 8 | 95.76–97.63% | 96.68% |
| 9 (newly trained, 3 seeds) | 95.90–97.34% | 96.73% |
| 10 | 88.02–94.85% | 91.21% |

M_16,10 (95.31%) is a separately trained model, better than every zoo seed at 10. Accuracy falls slowly up to 9 numbers and then drops at 10. The 7 and 9 results, from only 3 seeds each, are level with 8 within noise.

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

**Later updates:** T241 measured the neuron roles quantitatively (see 2.5), and T232/T239 tested the intuition (see 1.3 and 2.4).

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

**Later updates:**
- **T232:** the trained network has as many pieces as random ones, so the flood of pieces is not special to the error part.
- **T239 (04:34):** the intuition is right about where the complexity is and wrong that it is random.
  - Near ties there are 26–27× more pieces per unit of probability than far from ties.
  - Error-carrying pieces multiply 2.4–3× faster per step.
  - The near/far fragmentation ratio is 44× for the trained network against 1.3–6.6× for random ones. Near ties the trained network has 358,000 pieces per unit of probability against 32,000–170,000 for random networks; far from ties it has 2,900 against 11,000–25,000.
  - Training moves pieces onto the tie walls (H-E15).
- **T239 follow-up:** wrong pieces occur only near ties or in the blind region. At (16,4), far from ties there are 2,052 correct pieces, 31 mixed and 0 wrong. 42% of the probability sits in correct pieces that never touch an error, and 10 blocks hold 72% of the probability, but each block has 1,500–4,000 pieces.
- **T280 confirmed both findings** (scoped k ≤ 4): region count 0.8–1.0× random, and concentration at near-ties about 37× against about 5× for random.
- **Experiment 4:**
  - T240 (Codex) ended as a bounded diagnostic. Its board row reports certified-correct mass of about .068 and .059 under the gap settings tested, and makes "no large easy-set/full-cost claim".
  - T279's cone certification proved only 24% of clearly separated inputs with 10k cones even at 4 steps, so it was closed.
- **Experiment 5:** not reported in this session.

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

**Later updates (T241, core confirmed by an independent review):**
- **Causal map.**
  - Disabling any single neuron at the last step costs 21–67 pp, except n4 (about 4 pp, up to 27 at step 1) and n9 (nearly inert).
  - In the stricter same-structure swap, the comparators cost 48–54 pp, the clocks 24–35 and n2 27.
  - n4, n5 and n9 barely matter at the last step.
- **Formulas.** n1, n2, n4, n6, n7 and n8 fit at ≥ 99.9%, for example n7 ≈ ReLU(−13.2·last + 10.9·max + 2.0·second max). The clocks fit at 93–99% and n3/n13/n15 at 98.5–99.4%.
- **Replacement.** Agreement with the real network is 95.5% (96.6% in the review), against the 99% target. The accuracy gap was 0.11 pp in the original and 0.82 in the review, so that match is not robust. Reaching 99% would need formulas about 1,000× more precise.
- **Leftovers.** 78.5% of disagreements sit near ties or in the blind region (19.6% of inputs).
- **Recurrent test.** With the formulas taking over at a given step, agreement is 100% at steps 1–2, 99.0–99.95% at steps 3–7, and 98.3 / 97.8 / 95.5% at steps 8 / 9 / 10.
- **Corrections.**
  - The "delayed comparator" finding for n5/n9 is weaker than first stated.
  - The most damaging mode is the slower clock (8.37-step period), and a random direction in the same space does about as much damage.
  - The n3/n13 "special importance" claim was withdrawn.
- **Compression.** T260 (Codex) found that the formulas (about 50k bits) are longer than the weights themselves (about 35k bits).

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

**Later updates:**
- T239 and T280 sharpened point 3: the band is a trained "fingerprint", finer than a random network's near ties and coarser elsewhere (see 2.7).
- Fable round 1 found that about 14% of the error sits in a soft tail away from ties, so "errors only near ties" was overstated. Only the fully wrong pieces are confined to ties.

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

**Later updates:**
- T280 (08:14) confirmed the region-count and tie-concentration findings (scoped to 4 steps).
- T292 found that weaker retrains differ mainly by a widening of the soft tail away from the closest ties: 60% of the model-to-model differences, spread thinly over many position patterns.
- T337 found that weight noise creates errors far from ties (see 3.12).

---

## 3. Our methods and results

<a id="q3-1"></a>
### 3.1 "Better than sampling only on the clean model"

**Q** (2026-10-18 04:38 UTC): "In number one, when we say it's better than sampling only on clean model, could you elaborate a bit more? Be super concise and talk in simple English, simple terms."

**A:** ARC tests a method on the clean model (M_16,10 as trained) and on noisy copies (the same weights plus small random changes), and it has to beat random testing on both. Our best calculation method (T223) had 10–25% smaller error than random testing on the clean model, but was no better on the noisy copies, and 3–6× worse on the full set of 30 models.

It isn't a real solution, for two reasons:
1. It only works on the one network it was tuned around.
2. It relies on a fixed set of test points, which ARC would likely count as disguised testing.

**Later updates:** on the sealed panels the fixed-point averaging method (v5q, inductive) is about equal to sampling:
- HP1 at 16k FPE: F 0.9–1.7 across groups.
- HP2 retrains at 1k–64k: F 1.06–3.5.

T271 found that the Sobol method improves at exactly sampling's rate and never pulls ahead. It is not an ARC candidate (see 3.2).

<a id="q3-2"></a>
### 3.2 The fixed set of test points

**Q** (2026-10-18 04:39 UTC): "Concisely elaborate the second point about fixed set of test points."

**A:** Plain testing runs the network on random inputs and counts how often it is right. Our method runs it on a fixed, evenly spread list (a Sobol grid) and averages. It is smarter about where the points go, but it still checks one input at a time and never uses the weights to reason about the answer. ARC's rule forbids random or pseudorandom inputs; a fixed grid isn't random, but it does the same job, so ARC would likely see it as testing in disguise. A real solution computes accuracy from how the network works.

**Later updates:**
- Hilton's talk confirms this reading (see 1.7): no sampling even on a subset, and no extrapolating from observed behaviour.
- T263 lists repeated shuffled-Sobol averaging as sampling.

<a id="q3-3"></a>
### 3.3 Our solution for 4 numbers at 16 neurons

**Q** (2026-10-18 05:08 UTC): "Remind me, what was our solution for 4 numbers at 16 neurons? Be very concise."

**A:** It is "recipe108", the same recipe used on the smaller models:
1. Split the input space into cells by input order.
2. Follow each neuron's on/off switch step by step, splitting only when a weight-derived bound says a switch could change the answer.
3. Compute each cell's error exactly as a Gaussian cone mass, with no random inputs.
4. Add up the errors.

At (16,4) it beat sampling on all 5 seeds and 4 noisy copies: worst MSE ratio 0.035, about 30× better, at about 3 billion operations per model. This is unverified, it costs nearly as much as exact enumeration, and from 5 numbers on it loses.

**Later updates:** INDEX T194 lists the (16,4) ratios as ≤ 0.016 in the older, pre-T196 cost units, with cells about equal to regions (0.80–0.96) and recipe 2.7–3.0e9 ops against 4.2–4.9e9 for exact enumeration.

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

**Later updates:**
- **T269 was negative (05:46).**
  - The exact integral is accurate to < 1e-5.
  - A per-model fit matches its own model only tautologically.
  - Formulas don't transfer: they are off by 0.4–19.5 pp on other models.
  - As a control variate the gain is only 1.15–1.45×, against 4× pre-registered.
  - Phase 2 was not run.
- **The idea's deductive descendant became the lenient result.** It walks the network exactly along the two gap directions (g12, g23) of each top-three placement:
  - T292: the top-three closure, 0.51 pp floor.
  - T300: F 6.0 on DEV2 and 9.7 on HP2, partly inductive.
  - T302: the exact walk, 21×.
  - T304: 11.7×.
  - T317 family: HP2 9.57, T317b 7.75, T317c 6.72, then HP3 5.07 (T317c–e), then **T317f HP4 F 2.63 [1.93, 3.39]**.

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

**Later updates:** the probe family is **dead (verified, scoped)** (E7 in INDEX):
- T271: its error does not shrink with compute. The 0.74 → 0.25 sequence compared three versions, not three budgets.
- T274a: v5a narrowly failed the pre-registered 4k line (upper CI 12.6). The fixed-point control (v5q) passes, but it is inductive.
- Hilton's talk makes the probe borderline inductive anyway.

The lenient bar was reached instead by the walk family (see 3.5 and 3.8).

<a id="q3-7"></a>
### 3.7 Where is our hope now?

**Q** (2026-10-18 05:35 UTC): "Where is that our hope now?"

**A:** Three running tests plus one unknown:
1. **Lenient bar with a mechanism-based method, maybe 25–40%.** Either the probe (about 23× on the clean model; if its rate is real (T271), v5 (T274) might reach about 6×, and it must stop breaking on noisy copies) or the gap formula (T269).
2. **Strict bar by exact counting, about 5–10%.** T270 computes the step-6 masses: growth below about 7 per step could make it feasible, while about 12 or more kills it.
3. **Smoothing the tie band (T272), a long shot.** If freezing switches inside the band changes the answer by < 0.02 pp, the band becomes smooth.
4. **The unknown: ARC's rules.** At ARC's small-model noise level, M_16,10's accuracy drops to 12–28%, which is a different problem. We don't know their noise level, budgets, or whether fixed grids count.

**Later updates:**
- Item 1: the probe is dead and T269 negative, but the walk family (T317) passed the lenient bar on sealed retrains: HP3 F 5.07, HP4 F 2.63.
- Item 2: the exact route is dead. T270 found growth ≥ 24.5×/step at step 6 (kill line 12).
- Item 3: T272 was a kill: freezing shifts the estimate by 2–21 pp.
- Item 4: HP1 measured ARC's σ 0.05 at 10 steps as 11–34% accuracy, close to chance.
- **Strict-bar arithmetic.**
  - T326: cost cannot drop below about 2.2k FPE. The DRI corrected the child's arithmetic: at the current error that floor gives F ≈ 1.9, not about 6, because F scales linearly with cost.
  - Fable round 5 estimates the approach tops out at F ≈ 1.2–1.7 even if every fix works (about 5% odds).
  - T333: the exact two-dimensional combination would add ≥ 13–19k FPE.
  - T339: the spread closure brought the model-to-model scatter near strict level (F ≈ 1.06 from variance alone) but exposed a bias of about +0.76 pp. **Withdrawn 14:15:** an independent check on 100 fresh DEV3 models (T343) did not replicate the scatter reduction (sd ratio .912 vs .68 on DEV2); it was specific to the development models.

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

**Later updates (12:33):** **T317f** (T317e plus a declared early-stop rule) scored **F 2.63 [1.93, 3.39]** on fresh sealed HP4, scored once. RMS is 0.75 pp, bias +0.44, at about 3,300 FPE; under stricter counts F is 2.67–2.98, and the plain-sampling control is about 1. Sampling still has about 2.6× lower error, so strict fails. HP4 is now spent and HP5 is held in reserve (H-L1).

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

**Later updates:**
- T333: every free way to combine the two walks is far worse, and the exact 2-D answer adds ≥ 13–19k FPE, so the product rule cannot be removed cheaply.
- T327: a fixed alternating spread of the 7 low inputs cuts the closure floor from 0.48 to 0.34 pp (0.26 with the reverse pattern) at no cost.
- T339 then exposed a bias of about +0.76 pp; nothing was frozen.

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

**Later updates:** see 3.12 for the breakdown point (between 0.1% and 0.25%).

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

**Later updates:**
- T327's fixed spread pattern reached 0.26 pp, so T338's go line was raised to beat 0.255.
- **T338: NO GO (13:02).** The weight-derived choice scored 0.41 pp, no better than a simple "most recent inputs" rule, and it lost to the fixed spread's 0.26. Lead: the weights rank the *earliest* low inputs as most important.
- **T341 refuted the running-max reading (13:12).** Giving the closure the true running maximum overestimates accuracy. The reason the spread works is open again (speculative: a sign-alternating, period-2 recurrent mode).

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

**Later updates:** T338 was NO GO and T341 refuted the running-max reading (see 4.1). So far no estimator improvement has come from neuron-level understanding.

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

**Later updates:**
- 13:05: T337 showed that noise errors are exactly these far-from-tie islands, so T340 also checks whether the walks capture them.
- 13:14: T340 is the top priority, and T344 adds certified upper bounds on the same events.
- **13:17, first preliminary results** (judged against a smaller sampling check; final billion-sample truths were due about 45 minutes later):
  - **T340:** the cheap version (with the usual shortcuts) misses the far-from-tie error islands entirely, by factors of 80 up to 10¹². A careful version that walks each line at many more points, costing 200,000–300,000 FPE, lands at 0.6–0.9× the true probability on two retrains but only 0.24–0.33× on the main model. The main model's shortfall comes from the walk's simplified treatment of the off-line directions; the closure keeps the rate within about 30%. At that cost sampling would see only a handful of events, if any, so even a factor-of-3 estimate is in the same range. Whether it beats sampling waits on the final truths.
  - **T344:** a fully rigorous guarantee is much harder than expected. Proofs work only on tiny input boxes, because the 10 steps amplify uncertainty, and the only rigorous bound so far is the obvious one, 250× to 24,000× too loose. Next: test whether exact enumeration is feasible in the far-gap area, falling back to a bound that is rigorous given the declared closure. T340's scan confirms the rare far-gap errors exist; on the main model they are about 10,000× rarer than the obvious bound allows.

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

**Later updates:** T344 (certified upper bounds on T340's events) was added at 13:14 to cover point 1 for rare events. Its first finding (13:17): rigorous bounds work only on tiny boxes, and the best so far is 250×–24,000× too loose. T340's careful walks reach 0.6–0.9× of the truth on two retrains and 0.24–0.33× on the main model, at 200k–300k FPE (preliminary; see 4.3).

<a id="q4-5"></a>
### 4.5 When will T340 be resolved?

**Q** (2026-10-18 13:12 UTC): "When should we have resolved T340? I think this is the part that we should focus. I mean, your answer to my last question, about questions regarding the network that our approach would be able to better answer than the sampling method. …" (The rest of the message asks for this document; see [Owner directives](#directives).)

**A (13:14):**
- T340 was assigned at about 12:55 with a 2-hour budget, so the first result should land around **15:00 UTC**. The DRI asked for an exact ETA, and the GPU agent can do the billion-sample truth run in minutes if that is the bottleneck.
- T340 is now the top priority on the board, with a second lane:
  - **T340 (estimate):** how accurately the exact walks estimate error probabilities of 1e-4 to 1e-6, against sampling at equal compute.
  - **T344 (proof):** a guaranteed upper bound on the same probabilities, using the rigorous check built earlier, on T340's event definitions.
- This Q&A document was commissioned at the same time.

**Later updates:** preliminary T340 and T344 results arrived at 13:17 (see 4.3). The final truths were still being computed.

---

## 5. Compute, GPU and process

<a id="q5-1"></a>
### 5.1 Which task needs a GPU?

**Q** (2026-10-17 23:01 UTC): "Which task required GPU? I have a Google Colab with 300 compute units. I want to set that up first. …" (The rest is in [Owner directives](#directives).)

**A:** **T192**, the compiled/GPU exact reference and cost law. It moves exact calculation and the recipe onto GPU code, measures cost growth at (8,5) and (8,6), and decides whether an exact answer key for the full network is affordable. That last part is T5, which was blocked on compute approval and is included. Child session `session_016BGedpds19gq95yjWmDMn5`, branch `claude/t192-gpu-cost-law`. Its rules:
- talk to the owner before coding;
- reproduce the CPU results first;
- ask before spending more than about 30 units on any one stage;
- report its spending.

At 23:10 its budget became 150 of 300 units without asking, with a stop if any stage passes 40.

**Later updates:**
- T192 produced the answer key: 95.31175% ± 0.00004 pp from 300 billion seeded samples.
- It found the (8,5) recipe failure, built the fused training kernel and ran T270 and HP1–HP5.
- At 04:14, 27.8 of 305 units were used. By 08:16, about 43 of the 115 units set aside for that phase were spent.
- The exact route it was built to test is dead (H-E13).

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

**Later updates:**
- 00:37: the subagent's chamber check answered #1. The chambers exist only for the recipe's gate-freezing rule; exact enumeration can start from 180 truth cones, and the chambers explain about 10% of the growth.
- The owner relayed the previous DRI's answers at 00:37, saved with confidence labels at [review/dri_handover/QA_2026-10-18.md](review/dri_handover/QA_2026-10-18.md). They establish that:
  - brute force was excluded only by the project's own "not understanding" framing, not by an ARC rule;
  - "≥ 1e17" was an extrapolation;
  - no protocol was ever fixed;
  - the ARC claim came from the owner's private message.
- The DRI then set one rung protocol, queued noisy-copy cost checks, and added leaderboard-repair and mechanism-audit rows.
- 00:41: the low exact masses came from reused random streams, not bias.

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

**Later updates:**
- Codex later delivered T240, T260, the T261 kernel and two Astra reviews.
- It went silent from about 08:30. Astra's round-3 review never arrived, and the DRI suggested the owner check whether the Codex manager is running.

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

**Later updates:**
- The CLI was already 0.7.4 and the machines really were gone, which rules out the CLI bug.
- The blocking keep-alive went live at about 03:45, and there were no reclaims until one A100 was lost between 07:23 and 08:12. That cost about an hour of the training grid, so progress now syncs every 8 minutes.
- At 13:07 the HP5 job was found to have silently stopped at 11:31 and was restarted.

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

**Later updates:**
- The neuron review landed at 04:56 and the 2–10 table at 05:22.
- The training ceiling was reorganised into a 1,050-model grid after the speed-ups (5.6–5.8), and its final stage is now due about 17:05 UTC.
- T240 ended as a bounded diagnostic.
- The owner's 06:56 directive ("Do not stop until we have solved the challenge") reopened the main question, which led to the lenient-bar work.

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

**Later updates:**
- After the owner's 04:51/04:52 push, the kernel became mandatory for T192, with a Codex backup (T261) running in parallel.
- **T192 found the real bottleneck (05:13).** 90% of each step went into one badly shaped weight-gradient matmul that ran on a single core cluster. Chunking it gave 61 → 2.4 ms per step at batch 8,192 and 9.4 → 0.81 ms at the standard batch, and 0.03 ms per model per step with 40 models packed. Gradients still match to 6e-16.
- **The fused Triton kernel was built anyway.** It runs at about 0.006 ms per model per step, and 108 models at once take 0.64 ms per step. It matches the reference within about 1e-6.
- The 40-vs-40-seed comparison showed no significant accuracy difference; an apparent early gain was chance.
- Codex's independent kernel also passes on an A100 (within about 5e-7, 0.010 ms per model per step).

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

**Later updates:** T192's Triton kernel does exactly this: data, labels, forward, loss, backward and updates in one kernel. Its correctness checks passed (05:19 and 05:41).

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

**Later updates:**
- At 06:32 the DRI approved a 1,050-model grid in a single launch (about 100 runs per setting: 5 data sizes, learning-rate and warm-up variants, three hard-case mixes) at about 39 GPU units.
- The kernel's in-kernel accuracy matches the reference evaluator within about 0.0001.
- The grid's copies of HP2's 100 models were bit-identical to the separately trained panel.

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

**Later updates:** at 13:23 the owner asked to add this as a low-priority backlog item. It is **T345** on the board: 1–2-layer ReLU networks with about 432 parameters (and a 2× budget), accuracy reported, then our exact tools vs plain sampling at equal compute, with both caveats recorded. Not started.

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

<a id="status"></a>
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
- **23:07:** "I'm helping the child session that works on the GPU to set up the Google Colab CLI and log in. After that, you are the DRI and you should manage it. It should not wait on me, and you should never wait on me anymore. Start child sessions instead of sub agents if you're asking them to do any experiment, so they have their own compute. If the task does not require any compute, for example, for researching about the method or such, you can spawn your own subagents. Try to keep your context window clean so you don't do the tasks and you delegate as much as possible. But you need to be a really good scientist that understands what's going on so you can prioritize the tasks and manage the sessions."
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
- **05:19:** "Okay, then let's also track if we can get up to 10x worse than sampling, because that seemed to be also something that Arc might accept." Done: Directive 8, every result reported against the strict (F ≤ 1) and lenient (F ≤ 10) bars; later amended (fixed budgets, then own-cost scoring with a 1k-FPE floor, and a strict-bar wording fix).
- **05:42:** "[.srt transcript of Jacob Hilton's talk] This is a talk from about a month ago, I think."
- **05:44:** "[.txt transcript] Read this in full yourself." See 1.7.
- **06:56:** "If you're stuck, ask for Fable and Astra help. Do not stop until we have solved the challenge. Think as a scientist." This led to the 07:03 reframing (cheap regime, own-cost scoring, sealed retrain panels), Fable/Astra round 2 and later rounds, and the T286–T317 lineage that reached the lenient bar.
- **13:12:** "When should we have resolved T340? I think this is the part that we should focus. I mean, your answer to my last question, about questions regarding the network that our approach would be able to better answer than the sampling method. I asked many questions and you provided very informative answers. I would like to make sure that these are all written down, so it's easy for future agents to read them and find them. Could you make sure that all these points are captured somewhere, including the questions that I asked?" T340 was made top priority with T344 added (4.5), and this document was written.
- **13:23:** "Add this small feed-forward network idea to the backlog. It's low priority." Done: backlog row **T345**, low priority (6.1).
- **13:40:** "And exploring the ideas about clever exact methods. Or exact computation as a component. We need to discuss that with Fable and Astra. Write down any idea that you have regarding it. The whole thing is as step two, brainstorm. This is the new task at the back of the backlog with low priority." Done: backlog row **T346** (low priority) at the end of the backlog; the DRI's seed ideas A-J with kill tests are in [m16/approaches/exact_brainstorm/](m16/approaches/exact_brainstorm/README.md); step 2 is independent Fable and Astra reviews (6.2).
- **13:58:** "Yes, please add the cheap way to test by retraining M1610 with binary and int for activations using quantization aware training. We can also consider the case that the inputs are also discrete, like int 4, int 8." Done: backlog row **T352** (after the rare-event lane); see 6.5.
- **23:3x (after the DRI listed options A-D with costs):** "C" — run T370 (fix the rare-event closure; CPU, est. 10-100 CPU-hours plus <= 3 Colab units, with a stop rule after stage 1 on seen models) and T352 (quantised variants; est. 5-20 Colab units, cap 20) in parallel. Done: both reserved and assigned (TASKS In progress). The ARC progress note stays undecided; no contact.
- **23:4x:** "You should not ever contact them. I will contact them when I deem it's the right time. Also in parallel, we need to work on the other ideas that we have in the backlog that were lower priority. For example, the quantization. We have many of them that we can run each in parallel using a separate session. So each have their own sandbox." Done: recorded in TASKS (directives; T41 closed; T203 (d) for the owner only), STATUS, DRI_HANDBOOK. Agents never contact ARC. Five backlog ideas started in their own new sessions: T345 (feed-forward control), T371 (rare events on weight-noised copies), T372 (exact weight-gradient of a rare-event probability), T373 (certified rare-event probability by event-cone enumeration, kill test), T6 (essay re-run); T352 (quantisation) was already running in the T192 child's own session.

---

## Addendum: questions from 13:30 to 13:58 UTC

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

**A:** Exact counts at 7 steps with today's CPU-bound code would cost ~30x the 6-step run: ~500-700 A100-hours, ~3,000 Colab units (~10x the budget), or about a day on 30 VMs. A sampling census instead (>= 1e9 inputs, distinct gate patterns, mass coverage, restricted to near-tie and error inputs, k = 4..10) takes ~15-30 min of GPU and ~0 extra units. Queued as **T348**; validated at k = 4 against exact counts (394 / 1,699 vs 396 / 1,737). Caveat: at k >= 8 most patterns are singletons, so fine-delta counts are lower bounds with a Good-Turing missing-mass estimate.

<a id="q6-5"></a>
### 6.5 Would quantising the model (binary, FP4, FP8, int4, int8) let us solve the challenge?

**Q** (13:50 UTC): "What do you think if we quantize this model? Would it help? ... binary model or FP4 or FP8 or int4 or int8 ... Assuming that ARC proposed such architectures, we want to know if we can solve the challenge."

**A** [speculative, untested]:
- **Weights only** (int8/FP8/int4/FP4/binary weights, real activations): little help. The network stays continuous piecewise-linear on Gaussian inputs, and the x23-33/step region growth comes from composing 10 steps. Integer weights make exact geometry cheap and robust. Accuracy cost: the model is very sensitive (0.25% relative noise costs points), so int8/FP8 post-training rounding costs accuracy, and int4/FP4/binary weights need quantisation-aware training (QAT), likely with lower accuracy at 432 parameters.
- **Activations quantised** (finitely many hidden states): the big change. The RNN becomes a finite state machine; histories reaching the same state merge automatically (the "merge identical futures" idea built in). An exact forward dynamic programme over (state, input interval, label summary) costs about reachable states x intervals x steps. Binary activations: <= 65,536 states, ~17 input intervals each: likely solvable exactly at roughly the cost of sampling to 0.1 pp. int4: depends on how many states are reachable (unknown). int8/FP8: probably too many states unless the dynamics are very low-dimensional. Discrete inputs (int4/int8) make the whole problem finite and simpler.
- **Would ARC accept it?** By the letter yes (an exact answer has zero error); by the spirit it is debatable: deductive and arguably mechanistic, but it sidesteps the hard part (real-valued states composing over many steps), which is what real models do.
- Follow-up **T352** (owner, 13:58): QAT retraining with binary/int4/int8 activations, reachable-state census, exact DP for the binary model, and discrete-input (int4/int8) variants.

<a id="q4-6"></a>
### 4.6 Rare events: the first results (update to 4.3-4.5)

T340 (13:43): on 3 models, rare error probabilities of 2e-7 to 4e-4. A cheap variant (11-19k FPE) estimates the "wrong with both gaps large" events within 3x in 6/6 cases; a careful variant (0.2-0.3M FPE) estimates the far-gap events at .64-.89 of the truth on the retrains and .20-.29 on the main model. Equal-compute comparison with executed costs (post hoc, ties separated): each fixed variant beats sampling in 5/15 cases; using each event family's better variant gives 10 wins, 3 ties, 2 losses. The DRI's first "12 of 15" counted ties as wins and was corrected. No single estimator wins a majority. An independent re-check on fresh DEV3 models (T350) fixes that family rule in advance. T344 (certified bounds): rigorous volume certificates fail for this architecture (certified boxes <= 1e-4 wide even in 3-D; sub-cone enumeration grows to ~2e8 regions), so only the trivial bound is rigorous (8,200-27,500x loose).

**Later update to 4.6 (14:18):** the independent re-check on 6 fresh DEV3 models (T350, pre-registered, own code) did NOT replicate: the family rule won 11, tied 6, lost 13 (46% of decided cases vs the 2/3 line). The cheap variant still beats sampling on the "both gaps large" events (7 of 9 decided); the careful variant loses on the far-gap events and was 6-10x low on one model. A correctness bug was then found in the long-line walks (simultaneous gate crossings when an input crosses zero), so the careful-variant numbers (T340 B, T340c, T354) are suspect until fixed and rerun; the main accuracy estimator T317f is believed unaffected (verification pending).

**Later update (14:35), strict bar:** H-S1 marked dead (verified), scoped to the T317 paradigm (point closure + exact gap-line walks + declared combination, zoo retrains, C <= ~6k FPE), after the T356 checklist and the independent T357 re-check. The closure is not the binding floor (perfectly integrated it gives F .89 at 3.3k FPE on DEV3); the walk integration error would have to fall ~4.6x at no added cost. A lattice quadrature (the most promising untried scheme) missed by ~5x. Not a proof; other paradigms remain open.

**Later update (14:47), rare events:** T358 (20 more fresh DEV3 models, rarer thresholds) is formally a GO but substantively limited: the cheap estimator is usable only near p ~1e-4 to 1e-5 and is 5-20x low at p <~ 1e-6, where it only beats sampling because sampling returns zero. The rare-event primary criterion is now 'within 2x of the truth'. No rare-event estimator yet is both accurate and cheaper than sampling on fresh models; the accurate careful version (T340c) is being rerun after the walk-bug fix.

**Later update (15:20):** T361 reproduced Fable r6's four-term split of the strict-bar integration error and killed the one free idea (homogeneous closure: floor .762 > .45). T359 cut the accurate rare-event estimator's cost to ~.53x at unchanged accuracy. Fable's T346 brainstorm (DRI branch review/fable_t346): exact routes for the accuracy number < 1%; kept as components: exact 1-D walks, exact sensitivities, shared products, sub-cone certificates for p <= 1e-8; new tasks T362 (conditional MC along exact lines, sampling-class, rare events) and the T348 census extension (error-pattern growth, the most informative single experiment).

**Later update (15:25), rare events:** T360 (independent agent, own code, 6 fresh DEV3 models) confirms the accurate estimator for far-gap rare errors: 18/18 within 2x (medians .99-1.06), registered as ledger row H-R1 (confirmed, scoped). The interior 'both gaps large' event stays one-signed low (76% within 2x). At equal compute it still mostly loses to sampling (21 win / 22 tie / 67 loss), because it costs 1-1.5M FPE (E1) to 3.7-6.7M (E2).

**Later update (16:20):** T362 (conditional MC along exact lines, sampling-class) unbiased but gain only 1.2-4.7x (no go). T363 traced the interior event's low bias to the lows closure (56%) and the v4 position (21%); T365's spread-lows fixes reached 20/32 within 2x vs the 24/32 line (scoped negative; a real fix needs ~10-100x the cost). T364 (p ~1e-8 to 1e-10) pre-registered with an exact conditional-law truth method. T348 census interim: entropy-effective pattern growth x5.1 -> x3.2 per step by k = 9.

**Later update to 6.4 (16:36), GPU census result (T348):** on 1e9 inputs, the probability-weighted number of gate patterns grows ~x4-6 per step (all inputs and error inputs alike, slowing from ~x5-6.5 at k = 5 to ~x3.6-4.8 at k = 8), far below the x23-30 growth of raw region counts but above the ~x3 needed to revive exact counting for rare events. Late steps carry most of the pattern diversity. Raw distinct counts beyond k = 5 are limited by the sample size, not the model. A sharper answer would need ~1e10 samples (~7 GPU-hours, out of budget).
- **17:27:** "Let's stop using Fable from now on." Done: no further Fable reviews or sub-agents; recorded in TASKS (owner directive) and DRI_HANDBOOK. Existing Fable reviews (r2-r6, T347 note, T346 brainstorm) stay in the record.

**Later update to 2.1 (18:24), training ceiling final (T234, 100 retrains per setting, unverified):** mean accuracy by training data 2^30 / 2^31 / 2^32 / 2^33 / 2^34 sequences: .911 / .936 / .938 / .944 / .952; the best single models reach ~.98 from 2^31 on and do not improve further. So the typical model keeps improving with more data (~0.6-0.8 pp per doubling) while the best models have saturated near .98. The blind-input error (~0.7%) is a floor of the recipe. Doubling the learning rate at 2^31 matches 4x more data (.9465). M_16,10 (.9531) is at the 53rd-80th percentile of the largest-data retrains, below the ~.98 ceiling. Colab: 164.4 units left; all runtimes stopped.

**Later update to 4.6 (18:50), rare events:** T367 (fair test of the near-tie fix, truth-blind selection, sealed truths) narrowly missed its bar: 6/8 within 2x (7 needed), while plain K3 was within 2x on 3/8 and missed by up to ~3e5x on dev3_s541, where every rare mistake is a three-way near-tie; the fix got s541 right (1.25, 1.35) but was 2.1-2.4x high on s547. T368 (independent agent, own code) confirmed the deep-tail extension on ordinary networks: 10/10 within 2x, so H-R1 now reads 'within ~1.4x from 1.8e-10 to 7e-7 on non-triple-tie models'. T369 showed the s547 overshoot is not quadrature (the graded rule is converged) but the closure: pinning the smaller inputs at mu(v4) and fixing v4's position inflates the near-tie islands; with exact lows and v4 position the estimate is 1.04x of the truth. So the one remaining failure mode of the rare-event estimator is that simplification, which also causes E2's low bias. A proper fix (T370, backlog) costs 10-100x more per estimate: a correctness demonstration rather than a cost win. Answer to 'are we closer to estimating rare bad behaviour?': yes for far-gap mistakes on ordinary networks (confirmed); not yet on near-tie-dominated networks (including the target at its deepest tails).
