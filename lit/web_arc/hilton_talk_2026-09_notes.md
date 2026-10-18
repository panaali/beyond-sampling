# Hilton talk notes (ARC, ~Sept 2026, interpretability workshop)

Source: auto-generated SRT, cleaned in `hilton_talk.txt` (same folder). Quotes are approximate (fillers removed, transcription errors possible). The DRI should log this source in `lit/web_log.md` (rule 7).

## a) What the challenge asks

- **Model:** only M_16,10: "432 parameters" [10:11], "around 95% accuracy" [10:44]. The task is stated backwards at [09:04]. M_2,2 and M_4,3 are not mentioned.
- **Goal:** "a particular challenge that we proposed is ... as yet completely open" [33:00]: "produce a mechanistic estimate of the accuracy ... match random sampling in terms of the like amount of computation relative to like mean squared error" [33:12].
- **Protocol:** "doesn't really make a lot of sense for a single model", so a distribution of models: "taking a random seed or maybe adding noise to the weights" [33:22-33:31]. **No sigma, no relative vs absolute noise, no number of copies, no seed list, no input distribution.** "Mechanistic isn't really formally defined" [33:45].
- **Compute:** only "amount of computation"; no FLOPs, operations or budgets.
- **MSE baseline** (from a question): "you sample random inputs and you look at empirically what the accuracy is" [35:18]; MSE is averaged over "randomness over the weights" [35:29]. No tolerance factor.
- **Mechanistic:** "deductive rather than inductive" [24:57], "just like no sampling" [25:02]; "even on some subset of the network, you're not allowed to just like sample some inputs ... and assume that like other things will behave similarly" [25:05]. Approximations are fine ("roughly zero. I'm ignoring some noise" [25:22]). Fitting a regression on samples is "a little bit cheating" [17:52]. Each step must be a "valid deductive step" [28:50], but the whole may be "too big for a human" [29:00]. Q&A: "you can still give inputs to the model" [30:01]; "We don't currently have a formal definition" [30:05]; "I want to allow putting inputs through the network. It's more about ... what do you do with that?" [30:49]. Seeing which weights fire on one input and then going on to "extrapolate deductively" [31:17] is fine.
- **Quasi-random or fixed nodes:** not mentioned. [inferred] His test is about the *inference*, not whether the inputs are random. Averaging over fixed Sobol nodes assumes the unevaluated inputs behave like the evaluated ones, so it counts as inductive (sampling). Fixed inputs used as witnesses for exact reasoning over a region are allowed.
- **Advice / hard-coding:** not mentioned. [inferred] The estimator "takes in the weights" [24:29] and is scored over a distribution of models, so a hard-coded number fails. His by-hand analysis is "kind of like a guide" [43:41]. Advice computed by sampling falls under "not allowed ... even on some subset".

## b) Stated progress

- **No solution claimed:** "as yet completely open" [33:00]. The understanding is partial: a closed 2x2 block holds the running max and the "running maximum except for the very last" input [15:14]; the next rows hold the most recent input and the "leave-one-out maximum" that skips x_{t-2} [16:20-16:26]; "okay-ish understanding of like neurons one and zero [sic]" [17:05], "pretty much no understanding of ... the rest" [17:09]; by hand "gets a little bit stuck" [17:26].
- **Admitted gap:** the near-zero weights are "not exactly zero" and may compute "some ... slightly non-trivial function" [18:54]. When asked whether the analysis assumes insensitivity to perturbation: "I think that assumption is just like false" [19:16].
- **Other ARC progress:** Mike (Winer) "match[ed] the performance of random sampling in the case of ... randomly initialized models" [34:41]. Paul (Christiano) will present a conjecture that it is "always possible to match performance" of sampling [34:25].
- **New team member:** no hint. ARC is "five full-time staff", "hoping to have more people join by January" [00:41-00:44]. Zihao is credited as the first to study the model [00:55]. [inferred] A solution by someone who joined after this talk (~1 month ago) would be consistent with it (H-E5); a pre-talk solution would contradict "completely open".

## c) Difficulty and expected methods

- **Difficulty:** the approximate sparsity covers only a few neurons, small weights matter, and fitted correlations are not justified by the weights. A sparsity penalty may not "work in the worst case" [27:30].
- **Expected route:** theoretical "general schemes" rather than "doing it by hand in one case" [43:34]; tiny models are "maybe like 10%" of ARC's work [43:26]. **Not mentioned in the talk:** brute force after symmetries, surprise accounting, heuristic arguments, presumption of independence (these come from the Jan 2026 blog). The only symmetries used are ReLU rescaling and neuron permutation, for readability [11:20-12:18].

## d) Against our assumptions

1. **Relative sigma 0.0025-0.02 x RMS:** neither supported nor contradicted. Seeds are named as an equal alternative [33:31]. [19:16] fits our finding that M_16,10 is very sensitive to noise.
2. **FLOPs, ~5,948/input:** the talk only says "amount of computation". Our convention is still our own choice.
3. **Bars:** the talk says "match", with no "roughly" and no M_4,3. The literal reading supports the **strict bar (<= 1)**. The lenient bar (<= 10) still rests only on the M_4,3 blog wording. Matching needs "much much more understanding than we currently have" [34:09].
4. **Sobol nodes:** our "probably not mechanistic" is sharpened. Whether the nodes are fixed or random does not matter; the deciding question is whether there is a deductive link from the evaluated inputs to the rest.
5. [inferred] **Coefficients fitted by sampling** (regressions, correctors, tables) look like what he calls "a little bit cheating" [17:52]. This is a risk for E15's uncharged tables and similar.

## e) Other models and tasks

- A hand-designed RNN with "maybe 21" hidden neurons [19:44] (the blog says 22) reaches "100 minus epsilon" [19:46] at about "90% sparsity" [22:42]. "21 is the minimum" is a guess [20:18]. A better 16-neuron model probably exists, because "SGD is like failing" [20:01].
- Other same-architecture models have "10 or so checkpoints" [21:36]; M_16,10 has none [22:00].
- Also: randomly initialized models (Winer's talk) [34:41], and a Claude attribution graph for 23+28 [04:00]. Despite the title, no other AlgZoo task or size is mentioned. [36:10-43:03] is an off-topic alignment section.

## f) Relevant Q&A

- Turing-machine analogy [31:30-32:48]: the weights were optimized only for accuracy, so explaining accuracy "pretty much" explains everything [32:15].
- Also: 100% at this size [19:30-20:18]; recent ChatGPT versions not tried on this model [20:50]; the first analyses were "correlational studies" [23:06]; a sparsity penalty has not been tried [26:57].

## g) Summary

Hilton presents only M_16,10. He walks through the known features and admits that the rest of the network is not understood and that the near-zero weights are not negligible. He restates the challenge as open: a mechanistic, deductive (not inductive) accuracy estimate that matches random sampling in MSE per unit of computation over a distribution of models (random seeds or weight noise). He gives no noise level, copy count, compute unit, budget or tolerance. There is no hint of a solution or of a new team member's result. ARC's only claimed match with sampling is on randomly initialized models (Winer). The talk supports our strict bar and our view that Sobol quadrature is not mechanistic. Our sigma and FLOP conventions remain our own choices, and estimators with sample-fitted constants are at risk.
