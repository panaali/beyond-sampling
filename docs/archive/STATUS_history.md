# Status history (archive)

↑ [STATUS](../../STATUS.md) · [INDEX](../../INDEX.md)

The full STATUS.md as it stood before T396 (2026-10-19), moved here verbatim: the 2026-10-18 14:00 page, every dated update to 10-19 06:35, and the technical summary. Snapshots only; current numbers and claim status are in [INDEX.md](../../INDEX.md). Its top-level title is demoted below.

## Project status (2026-10-18 14:00 UTC; dated updates to 10-19 06:25)

↑ [Project README](../../README.md) · [TASKS](../../TASKS.md) · [INDEX](../../INDEX.md) · [Owner Q&A](../../OWNER_QA.md) · [RESULTS](../../RESULTS.md) · [FINDINGS](../../m16/FINDINGS.md) · [Approaches](../../m16/approaches/README.md)

This page summarises [`INDEX.md`](../../INDEX.md), which holds the registry, the hypothesis ledger and every number with its
source. Each claim below cites its INDEX row (T-, E- or H- number). "(unverified)" means no second agent has checked it
yet. Nothing here is confirmed unless the ledger says so. Most result folders live on the child sessions' branches,
not on this branch; each INDEX row names the branch and commit.

## In plain words

**In one paragraph.** ARC's challenge is still unsolved. ARC asks for a way to calculate this network's accuracy from
its weights, without trying random examples, that is at least as precise as random testing for the same amount of
work. Our best calculation is about 2.6 times worse than random testing in squared error at equal effort (it is too high
by about 0.44 points, and its typical error is about 0.75 points). That passes the project's own "within 10 times" bar but not ARC's "match" bar,
and it works only on freshly retrained networks, not on networks with noise added to the weights. Since 13:13 UTC
today the owner's priority is rare failures, which is ARC's long-term goal. There we have one confirmed positive result (H-R1): for one kind of rare mistake (far from ties) our calculation lands within 2x of the truth for chances down to 1 in 5 million, and within about 1.4x down to 2 in 10 billion on ordinary networks, where random testing at the same cost sees nothing. It still fails on networks whose rarest mistakes are three-way near-ties, and it is not cheaper than random testing for common events. Counting the network's pieces exactly is ruled out.

**What we set out to check.** ARC's challenge is to calculate a small network's accuracy from its mechanism at least as precisely as random testing at equal effort.

**2. How the 432-number network works: mostly understood, but not precisely enough to calculate with.**
- Parts: a "highest so far" tracker, a "new record" alarm, checkers comparing the newest number with the high score,
  and rotating "clocks" whose angle records how long ago the biggest number arrived (measured 41.47-41.71 degrees per
  step on two seeds, against 42.75 from the weights; H-M1).
- Its mistakes come from two situations: near-ties between the 2nd and 3rd largest numbers, and inputs that are almost
  all negative.
- A neuron-by-neuron account was checked by a second agent (T241, core confirmed). But swapping all 16 neurons for our
  formulas agrees with the real network on only about 95.5-96.6% of inputs (.9554 and .9659 in two implementations),
  and its accuracy is off by +0.11 to +0.82 points. A calculation needs far more precision than that, and so far no
  estimator has been improved by the neuron-level story (T338, T341; [Owner Q&A 4.2](../../OWNER_QA.md#q4-2)).
- Some prettier claims (a neat "harmonic" clock series, "all 16 neurons confirmed") were weakened or withdrawn by our
  own checks.

**3. ARC's challenge: not solved. Our best calculation is about 2.6 times short.**
- **How it works.** For each of the 720 ways to place the top three numbers among the ten positions, it follows the
  network exactly, switch by switch, along the two directions in which the top numbers come close to tying. It then
  adds up the probability of the wrong stretches. Seven simplifications are declared in advance, for example "cases
  with a large gap are assumed correct" and "the two directions are treated as independent"
  ([Owner Q&A 3.9](../../OWNER_QA.md#q3-9)). It never runs random examples.
- **Score (T317f, T335).** It was scored once on 100 freshly trained copies that nobody had looked at (the sealed
  panel HP4, run by the independent scorer). Typical error 0.75 points, overestimate 0.44 points, about 3,300 model runs
  per network. Random testing at the same effort has about 2.6 times smaller squared error, and it reaches our accuracy
  with about 1,255 runs. Result: it passes our lenient bar (within 10 times) and fails ARC's strict bar (match).
- **Limits.**
  - It works only on retrained networks. With noise added to the weights it breaks down between 0.1% and 0.25%
    relative noise (T337, unverified), because noise creates mistakes far from the near-ties, where the method assumes
    the network is right.
  - No random examples are used when it runs. But its simplifications were chosen by checking them against random
    testing on 80 development networks, so it is not "deductive" in Hilton's strict sense (second-agent audit, T324).
- **Why it is stuck.** A second agent split the error into parts (T324; the two biggest parts were re-measured by
  another agent with its own code, T334). Treating the two directions as independent causes about 40% of the squared
  error, and assuming large-gap cases are correct about 20%. Summarising the 7 smaller numbers by one value causes 34%.
  Every repair so far either costs as much as it saves or swaps one error for another:
  - cheaper settings: none helps, and there is a floor of about 2,200 runs (T326);
  - proofs that a walk can stop early: exact, but they save almost nothing. A declared early-stop rule is what made
    T317f (T330, T330b);
  - walking further: more accurate, but too costly (T331);
  - spreading out the 7 smaller numbers instead of using one value: its errors vary much less from network to network,
    but all its estimates move up, so it is too high by 0.76 points overall (T327, T339);
  - handling the two directions jointly and exactly: about 17 times the whole budget that ARC's bar allows (T333);
  - a formula for how much "independent" overstates accuracy: it removes most of the overstatement, but it missed its
    pre-registered accuracy line (T347);
  - combining the last two ideas: the pre-registered choice kept T317f. The combination had the smallest error (0.61
    points) but cost about 2,800 extra runs (T351).
- **Newest result (T353, unverified).** A cheaper rewrite of that combination gives the same estimates at 5,079 runs. On
  the 80 development networks it scores 2.60, against T317f's 3.14 on the same networks (lower is better; 1 means
  "matches random testing"). Next it is checked on fresh development networks (DEV3). If it holds, it becomes the
  single submission for the last sealed panel (HP5). Even then, ARC's bar needs a typical error of about 0.38 points at
  that cost (T351).

**Strict bar, verdict (14:35):** for our whole approach (simplified closure + exact walks along the tie directions + declared combination), the strict bar is now marked **dead (verified), scoped**: a written checklist of ~27 variants (T356) was independently re-checked by another agent with its own code (T357). The sharper finding: the simplified closure itself would be good enough if integrated perfectly (F .89 at our cost); what fails is the integration accuracy, which would need to be ~4.6x better at no extra cost. Not a proof; other approaches remain open.

**Since 14:35 (short):** the accurate rare-event estimator was rerun after the bug fix (within 2x in 14-15 of 15 cases on 3 networks; T340 final) and made about 2x cheaper by rigorous walk cuts (T359: ~1.2-1.35M runs); its independent fresh-network test is running (T360). Fable's round 6 found that the strict bar's remaining error splits into four independent pieces, each too big on its own, and a quick idea to remove two of them failed its test (T361). Fable's exact-methods brainstorm (T346) puts any exact route for the accuracy number below 1%; two ideas from it are being tested: a smarter sampler for rare events (T362, sampling-class) and the GPU census of how fast error patterns grow (T348).

**Rare-event result confirmed (15:25, H-R1):** a different agent, with its own code, confirmed on 6 fresh networks that the accurate estimator gets the far-gap rare mistakes (chances 1 in 5,000 to 1 in 5 million) within 2x of the truth in 18 of 18 cases, without random testing. It is not yet good for the 'both gaps large' mistakes (one-signed low on some networks) and it is still more expensive than random testing at equal accuracy.

**16:20:** the smarter sampler (T362) was unbiased but only 1.2-4.7x better than plain random testing (needed 10x). The 'both gaps large' rare mistakes are underestimated because the simplified treatment of the smaller inputs erases their error islands (T363); spreading those inputs over a few patterns helped (9 -> 20 of 32 cases within 2x) but not enough (T365); a full fix would cost 10-100x more. The deep-rarity test (T364: chances around 1 in a billion, where random testing at our cost sees nothing) is pre-registered and being scored. The GPU census (T348) shows the effective number of network states growing ever more slowly per step (x5.1 down to x3.2); the decisive error-only counts are pending.

**17:20:** the deep-rarity tests: T364 failed on the main network and one other (their deepest mistakes sit in thin three-way near-tie slivers), but a sealed fresh test (T366) passed 8/8 within 2x, and across the fresh networks the estimates are within 1.3x on 10/10 cases from 2 in 10 billion to 7 in 10 million, where random testing at the same cost expects 0.005 to 1 events. A fix for the near-tie slivers is being tested fairly (T367); an independent re-run with different code is set (T368). The GPU census (T348): error patterns grow ~4-6x per step, too fast to revive exact counting for rare events. The final exact count at 6 steps (T270): 19.7M regions.

**18:50 (end of day):** the fair test of the near-tie fix (T367) narrowly missed its pre-set bar (6 of 8 within 2x; 7 needed), but the plain method missed by up to ~300,000x on a network whose rarest mistakes are all three-way near-ties, and the fix got that network right. An independent re-run with different code (T368) confirmed the deep-rarity result on ordinary networks: 10 of 10 within 2x (stated as 'within about 1.4x'). The follow-up (T369) found that the fix's overshoot on the one bad network (2.1-2.4x) is not a tuning problem: it comes from the simplified treatment of the smaller inputs (they are pinned to one value, and one input's position is fixed). Letting them vary in a check brings the estimate to 1.04x of the truth. A proper fix (T370, backlog) would cost 10-100x more per estimate. Training ceiling (T234, final): typical retrains keep improving with more data (mean .911 to .952 from 2^30 to 2^34 sequences); the best reach ~.98; M_16,10 (.953) is typical, not at the ceiling. On the owner's instruction, Fable is no longer used. Nothing is running tonight. Next eligible work: T370 (rare events, costly), T352 (quantised variants), T345 and T346 (low priority).

**10-19 02:10 (overnight, parallel sessions; all unverified unless noted):** by owner directive, backlog ideas now run in parallel, each in its own session. Finished: T10, the smallest case where our calculation starts losing (16 neurons, 5 steps), now beats random testing on all 5 networks (squared-error ratio <= .23), but with a fixed-directions quadrature, not a mechanistic method, and with groupings chosen on the same networks (a fresh-network check, T379, is running); at 6 steps it narrowly fails (9/10 checks), exactly as its own law predicted, because each step roughly doubles the mistake rate. Negative: polynomial descriptions of the network (T291, T375) cannot even represent its outputs; simplified 'clock-switch' descriptions (T3, T376) fail; exact counting inside the rare-event region (T373) fails. Neuron accounts depend more on how fragile the network is than on its size (T24, T377). Running: rare-event fix (T370), rare events with weight noise (T371: noise does not break the method; three-way near-ties do), rare-event weight gradients (T372), quantised networks (T352), the all-at-once network (T345: the simplest one reaching 95% is one layer of 128 neurons, mostly pairwise comparators, but just as many pieces as the RNN), a lower-noise walk (T380). The interactive tutorial (T374) is live as a private Artifact (https://claude.ai/artifact/22zmMZh9WU6LAPLWqJmXRX); a Codex prose rewrite (T382) is queued. Agents never contact ARC (owner directive).

**10-19 04:15:** **New confirmed result (H-E16, scoped):** at 16 neurons and 5 steps, a fixed-directions walk calculation beats random testing; an independent agent reproduced it on 5 freshly trained networks (T379). It is quadrature-class (not mechanistic), clean networks only, and slower than random testing in wall-clock time as coded. At the full 10 steps the same method roughly ties random testing (T380). Rare events: the estimator survives ARC-style weight noise (T371: 89% within 2x; all misses are three-way near-tie cases), and the rarest mistakes of M_16,10 are driven mainly by the balance of two comparator neurons, n6 and n7 (T372, sampling-class gradient; an independent check with an intervention test, T385, is running). Negatives: exact counting inside the rare-event region (T373, at least ~1e15 pieces). Architecture: the simplest all-at-once network reaching 95% is one layer of 128 comparator neurons; a walk along circles beats random testing on it (F .15-.30 with fixed circle sets, T387), but no sampling-free method yet. Quantised networks (T352, interim): 8-bit keeps accuracy, on/off loses it at 16 neurons; wider versions are training. The tutorial passed an independent Codex fact-check (T384); its corrected version is being republished at the same link.

**10-19 06:25:** at the full 10 steps, three deterministic quadrature methods sit at or near the random-testing line (E33 ~1.6x better on 5 zoo seeds; T10's walk ties on 100 retrains; T389 .74 at the smaller budget but 1.22 at the larger); none has a sealed pass yet, and ARC acceptance of fixed-line averaging is unknown (OWNER_QA 4.7, 4.8). A single sealed test of all frozen candidates is planned after T391. The ladder is resumed (T86/T111, T46), and new CPU-heavy lanes run in parallel: chunked-input networks (T392), a compiled walk for real wall-clock cost (T393), an independent check of the all-at-once network (T394), a de-aliased rare-event gradient (T386), a 1e10 error-pattern census (T395). Quantisation: 4-bit at 64 neurons reaches ~95% but its states are per-input (no exact counting); on/off states are tiny and exactly countable, accuracy pending at larger widths. The rare-event closure fix (T370) now gets the hard network s547 to 1.02/.87 of the truth.

**4. Rare failures: the owner's priority (ARC's long-term goal).**
- ARC ultimately wants to estimate the chance of rare bad behaviour. Random testing cannot see a one-in-a-million
  event with a few thousand tries, but a calculation's cost does not depend on how rare the event is.
- **First test (T340, partial positive, unverified).** On 3 networks we looked at rare kinds of mistakes with true
  chances between about 2 in 10 million and 4 in 10,000 (measured with 1-10 billion GPU tests).
  - A cheap version (11,000-19,000 runs) gets "wrong even though both gaps are large" within a factor of 3 in all 6
    cases. But it misses the "far-from-tie islands" completely.
  - A careful version (210,000-320,000 runs) finds the islands at 0.64-0.89 of the truth on the retrained networks,
    but only 0.20-0.29 on the main network.
  - At equal effort, each version alone beats random testing in 5 of 15 cases. Using the better version for each
    kind of event wins 10, ties 3 and loses 2. That rule was chosen after seeing the results, so it is being re-tested.
- **The fix worked for accuracy (T340c, board T349, 14:15, unverified).** Computing the off-line direction properly
  puts the careful version within a factor of 2 of the truth in every case, including the main network, for chances
  from 2 in 10 million to 4 in 10,000: the first time in the project. But it now costs 1.8-4.5 million runs, so it
  beats random testing only for the rarest events (4 wins, 3 ties, 8 losses of 15). Its cost does not grow as events
  get rarer, while random testing's does, so the rarer the failure, the bigger the advantage.
- **The independent re-check on fresh networks failed (T350, 14:18, unverified).** With the rule fixed in advance, the
  combined approach won 11, tied 6 and lost 13 cases on 6 new networks, below the two-thirds line. The cheap version
  still beats random testing on the "both gaps large" mistakes (7 of 9 decided cases), but the careful version loses
  on the far-gap islands and was 6-10x too low on one new network. Separately, a bug was found in the walks that run
  far along a line (they miss gates that all switch at once when an input crosses zero); the careful-version numbers
  are suspect until it is fixed and rerun. The main accuracy estimator (T317f) is confirmed unaffected (its short walks match direct network runs exactly).
  The re-check's own code does not have this bug (dense check agrees), so its negative stands; what replicates on
  fresh networks is the cheap version's win on the "both gaps large" mistakes.
- **That win is narrow (T358, 14:47, 20 more fresh networks).** The cheap version is a usable estimate only for
  chances around 1 in 10,000 to 1 in 100,000. For rarer cases it is 5-20x too low, and it "beats" random testing only
  because random testing sees nothing. So we do not yet have a rare-event estimator that is both accurate (within 2x)
  and cheaper than sampling on fresh networks; the accurate careful version (T340c) is being rerun after the bug fix.
- **Running now:**
  - cutting that cost by walking only the part of each line where the event can happen (T354);
  - an independent re-check by a different agent on new networks (DEV3), with the rule fixed in advance (T350).
- **Guaranteed limits failed (T344, unverified).** The 10 steps blow up uncertainty, so proofs only work on tiny
  regions. The only guaranteed upper limit is 8,200-27,500 times too high.

**5. Counting the pieces exactly is ruled out (H-E13).**
- With every switch fixed, the network is a simple linear function. But the number of such pieces multiplies by about
  23-33 per step: at least 15.5 million after 6 of the 10 steps (T270, an exact lower bound). A geometry bug (T316)
  means the true count is higher. Extrapolated, about 10 trillion (1e13) pieces at 10 steps.
- A data-centre-scale run might finish, but at roughly a trillion (1e12) to 1e16 times the work of random testing (an
  order-of-magnitude estimate, [Owner Q&A 6.3](../../OWNER_QA.md#q6-3)). That would meet the letter of ARC's challenge, not
  its point.
- Exact calculation as one *part* of a method is alive: T317f is exact along its chosen lines.

**6. New test panels.**
- HP4 is used up (T317f).
- HP5 is sealed with the scorer as the reserve for one more sealed submission (T336, 14:08; nothing scored yet).
- DEV3 is being prepared as a fresh open development set (T342), so that new ideas are checked on networks not used to
  design them. The 80 networks of DEV2 have been reused for about 40 variants.

**7. The owner's ideas on the backlog** (not started):
- **T345:** a small feed-forward network on the same task, to test whether the 10-step depth is what makes this hard
  (low priority).
- **T346:** a brainstorm of cleverer exact methods, with two independent outside reviewers (low priority). Step 1, the
  DRI's seed ideas with kill tests, is in [m16/approaches/exact_brainstorm](../../m16/approaches/exact_brainstorm/README.md).
- **T348:** a GPU count of how fast the relevant pieces grow up to 10 steps, as a diagnostic. It is reserved, but
  queued behind the rare-event answer keys and DEV3.
- **T352:** versions of the network with coarse activations (on/off, or a few levels; possibly also coarse inputs).
  These turn it into a finite machine that an exact calculation might handle cheaply. Planned after the rare-event
  work. Whether ARC would count that is debatable ([Owner Q&A 6.5](../../OWNER_QA.md#q6-5)).

**8. ARC.**
- **Public position:** on 2026-08-17 Hilton wrote that progress on AlgZoo is "fairly marginal". His talk from about
  2026-09 calls the M_16,10 challenge "as yet completely open" ([lit/web_log.md](../../lit/web_log.md),
  [lit/web_arc](../../lit/web_arc/README.md)).
- **Their own bar:** ARC's published estimators for its small 32-number network were 2.4 to about 10 times worse than
  sampling by our reproduction (E32), and ARC called that "roughly match". That range is where our lenient bar comes
  from. It is our convention, not ARC's rule.
- **A reported solution:** the owner relays, second-hand, that a new ARC member has a solution for M_16,10 (H-E5,
  unverified).
- **Contact:** only the owner contacts ARC, when they judge it right (owner directive 2026-10-18 23:4x). Agents never
  contact ARC and do not recommend contact.
- **The question to ask ARC:** a real solution must handle the thin near-tie regions, broken into very many pieces, at
  10 steps. That is where every approach here broke.

**A precise answer key exists:** the network's accuracy is 95.31175% ± 0.00004 points (300 billion samples, T192).

**How we got here.**
- 2026-10-17: about 30 estimator attempts were scored on shared leaderboards, and every pure calculation lost on the
  10-step network. For ARC's 32-number network (M_4,3), a calculation built from the mechanism beat sampling about 50
  times in squared error (E36, confirmed, scoped).
- Overnight to 2026-10-18 morning: a step-by-step "ladder" toward the full network passed (8 neurons, 4 steps) and then
  failed on one seed at 5 steps (E46, T192, unverified). Exact counting was tested and ruled out.
- 2026-10-18 morning: the scoring moved to each method's own cost and to sealed panels of retrained networks.
- 2026-10-18 midday: the T317 family reached the lenient bar (HP2, then HP3 at 5.07-5.08). T317f followed at 2.63 on HP4.

## Technical summary

| Question | Verdict | Where |
|---|---|---|
| Q2 Legible vs understood (AlgZoo) | M_4,3 control passes; M_16,10: 14/16 neurons legible, 5-constant labels rebuild 54% (driven by the constant limit); per-neuron account core confirmed (T241), all-16 replacement accuracy match not robust (+0.11 vs +0.82 pp) | [`RESULTS.md`](../../RESULTS.md), H-A1/A2, T241 |
| Q3 Mechanism of M_16,10 | running max / record detector / comparators / damped rotating modes (phase = time since max); error anatomy exact; Fourier/harmonic reading weakened | [`m16/FINDINGS.md`](../../m16/FINDINGS.md), H-M1..M8 |
| Q4 ARC challenge | **open**: strict bar unmet. Best sampling-free estimator T317f, HP4 F 2.63 [1.93, 3.39] (lenient pass, strict fail, retrains only); rare-event lane partial positive (unverified) | [`INDEX.md`](../../INDEX.md) section 4b, H-L1 |

Score used below: **F** = MSE x C_eff / mean p(1-p), with C_eff = max(C, 1k FPE) and FPE = forward-pass equivalents
(TASKS Directive 8, amendment 2). Plain sampling scores F = 1. **Strict bar:** F <= 1 (ARC's "match"). **Lenient
bar:** F <= 10, and a "candidate" also needs the upper 95% CI <= 10 (a project convention taken from E32's 2.4-10x).

### Q4 scoreboard: the T317 family on sealed retrain panels (scored by the independent scorer, T281)

| Version | Change | Panel | C (FPE) | F [95% CI] | Notes |
|---|---|---|---|---|---|
| R1W V1 (T304) | exact walks + declared closure | HP2 | 12.1k (written convention) | 14.92 [10.74, 19.38] | fail |
| T317 | continuity restarts + executed prefix tree, G=48; estimates = V1 to 6.7e-16 | HP2 | 7,880 | 9.71 [6.99, 12.61] | point pass, not a candidate |
| T317b | G=20 | HP2 | 6,405 | 7.89 [5.68, 10.25] | point pass, not a candidate |
| T317c @c70e22cd | exact cost-only savings | HP2 / **HP3** | 5,489 / 5,427 | 6.76 [4.87, 8.79] / **5.08 [3.42, 7.04]** | HP3 fresh, scored once: RMS .805 pp, bias +.415 pp |
| T317d | model-free tables precomputed; exact-law transverse points | HP3 / HP2 | 5,431 | 5.07 [3.41, 7.03] / 6.75 [4.86, 8.78] | |
| T317e (T325) | self-contained clean-up | = T317d | 5,490 (DEV2) | = T317d | bitwise identical to T317d; no random draws |
| **T317f (T335)** | T317e + T330's C1 frozen-gate margin screen (declared) | **HP4** | **3,302** | **2.63 [1.93, 3.39]** | fresh, scored once: RMS .749 pp, bias +.436 pp, n* 1,255. Lenient pass, strict fail |

- **T317f sensitivities and checks:** screen-charge audit F 2.67 [1.96, 3.45]; stacked counts C 3,743, F 2.98
  [2.19, 3.85]; HP4 plain-MC control .83-1.25 (every CI covers 1). Secondary scores on spent panels: HP3 3.22 [2.19,
  4.42], HP2 4.41 [3.21, 5.66]. DEV2: RMS .827, bias +.489, F 3.14. It was chosen among ~6 DEV2 variants (forking-paths
  note in its README).
- **Deductive status (T324 audit, second agent):** sampling-free at runtime (RNG-patched runs are clean), but the
  approximation scheme was selected and validated by MC on DEV2, so it is **not deductive in Hilton's strict sense**.
- **Verification:** R1W V1 and T317c were re-implemented by another agent (T312, T320; the same agent did both):
  T317c's estimates agree within .0077 pp, and the cost has ~9% latitude. H-L1 is recorded as *measured on sealed panels*, not confirmed.
- **Scope:** retrains of the zoo configuration only. T317c on HP1's noisy copies scored F 39-206. The lenient bar is a
  project convention.

### What limits it: the error budget (T324; product and tmax terms confirmed by T334)

Removing one approximation at a time from V1 (Delta bias, Delta MSE), 80 DEV2:

| Approximation | T324 | T334 (independent code) |
|---|---|---|
| line-set product | -.57 pp, **-40%** | -.562 [-.623, -.502], -41% [-57%, -20%]; share across models 20-57% |
| tmax .5 cut | -.18 pp, **-18%** | -.183 [-.235, -.136], -20% [-33%, -1%] |
| blind g12 | -.07 pp, -9% | |
| mu-linearisation | +.02 pp, +2% | |
| single transverse point (alone) | +.27 pp, +25% (interacts with the product) | |

The walk approximations carry 66% of the MSE and the **closure c 34%**. The product overstates accuracy by +.30-.34
pp, and the single anchor understates it by -.34, so the two cancel at one anchor (Addendum A).

### Strict-bar lanes since T317e (80 DEV2 unless noted; all unverified unless noted)

| Lane | Idea | Outcome |
|---|---|---|
| T326 | cost-error frontier over 80 knob configs | **negative**, scoped to the family's knobs. None beats T317e. The structural floor is ~2.2k FPE; an ideal cap there would give strict F ~1.9 at unchanged RMS (DRI correction) |
| T330 / T330b | rigorous safe-to-tmax screens (interval propagation, per-gate-pattern operator norms) | **negative**: exact, but best C 5,525 / 5,561 vs 5,490. Tier C (a declared heuristic) became T317f. Zonotopes untried |
| T331 | raise tmax (exact walk beyond .5; affine extension) | **GO = NO**: best RMS .669 (tmax 1 + extension) at C +53%, MSE x C 1.08x T317e; with C1: RMS 1.034, bias -.708 |
| T327 | closure re-check and untried variants (oracle) | T319 reproduced (T319 **dead, verified, scoped**). Arm B (one-pattern spread lows): **GO**, closure floor .482 -> .343 pp at no added cost (reverse pattern .255) |
| T339 | spread closure in the estimator (T317g) | **NO FREEZE**. For s=-1, error sd .667 -> .455 and corr with truth .978 -> .991, but bias +.758 (T317f +.489), RMS .884, F 4.04: the closure's negative bias had been cancelling part of the walks' positive bias **[T343, 14:15: does NOT replicate on fresh DEV3: sd ratio .912 [.76, 1.11] vs DEV2 .68; the variance cut was DEV2-specific]** |
| T333 | replace the product | No zero-cost rule works. The exact joint set J2 adds ~21k FPE, ~17x the ~1.2k strict allowance: **out of reach by arithmetic**. The net overstatement (+.574 pp) does not track accuracy and lives in the L-band near both tie walls |
| T347 | first-principles tilt correction of the product | **NO GO** on the pre-registered rule (mean abs error .230 vs go .1). Post hoc: bias +.439 -> -.013, RMS .796 -> .667 at ~1.4k FPE extra (4.3k crudely counted) |
| T351 | T317f + spread + tilt, four arms | **NO FREEZE**: the selection keeps T317f. Spread+tilt has the family's best RMS (.609) but costs 6,543 FPE |
| **T353** | cost-only rewrite of spread+tilt (reuse the crossing-cell state) | **GO on DEV2 (unverified)**: estimates bitwise equal to arm A3; C 5,079 FPE; bias +.413, sd .451, RMS .609; F 2.60 (T317f 3.14). Frozen as `t351/estimator_h2.py`. A pre-registered DEV3 check is pending; if it is confirmed, it is the candidate for HP5 |
| T338 / T341 | why the spread works | weight-chosen lows no better than recency (NO GO); the running-max reading is **refuted**. The reason is open |
| T314 / T332 | information ceilings (inductive diagnostics) | best <= 4k-FPE features .855 pp; ridge on T317e's own outputs .577 pp [.46, .68] |

Strict needs RMS <= ~.38 pp at ~5k FPE (T351). T343 (DEV3) independently re-checks the T339 decomposition.

### Noise robustness (T337, unverified)

| Relative sigma | 0 | .0005 | .001 | .0025 | .005 |
|---|---|---|---|---|---|
| T317f F | 3.3 | 3.1 | 4.0 | 12.0 | 32.4 |

The breakdown sits **between .1% and .25% relative noise**. Noise creates errors far from the tie faces, which the
geometry assumes correct: interior errors with both gaps >= tmax grow from .09 to .76 pp, and rank-4+ picks from .97 to
3.2 pp.

### Rare events (owner priority; T340 partial positive, unverified)

- **Truths:** GPU float64 (orig 1e10, seeds 1e9); an independent CPU run agrees within 1-1.7 s.e.
- **Events:** E1(t0) = rank-3 pick with g23 > t0 (far-gap islands); E2(t) = wrong with g12, g23 >= t.
- **Variant A** (T317's transverse point, 11-19k FPE): E2 within 3x in 6/6. It misses the E1 islands (100x-1e14x low).
- **Variant B** (uncut walks, per-node transverse points, 0.21-0.32M FPE): E1 est/truth .64-.89 on the seeds and
  .20-.29 on orig. It is one-signed low, which the diagnostic blames on walk-line placement.
- **T340c (T349, 14:15):** variant B with g12 at K exact-law Gauss nodes: K3 E1 8/9 within 2x (orig 1.73 / 1.24 / .49,
  seeds 1.02-1.27), E2 6/6 within 2x (one-signed high); K5 9/9 E1 within 2x; C 1.8-4.5M FPE; equal compute 4 win /
  3 tie / 8 loss of 15 (wins where pC <~ 90). Cost lever T354.
- **Equal compute (post hoc, executed C):**
  - variant A wins 5, ties 7, loses 3 of 15;
  - variant B wins 5, ties 3, loses 7;
  - the better variant per event family wins 10, ties 3, loses 2;
  - no single estimator wins a majority. The DRI's earlier "12 of 15" was corrected.
- **Open:**
  - T349 (T340c): g12 at exact-law Gauss nodes for variant B.
  - T350: independent re-implementation on >= 6 DEV3 models, with the family rule and a go line of >= 2/3 of cases
    pre-registered.
- **T344 (certified upper bounds): scoped feasibility negative (unverified).**
  - Interval certificates hold only on boxes <= 1e-4 wide, even in 3-D.
  - Far-gap sub-cone enumeration projects to ~2e8 regions at step 10.
  - The only rigorous bound, P(g23 > t0), is 8,200-27,500x looser than the target's truths.
  - Not pursued: a bound conditional on the declared closure.

### Exact route: dead (H-E13; DRI ruling, scoped to region/coverage enumeration)

- **Prefix counts** 34 / 792 / 21,032 for k = 2..4 (confirmed); 632,394 at k = 5.
- **T270:** N6 >= 15,483,505 (exact lower bound), so g_N(5->6) >= 24.5. The estimate of 31-33 is biased low.
- **T316:** a geometry bug in estT's k >= 6 cells undercounts both masses and regions (subtree 125: 7,659 -> 8,495 with
  the patch). The patched run is ~35x slower.
- **Mass coverage** still grows >= 7.58-14.16x per step (provisional lower bounds, delta 1e-3..1e-7).
- **Extrapolation:** ~1e13 regions at k = 10. This is an extrapolation, not a proof.
- **Datacentre estimate:** exact enumeration at datacentre scale would cost roughly 1e12-1e16x sampling's compute
  ([Owner Q&A 6.3](../../OWNER_QA.md#q6-3)).

### Panels

| Panel | Row | Contents | State |
|---|---|---|---|
| HP1 | T276 | 150 models: orig + zoo seeds 0-4, 8 noisy copies each at abs sigma .05, rel .005, rel .01 | sealed; used |
| HP2 | T284 | 100 retrains, seeds 100-199 | sealed; used by the lineage (5 submissions) |
| HP3 | T321 | 100 fresh retrains, seeds 200-299 | sealed; T317c scored once, then T317d (cost-only) |
| HP4 | T328 | 100 retrains, seeds 300-399 (mean truth .92265, sd .03184) | sealed; scored once (T317f); **spent** |
| HP5 | T336 | 100 retrains, seeds 400-499 | **sealed 14:08; reserve** for one submission named by the DRI |
| DEV2 | T284 | 80 open retrains | ~40 variants so far (forking paths) |
| DEV3 | T342 | 100 open retrains, seeds 500-599, truths published | **in preparation**; used by T343, T350, T353 |

Truths are >= 1e8-sample float64 MC per model on an independent stream. Sealed truths are held only by the T281
scorer.

### Earlier results (2026-10-17 and overnight), kept for the record

- **Leaderboard v3** ([results/leaderboard_v3_summary.txt](../../results/leaderboard_v3_summary.txt), LB3): every
  mechanistic estimator loses on M_16,10 by 37x or more (rank_exact 37-450, cone_enum 56-750, observer/reliability
  190-1100, propagation >4000). Fixed-node quadrature (sci_tao, sci_feynman) wins on low-noise groups but not at harsh
  noise, and it is inductive under ARC's definition. srb no longer beats sampling (0.78-1.97). Leaderboard v2
  ([`m16/leaderboard.py`](../../m16/leaderboard.py)) is superseded.
- **Cheap frontier (T286):** at their cheapest settings, no earlier deductive estimator came within 7x of the lenient
  bar.
- **Small models:** for M_4,3 a mechanism-derived estimator beats sampling (MSE ratio 0.019, E36; H-U1 confirmed,
  scoped). The ladder passed (8,4) (E46) and failed (8,5) on seed 2 (T192, unverified). E48's cost projection for
  M_16,10 is an extrapolation (~6.5e16-4.7e17 ops), not a bound.
- **Length scan (H-E11, confirmed):** tie-slicing quadrature beats sampling by 10^2-10^3x at T = 3-4 and collapses
  from T ~ 5-6. T277 locates the collapse in the prefactor (unverified). These wins are quadrature-class (fixed Sobol
  nodes), not deductive (T263).

### Key facts established (ledger IDs)
- H-M7: 1e7+ linear regions (Chao1 lower bound 3.8M, still rising). H-E13: exact enumeration is infeasible (see
  above).
- H-E14 (confirmed, k <= 4): training does not reduce the region count. H-E15 (confirmed, k <= 4, T <= 4): training
  concentrates the regions at the tie walls.
- H-E7: gate pieces are much finer than the correct/wrong pattern. H-E8: even an oracle additive description of the
  tie band misses by 4%.
- H-E9 (refuted as root cause): margin cancellation is real but does not track failure. H-E10: deterministic summaries
  of the low inputs overestimate accuracy (60/60).
- H-E12: the rare wide multi-cell bands are the crux.
- H-L1: the best sampling-free estimator (see the scoreboard).
- H-E5: the reported ARC solution is unverified.

### Corrections made along the way
- **Q1 study (R1):** the verdict softened to inconclusive; the random-direction control was neuron-dominated; the
  one-job proxy null was added.
- **Estimators:** stratified gains corrected to 2.1/1.8/1.4x; tube's high-budget gain not confirmed (bias floor).
- **Mechanism:** "harmonic series" weakened; "all 16 neurons confirmed" corrected to 13 in groups; the mode-based
  "most damaging" flag withdrawn (T241).
- **Truths:** the original model's truth fixed by pooling 28M samples; a "62x" figure retracted as an arithmetic error.
- **2026-10-18:**
  - E48's ">=1e17" restated as an extrapolation.
  - The k = 6 count found completion-censored, and quadrature wins reclassified as inductive (T263).
  - The cost convention written down after T312 found a charging discrepancy.
  - T320 found ~9% cost latitude.
  - T326's child F 4.8/6.3 corrected to ~1.9 (F scales linearly in C).
  - T340's "12 of 15" corrected to 10 wins / 3 ties / 2 losses for the post hoc family rule.

**Independence:** until 2026-10-17 no ARC AlgZoo material published after January 2026 was read. On 2026-10-17 the
project owner relaxed this rule (ARC's M_16,10 solution is reported to be unpublished). Every later ARC source read is
logged in [lit/web_log.md](../../lit/web_log.md). [`m16/arc_eval.py`](../../m16/arc_eval.py) and the T281 scorer can score any
estimator with the same interface, for a direct comparison.

**ARC contact:** the owner's decision (owner ruling 2026-10-18 00:42: no contact without a candidate solution;
T41/T203). **Superseded by the owner directive of 23:4x: agents never contact ARC; the owner contacts them when they judge it right.**

**Next steps** (prioritised and claimable in [TASKS.md](../../TASKS.md); details in [`INDEX.md`](../../INDEX.md) section 5):
1. The rare-event lane first: T340 with T349, then the independent T350.
2. The T353 DEV3 check, which decides the HP5 submission.
3. T343.
4. Building the DEV3 panel (T342; HP5 is sealed).
5. Then the owner's backlog ideas: T352, T348, T345, T346.

Background reading: [m16/SCALING_OBSTRUCTIONS.md](../../m16/SCALING_OBSTRUCTIONS.md) (section 9 covers the cheap regime),
[DRI_HANDBOOK.md](../../DRI_HANDBOOK.md), and [OWNER_QA.md](../../OWNER_QA.md) (every owner question, with answers and later
updates).
