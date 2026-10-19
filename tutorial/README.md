# tutorial (T374): interactive explainer of the project

↑ [Project README](../README.md) · [TASKS](../TASKS.md) · [INDEX](../INDEX.md) · [STATUS](../STATUS.md)

## Question
Owner request 2026-10-19 00:1x ([OWNER_QA](../OWNER_QA.md) section 8): a distill-style interactive tutorial, for a
reader with ML knowledge and no background in this area, that opens with our results, covers the theory, and builds
from the simplest version of the problem along every dimension we tried up to ARC's M_16,10 challenge.

## Method
- `src/page.html`: the page template (inline CSS and JS, no external scripts; Google Fonts only).
- `data/weights.json`: real AlgZoo weights (M_16,10 and zoo models), exported by `tools/export_data.py` from ARC's
  public bucket. All network runs in the page (live network, slices, exact gate walk, sampling, noisy copies) happen
  in the browser on these weights.
- `data/panels.json`: DEV3 truths (T342) and the T234 training-ceiling summary, copied from `claude/t192-gpu-cost-law`.
- `data/status.json`: the in-progress lane boxes. **The DRI edits this file**, then runs `python3 tutorial/tools/build.py`.
- `tools/build.py` embeds the data and writes `index.html` (standalone document) and `artifact.html` (the same page
  without the document wrapper, as published).
- Published as a private Artifact: https://claude.ai/artifact/22zmMZh9WU6LAPLWqJmXRX

## How to update (for this session or any later one, from this branch alone)
- Artifact URL: https://claude.ai/artifact/22zmMZh9WU6LAPLWqJmXRX (private). Re-publish `tutorial/artifact.html` to
  that URL (Artifact tool, `url` = the link above when publishing from another conversation).
- **Lane update:** edit `tutorial/data/status.json` (one entry per lane: `title`, `status`, `what`, `latest`,
  `branch`). `status` text drives the chip colour: starts with "in progress" = purple dashed box; "negative" = red;
  "confirmed" = green; anything else (e.g. "inconclusive (unverified)") = amber; finished lanes get a solid box.
  Where a lane appears: inline one-line notes via `<div data-lane="Txxx">` in `src/page.html` (sections 5, 9, 10, 12),
  and a full box for every lane in section 14, under the heading given by its `group` field (groups and lanes in
  key order of the JSON). Then `python3 tutorial/tools/build.py`
  and re-publish.
- **Weights / panels:** `python tutorial/tools/export_data.py --cache <dir> --dev3 <dev3 truths.json> --grid3 <grid3_summary.json>`
  with the two JSONs from `git show origin/claude/t192-gpu-cost-law:m16/approaches/gpu_cost_law/dev3/truths.json`
  and `...:m16/approaches/capacity_ceiling/results/grid3_summary.json` (needs torch, numpy; downloads the .pth files
  from `https://storage.googleapis.com/arc-ml-public/alg/...` into `<dir>`).
- Other numbers are typed into `src/page.html` next to their INDEX IDs; edit them there.

## For the prose rewrite (T382)
- Edit **only** `tutorial/src/page.html` (prose: headings, paragraphs, captions, table text, glossary).
- **Never change** numbers, INDEX IDs (`<span class="id">`), status chips (`<span class="st ...">`, `<span class="claim">`),
  or what a claim says (its scope and strength). Keep element `id`s, the `data-lane` divs and the `__WEIGHTS__` /
  `__PANELS__` / `__STATUS__` placeholders as they are; JavaScript strings that draw chart labels may be reworded only
  if the numbers in them stay identical.
- Rebuild with `python3 tutorial/tools/build.py` (writes `index.html` and `artifact.html`); open `index.html` in a
  browser to check. **Do not publish**; T374 checks numbers and labels, runs a review round and re-publishes.
- Lane text lives in `data/status.json`; T374 keeps updating it during T382, so do not edit it (merge conflicts).
- Published version at handover: Artifact version 1792370500-2c1e = `artifact.html` at the commit that added this
  section.

## Status
Version 11, 2026-10-19 ~23:00 UTC: coverage pass against ccr-6a61f25c-wn80ii d33a9b04 (all 333 INDEX registry and ledger rows covered). Paused by the owner except for updates they request.

## Review rounds
**Round 1 (00:35, fresh-context general-purpose agent, 40 items).** Applied: plain-language result cards and a
how-to-read note; lenient bar named as ours in the banner; "Partly understood" and "Est. 10^13-10^14"; separate chips
for H-R1 vs T366/T368; scoped "fewer pieces"; new section 4b (neuron roles; Figure 1 rows labelled by role); new
section 4c (enumerate / walk / quadrature, "face" defined) before the methods are used; start at two numbers (M_2,2)
and the rotational-symmetry reason for solid angles; small-model comparison table; corrected the tie-continuity claim
(now [inferred], with the hand-built exact network as counterexample); Fig 4/5 captions say which estimator and why
E33's T=10 win is not mechanistic; Figure 7b (pieces per unit probability near/far from ties, trained vs random,
H-E15); seeds-effect-on-method numbers (T353 DEV3 bias/sd); quantisation "no results yet" note, softened
expectations, FP4/FP8 activation row; architecture comparison table; plain-language section 10 table and uniform
"negative (unverified)"; T317 section: 720 explained, all seven simplifications listed, table legend, Fig 11 mapping
and the 101% note, strict-bar repairs as a bulleted list; rare event renamed "far-gap mistake" (no clash with E1),
example t0, quadrature explained; Fig 12 shows our T366 results and labels the repeats; Fig 6 adds k = 1, explains the
census line and the 74,000 line; calculator shows slider values and plain labels; F worked example, the 1,000 floor
and "points"; theory boxes get one-line takeaways and a skip link; #P transfer labelled "our reading"; no-coincidence
tied to the 10% vs 95% example; Fig 3 caption corrected (axes, six wedges, area vs probability), hatched errors,
labelled tie lines; Fig 1 caption softened (near-ties and mostly-negative inputs); Fig 10 caption (three lines; how a
length becomes a probability); inline lane boxes reduced to one-line notes, full boxes only in section 14, lane text
de-jargoned; glossary additions (pp, MSE ratio, quadrature, face, three-way near-tie, #P/CAPP, sealed vs open, DRI);
hero explains why ARC cares and what M_16,10 stands for.

Not applied, and why:
- (11) Remove the essay-question card: kept, because the DRI's brief requires the essay-side verdict in the opening;
  the card now says the project started from that essay.
- (16) Move IDs into tooltips: kept inline, because the brief requires every number to cite its row visibly; IDs are
  already small and muted.
- (13, partly) An interactive 90-unit construction and accuracy/region numbers for other architectures: we have no
  measured numbers for them yet (T345 is running); added a table with what we know instead.
- (14, partly) Noise selector on Figure 3 (regions under noise): not done in this round; noise is shown live in Fig 8.
- (21, partly) The no-coincidence paper's own reversible-circuit example: not in our notes, so not added; the 10% vs 95%
  example is used instead.
- (34, partly) The method at fewer neurons and T = 10: no such result in INDEX; nothing added.
- (6, partly) Fig 5's T = 3 level differs from Fig 4 because it is a random-face variant; explained in the caption
  rather than reconciled.

**Round 2 (02:20, after the T382 prose rewrite; fresh-context general-purpose agent, 27 items).** Before the round:
took `src/page.html` from e218888f and checked against a6bc61d5 that the multiset of numbers (outside scripts), all
`id`/`st`/`claim` spans, the script and the CSS are unchanged; fixed the three stale spots (T6 card, Fig 5 / T10, T345
row) and the rewriter's flagged points (HP2 served five submissions; F = 1 only at >= 1,000 runs; Fig 5's T list; the
#P takeaway; "ruled out" scoped; scaling-only symmetry attributed to the review; gain law now accuracy / (error rate x
CV^2 x c), which matches d5_law.log; Poisson described as an approximation; calculation cost "does not grow in
proportion to rarity"). Applied from the round: sections 9-10 moved to 6-7 so the simpler-networks material comes before
the full target, with sections and figures renumbered; Fig 6 band now 2e13 (x31/step) to 1e14 with the rule stated;
the 26-27x vs ~120x density numbers explained as different counts, "near" defined, 1M tick; Fig 1 caption (role tags
spelled out, three buttons); Fig 2 baselines explained, red line removed from the single-network histogram, dot labels
in words; T317f card carries the T324 "not strictly deductive" chip; "Closed" -> "Out of reach for this method";
"typical network" -> above average (~80th percentile of DEV3); exact score crossings in the gate walk, legend by
starting rank; Fig 13 hides our dots outside the tested range and explains 1.4x vs 2x; rare card states the cost and
the rough crossover; quantisation plan made visible and the non-negative-codes nit; H dimension: what was and was not
measured; section 7 renamed "Other shortcuts we tried" with a note on problem-side simplifications not tried; term
pointers in section 1 and the theory box; M_4,3 paragraph clarified; gain-law intuition; ladder paragraph; Fig 5
caption split from the status sentence; simplification names unified with Fig 12; Fig 9 caption says which models and
why noise breaks the method; Elekes box relabelled "proved for convex bodies; transfer is our reading"; lanes text no
longer implies live updates; timestamps unified; TOC title; dropped the m16/truth.txt aside; Fig 3 centred on (1, 1)
and names the network.

Not applied in round 2, and why:
- (15, 17) Fewer IDs in the cards, essay card removed: the brief requires visible IDs and the essay verdict.
- (16) One term per concept: partly (pointers added); a full terminology pass would undo the T382 prose; left for a
  later round.
- (21) A diagram of the T317 walk geometry and the per-walk cost: not added this round (Fig 11 shows one walk).
- (27, part) Splitting section 2 further: the T382 rewrite already restructured it.

**T384 fact-check (04:10).** Took `src/page.html` and `REVIEW_RESPONSE_T384.md` from the shared branch (17 adopted
findings, see that file). Checked against v4 (0039119b): status chips and CSS identical; script differs only by the five
declared display strings and one comment; the ID change (T337 -> H-E4 · T337) and three claim-label rewordings are
declared; changed numbers are the documented corrections (spot-checked 11/1,024, 44,700 runs, 3,743/F 2.98, 44.1/37.2
against sources).

**Round 3 (04:20, fresh-context agent, 15 items, short round).** Applied: Christiano named in the no-coincidence box
with the plain takeaway first; section 13 no longer points to representation lanes that are closed; T372 split into
the failed deterministic gradient and the sampling-class n6/n7 finding, plus a T373 sentence; T10 lane updated to
confirmed via T379/H-E16; growth rates unified (x23-31; effective 5.1 -> 3.6); the two fixed-grid T = 10 results told
apart (E33 1.6x on zoo models vs T10 recipe .97 on DEV3); Figure 13 now plots the independent K3 re-run (T368, ten
ratios 0.88-1.39 from rare_deep/results/deep.json) instead of the graded rule; rare-event card says "within 2x" is
accuracy, not a win over sampling; E1/E2 renamed in section 12; feed-forward prose now states the measured outcome
(as many pieces as the RNN) and defines the circle walk; float baseline and FP4/FP8 explained; section 9 jargon
(leaf, assigned mass) moved into a parenthesis; scoring paragraph split (E32 accounting into a note; C defined
plainly); legend explains pass/done/inconclusive/scoped; lane text de-shouted; section 14 grouped by theme via a new
`group` field in status.json; Figure 2 lines clipped to the axes.
Not applied: moving 4b after section 5 and a static face diagram in 4c (structural; the 4c text points to Figure 11).

**Final update (22:35, owner request d-1019-223x).** Banner and byline now say: two candidates, ARC acceptance unknown.
New result card for the line walks (H-E17, T431 sealed strict pass; T429 budget curve; T449/T451 no weight-derived bound)
and for the certified bound (T416 <= 1.36e-11 provisional; T400; T418/T455); rare card adds T370 stage 1 and H-R3;
candidates table rebuilt around H-E17 with ARC's "competitive with sampling" wording (OWNER_QA 4.13); new 1c "What is
new here" crediting prior work as lit/novelty_review.md does; section 5 adds T429/T431/T393; section 6 T352 negative and
the feed-forward row final; section 12 adds T436/T453/T454 and the safe-use rating; section 13 rewritten (what remains)
plus "Beyond this network" (T419, T421, T433, T444, T445-T447, T443). status.json: 47 lanes, finished ones marked with
their INDEX status. No redesign, no new review round (owner: one republish).

## Results
Not a research result. Every number in the page cites its INDEX row and status. Numbers computed by this folder:
the "quick_acc" accuracies in `data/weights.json` (1e6 samples, seed 0; M_16,10 .9531, M_4,3-size zoo(4,3,0) .9843,
zoo(2,3,0) .4098), reproduced by `python tutorial/tools/export_data.py --cache <dir> --dev3 <truths.json> --grid3 <grid3_summary.json>`.
Figure 5 uses the draft T10 log `m16/approaches/minimal_pair/d5_law.log` (claude/t10-minimal-pair e70e219a).

## Compute accounting
Export: < 1 CPU-minute. Page: everything runs client-side (sampling widgets use up to 20 x 1e5 forward passes on demand).

## Claims
- [measured] The exported weights reproduce M_16,10's accuracy (.9531 on 1e6 samples vs the T192 answer key .9531175).
- [inferred] The page's explanation of why errors concentrate at ties (a continuous score cannot jump at the tie
  plane) is marked in the page as "our explanation".

## Alternatives not ruled out
The theory section adds two standard references not in our notes (Valiant 1979; Dyer and Frieze 1988) for the
#P-hardness statement; everything else is from OWNER_QA 1.3, SCALING_OBSTRUCTIONS section 6 and lit/web_arc.

## Failure modes
- The rare-event figure simulates sampling statistics (Poisson draws), it does not run the rare-event estimator.
- The region-growth band to k = 10 and the data-centre calculator are extrapolations, labelled as such.
- Lane boxes go stale unless `data/status.json` is updated and the page rebuilt and republished.


## Coverage checklist (v11, INDEX at ccr-6a61f25c-wn80ii d33a9b04)
Every INDEX section-3 registry row and section-4 ledger row (333; the four Q-rows of section 1 excluded). "body" = the
ID appears in the page's prose, cards, tables or lane boxes; "register" = one plain line in section 15, from `data/findings.json`
 (drafted by helper agents from the row text; every number checked to appear in its row). To diff a
later INDEX: re-extract the IDs from sections 3-4 and look for IDs missing from this table.
New in v11: section 1a (most interesting findings, with why each matters), 1d (how far it scales: OWNER_QA 4.17, 4.18),
section 15 (register). Not re-reviewed by a fresh-eyes round (owner: coverage pass).

| ID | INDEX section | where |
|---|---|---|
| T114-review | registry | register |
| T110-T4 | registry | register |
| T105-two-pair | registry | register |
| T116-bulk | registry | register |
| T117 | registry | register |
| T49-three | registry | register |
| T105-signed | registry | register |
| T20-cold | registry | register |
| T67-measure | registry | register |
| T107 | registry | register |
| T105-positive | registry | register |
| T110-initial | registry | register |
| T91-interface | registry | register |
| T81-enrichment | registry | register |
| T82-switches | registry | register |
| T37-hidden | registry | register |
| T66-geometry | registry | register |
| T90 | registry | register |
| T78-directed | registry | register |
| T104/T106 | registry | register |
| T66-reset | registry | register |
| T76-history | registry | register |
| T102 | registry | register |
| T50-reset | registry | register |
| T49-reset | registry | register |
| T27 | registry | register |
| T26 | registry | register |
| T28 | registry | register |
| T45 | registry | register |
| T44 | registry | register |
| T40-support | registry | register |
| T99 | registry | register |
| T66-scalar | registry | register |
| T39 | registry | register |
| T57 | registry | register |
| T56 | registry | register |
| T30 | registry | register |
| T81 | registry | register |
| T82 | registry | register |
| T76 | registry | register |
| T83 | registry | register |
| T52-T6 | registry | register |
| T72-H4 | registry | register |
| T62 | registry | register |
| T71 | registry | register |
| T72-H2 | registry | register |
| T58-closure | registry | register |
| T48-T6 | registry | register |
| T52-T5-cost | registry | register |
| T68 | registry | register |
| T59 | registry | register |
| T61/T65 | registry | register |
| T64 | registry | register |
| T52-T4-cost | registry | register |
| T48-T5 | registry | register |
| T53/T54 | registry | register |
| T47-small/T51 | registry | register |
| E1 | registry | register |
| E2 | registry | register |
| E3 | registry | register |
| E4 | registry | register |
| E5 | registry | register |
| E6 | registry | register |
| E7 | registry | register |
| E8 | registry | register |
| E9 | registry | register |
| E10 | registry | register |
| E11 | registry | register |
| E12 | registry | register |
| E13 | registry | register |
| E14 | registry | register |
| E15 | registry | register |
| E16 | registry | register |
| E17 | registry | register |
| E18 | registry | register |
| E19 | registry | register |
| E20 | registry | register |
| E21 | registry | register |
| E22 | registry | register |
| E23 | registry | register |
| E24 | registry | register |
| E25 | registry | register |
| E26 | registry | register |
| E27 | registry | body |
| E31 | registry | register |
| E28 | registry | register |
| E29 | registry | body |
| E30 | registry | register |
| E32 | registry | body |
| E33 | registry | body |
| E34 | registry | register |
| E35 | registry | body |
| E36 | registry | body |
| E37 | registry | body |
| E38 | registry | register |
| E39 | registry | register |
| E40 | registry | register |
| E41 | registry | register |
| T86 | registry | body |
| T111 | registry | body |
| T428 | registry | register |
| T443 | registry | body |
| E42 | registry | register |
| E43 | registry | register |
| H-U5 | registry | register |
| H-U6 | registry | register |
| E44 | registry | register |
| E45 | registry | register |
| E46 | registry | body |
| E47 | registry | register |
| E48 | registry | body |
| T196 | registry | register |
| chamber_split | registry | register |
| T192-S1/S2 | registry | register |
| T194 | registry | register |
| T19-R2 | registry | register |
| T20-P1 | registry | register |
| T29 | registry | register |
| T31 | registry | register |
| T32 | registry | register |
| T33 | registry | register |
| T34 | registry | register |
| T37-P1 | registry | register |
| T195 | registry | register |
| T241 | registry | body |
| T269 | registry | register |
| T260 | registry | register |
| T268 | registry | register |
| T278 | registry | register |
| T285 | registry | register |
| T285F | registry | register |
| T280 | registry | body |
| T286 | registry | register |
| T298 | registry | register |
| T308 | registry | register |
| T290 | registry | register |
| T305 | registry | register |
| T303 | registry | register |
| T314 | registry | register |
| T309 | registry | register |
| T310 | registry | register |
| T317 | registry | body |
| T326 | registry | body |
| T324 | registry | body |
| T333 | registry | register |
| T347 | registry | register |
| T351 | registry | register |
| T353 | registry | body |
| T356 | registry | body |
| T361 | registry | register |
| T330 | registry | body |
| T331 | registry | register |
| T332 | registry | register |
| T323 | registry | register |
| T284/T321/T328/T336 | registry | register |
| T342 | registry | body |
| T234 | registry | body |
| T320 | registry | register |
| T319 | registry | register |
| T327 | registry | register |
| T339 | registry | register |
| T338 | registry | register |
| T341 | registry | register |
| T316 | registry | register |
| T337 | registry | body |
| T340 | registry | register |
| T348 | registry | body |
| T395 | registry | body |
| T413 | registry | body |
| T358 | registry | register |
| T349 | registry | register |
| T354 | registry | register |
| T359 | registry | register |
| T360 | registry | body |
| T364 | registry | body |
| T366 | registry | body |
| T368 | registry | body |
| T367 | registry | body |
| T369 | registry | body |
| T372 | registry | body |
| T402 | registry | body |
| T410 | registry | body |
| T411 | registry | body |
| T386 | registry | body |
| T405 | registry | register |
| T401 | registry | body |
| T407 | registry | body |
| T436 | registry | body |
| T453 | registry | body |
| T454 | registry | body |
| T456 | registry | body |
| T370 | registry | body |
| T385 | registry | body |
| T373 | registry | body |
| T390 | registry | body |
| T398 | registry | body |
| T400 | registry | body |
| T416 | registry | body |
| T399 | registry | register |
| T414 | registry | register |
| T418 | registry | body |
| T412 | registry | register |
| T415 | registry | register |
| T417 | registry | register |
| T384 | registry | body |
| T371 | registry | body |
| T388 | registry | body |
| T403 | registry | register |
| T404 | registry | register |
| T291 | registry | body |
| T375 | registry | body |
| T3 | registry | body |
| T376 | registry | body |
| T10 | registry | body |
| T11 | registry | body |
| T379 | registry | body |
| T393 | registry | body |
| T429 | registry | body |
| T431 | registry | body |
| T449 | registry | body |
| T451 | registry | body |
| T380 | registry | body |
| T389 | registry | body |
| T391 | registry | body |
| T397 | registry | register |
| T46 | registry | body |
| T352 | registry | body |
| T345 | registry | body |
| T408 | registry | body |
| T392 | registry | body |
| T419 | registry | body |
| T421 | registry | body |
| T423 | registry | body |
| T427 | registry | body |
| T433 | registry | body |
| T444 | registry | body |
| T387 | registry | body |
| T394 | registry | body |
| T24 | registry | body |
| T377 | registry | body |
| T6 | registry | body |
| T383 | registry | body |
| T382 | registry | register |
| T363 | registry | body |
| T365 | registry | register |
| T362 | registry | body |
| T344 | registry | body |
| T301 | registry | register |
| T318 | registry | register |
| T315 | registry | register |
| T312 | registry | register |
| T311 | registry | register |
| T307 | registry | register |
| T304 | registry | body |
| T302 | registry | register |
| T300 | registry | register |
| T297 | registry | register |
| T296 | registry | register |
| T292 | registry | register |
| T279 | registry | register |
| T289 | registry | register |
| T299 | registry | register |
| T295 | registry | register |
| T294 | registry | register |
| T293 | registry | register |
| T287 | registry | register |
| T282 | registry | register |
| T277 | registry | register |
| T281 | registry | register |
| T261 | registry | register |
| T276 | registry | register |
| T274b | registry | register |
| T274a | registry | register |
| T263 | registry | body |
| T271 | registry | register |
| T272 | registry | register |
| T216 | registry | register |
| T217 | registry | register |
| T218 | registry | body |
| LB2 | registry | register |
| LB3 | registry | register |
| T42-R1 | registry | register |
| R1 | registry | register |
| R2 | registry | register |
| R3 | registry | register |
| R4 | registry | register |
| R5 | registry | register |
| L1-L8 | registry | register |
| L9 | registry | register |
| L10 | registry | register |
| H-Q1 | ledger | register |
| H-Q2 | ledger | body |
| H-Q3 | ledger | register |
| H-Q4 | ledger | register |
| H-Q5 | ledger | body |
| H-A1 | ledger | body |
| H-A2 | ledger | register |
| H-A3 | ledger | register |
| H-M1 | ledger | body |
| H-M2 | ledger | register |
| H-M3 | ledger | register |
| H-M4 | ledger | register |
| H-M5 | ledger | register |
| H-M7 | ledger | register |
| H-E6 | ledger | register |
| H-M8 | ledger | register |
| H-E7 | ledger | register |
| H-E8 | ledger | register |
| H-E9 | ledger | register |
| H-E10 | ledger | register |
| H-E11 | ledger | body |
| H-M6 | ledger | register |
| H-E1 | ledger | register |
| H-E2 | ledger | register |
| H-E3 | ledger | register |
| H-E12 | ledger | register |
| H-E4 | ledger | body |
| H-U1 | ledger | body |
| H-U2 | ledger | body |
| H-U3 | ledger | register |
| H-U4 | ledger | register |
| H-L1 | ledger | body |
| H-R1 | ledger | body |
| H-R2 | ledger | register |
| H-R3 | ledger | body |
| H-C1 | ledger | body |
| H-S1 | ledger | body |
| H-E15 | ledger | body |
| H-E16 | ledger | body |
| H-E17 | ledger | body |
| H-E14 | ledger | body |
| H-E13 | ledger | body |
| H-E5 | ledger | register |
