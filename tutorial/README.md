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
Version 5 (T382 prose, T384 fact-check, three review rounds), 2026-10-19. Stays open for status updates from the DRI.

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
