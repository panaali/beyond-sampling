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

## Status
First full draft, 2026-10-19. Stays open for status updates from the DRI.

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
