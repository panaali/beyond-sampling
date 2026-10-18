# T303: rule-5 re-check of the "top-4 exact + spread lows" closure, and its integration cost

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T298](../r0_variants/README.md) · T297 (claude/t218-tie-tail-labels, `r0_integral/`)

## Question
1. Does T298's post hoc best closure reproduce when rebuilt from its written definition, with fresh code and
   disjoint draws? Agreement means within 20% (DRI).
2. What would a deterministic integral of it cost under T297's per-class cost model? Could it ever fit the cheap
   regime (<= 4k FPE, or the tighter <= 1.8k)?

## Method
- `recheck.py`: written from the T298 README definition only, not from `r0_variants/run.py`.
  - Own forward pass and own closure construction. The 6 free positions are taken in time order and get
    Q(Φ(v4)(7−j)/7), j = 1..6, descending and ascending; the verdict is the mean of the two points.
  - Paired bias against the network's correctness on the same x.
  - Seeds 303000+ (DEV2) and 304000+ (LB3), disjoint from T298's 298000+ / 298500+.
  - 400k inputs per model. R0 is re-measured as the reference. Regions as T292.
- Independence caveat: the code and the draws are new, but **the author is the same agent** as T298. AGENTS
  rule 5 asks for an independent agent for a verified negative or claim. The DRI may want a second agent.

## Results
Reproduce: `python3 recheck.py dev2 400000`; `PYTHONPATH=<alg_zoo shim> python3 recheck.py lb3 400000`
(about 10 CPU-min).

| panel | closure | RMS pp | mean pp | slope pp/pp | max \|b\| | blind | 2~3 | 1~2 | tail (mean / RMS) |
|---|---|---|---|---|---|---|---|---|---|
| DEV2 (80) | top-4 + spread | **.248** (T298 .244) | +.215 (+.210) | −.0109 (−.0107) | .72 (.66) | +.005/.013 | +.068/.079 | +.041/.047 | +.100/.126 |
| DEV2 (80) | R0 | .514 (T298 .515) | +.219 | −.0425 | 1.78 | +.040/.047 | +.122/.194 | +.007/.082 | +.050/.310 |
| LB3 (30) | top-4 + spread | **.386** (T298 .377) | +.297 (+.290) | −.0141 (−.0138) | .89 (.89) | +.000/.008 | +.066/.076 | +.022/.034 | +.209/.296 |
| LB3 (30) | R0 | 1.339 (T298 1.318) | +.938 | −.0644 | 2.81 | +.013/.019 | +.271/.319 | +.024/.103 | +.630/.973 |

**Agreement:** every headline number is within 2–3% of T298, against the 20% criterion. Reproduced.

The remaining bias is a systematic offset: +.21 pp on DEV2 and +.30 pp on LB3, mostly in the soft tail and the 2~3 band.
The cross-model sd is ~.12 pp (DEV2), against R0's .47.

## Integration cost under T297's model
T297 (`r0_integral/README`):
- R0 is integrated per position triple (720 classes, 3 continuous values). Its error set is not a per-triple
  threshold set: the boundary also moves with v3 and g12.
- Resolving it needs >= 30–50 evaluations per triple, i.e. >= 20–40k FPE. Fixed Gauss–Legendre work is ~290 FPE.

For top-4 + spread, the classes are ordered position quadruples: 10·9·8·7 = **5,040** (7x more), with **4**
continuous values and **2** points per skeleton.

| design | evaluations per class | classes | points | FPE |
|---|---|---|---|---|
| floor: one verdict per class and orientation, no boundary resolution | 1 | 5,040 | 2 | **10,080** |
| T297's R0 resolution, carried over unchanged (ignores the 4th dimension) | 30–50 | 5,040 | 2 | 0.30–0.50 M |
| plus a few (3) anchors along v4 | 90–150 | 5,040 | 2 | 0.9–1.5 M |
| Gauss–Legendre fixed work (4-D grid instead of 3-D) | | | | >= ~290 × nodes along v4 |

**Even the floor design is 10k FPE, 2.5x the 4k cheap bound and 5.6x the 1.8k target, before any boundary
resolution.** A design that actually resolves the boundary is 2–3 orders of magnitude over. A per-class integral of
this closure therefore cannot fit the cheap regime.

Only sharing across classes could change that, i.e. a boundary that depends on far fewer than 4 positions. T297 left
that untried.

## Claims
- K1 [measured] The top-4 + spread closure reproduces with new code and disjoint draws: DEV2 RMS .248 (vs .244),
  LB3 .386 (vs .377), slopes −.011 / −.014. It agrees within 20% (within 3%). Same author: an independent-agent check
  is still open.
- K2 [inferred, from T297's cost model] A deterministic per-class integral of this closure costs >= 10k FPE even with
  no boundary resolution, and ~0.3–1.5M FPE with T297-level resolution. It cannot fit the cheap regime unless
  boundary information is shared across position classes.

## Compute accounting
About 6e8 network forwards (3 closure points + 1 true per input; 400k × 110 models). Diagnostic sampling.
About 10 CPU-min single-threaded.
