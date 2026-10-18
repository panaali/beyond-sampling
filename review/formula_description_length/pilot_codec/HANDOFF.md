# T260 fixed-pilot description lengths independently verified

The bounded numerical codec phase completed once and independently passed
47ab09d7, handoffc0d5cf25/shared1c36ac69. Sources44c7450e, source/resource
review2b314033, rawd30fe7ab/sharedcb169ca1. Immutable gatebc6e9aa1 reached
sharedf01a4b30 before the separate producer launch. Independent byte-level parser,
stdlib NPY binding and AST normalization verify all eight exact frames,32controls,
23archives/68arrays,20producer events and20reviewer events. Canonical independent
results: review/t260_codec_saved_review/output_v2/results.json (56's tree).

| Fixed representation | Input → hidden bits | Input → labels bits | Inherited pilot evaluation agreement |
|---|---:|---:|---:|
| all12 selected tables | 28,408,630 | 28,419,477 | 924/1024 |
| no_hinge8 selected tables | 433,558 | 444,405 | 975/1024 |
| Original network source-style code | 24,520 | 35,367 | Reference network; no new score |
| H-A1 formula source-style code | 39,480 | 50,327 | Not scored on this pilot |

All twelve-family validation choices reduce to selected families3,8,11 plus nine
comparator-bank tables. The no-hinge choices select family3 for all16 outputs.
The declared code includes each selected family's complete16 output columns,
pooled fallback, ordered cell IDs, exact binary64 coefficients, output routing and
necessary bank tables once. This intentionally conservative full-column overhead
preserves the original BLAS shape; column-pruned precision/parity was not tested.
The normalized feature/inference AST and framed payload are both charged.

These are upper code lengths under the unchanged T25 grammar, conditional on its
Python AST schema/decoder and NumPy numeric semantics. They are not shortest code,
interpretability measurements, a full cost comparison or a population rate curve.
Other encodings, dictionaries, approximations and predictors remain possible.
The H-A1 figure includes a common all16-output wrapper and optional readout; it
therefore differs from the earlier T25 single-entrypoint36182-bit figure. Readout
conditioning is shown separately and consistently. No H-A1 agreement is transferred.

Inherited T258 pilot evaluation contains1024 fresh fixed inputs; all12 taskcorrect
945/1024 and no_hinge8 taskcorrect1003/1024 are distinct from agreement. These
observed pilot results do not become fresh validation by re-encoding artifacts.
There is no new fitting, selection, pruning, quantization, predictor evaluation,
RNN or RNG call. This numerical phase resolves the missing-artifact prerequisite
only for the newly retained pilot, not historical C2 or a full reproduction.

The executable specification is a valid-finite-input inference upper code. Exact
source/payload roundtrips do not establish general floating predictor parity,
including behavior of unselected-overflow checks. The previous T258 regression
publication-order violation remains unchanged; this later correctly ordered launch
does not retroactively repair it.

Compute:23 retained inference archive loads,32 fixed controls,eight framed
encode/decode pairs,20 events;57,855,764 bits and7,231,972 packed bytes total.
Producer internalCPU6.191943126,RSS277172KiB; external6.38wall/5.41user/.96sys.
One105soft110hardCPU2GiB120wall process, no failures/retries. One synthetic routing
call; no actual predictor evaluation. Independent review timing remains separate.

Independent reviewer recoveryCPU3.088197805,RSS200816KiB. Its first attempt
stopped on a progress-logger argument collision after32controls/68array bindings,
before artifact decode;1.033101265CPU is retained. Total recorded reviewer CPU
4.12129907. That reviewer-only error and h4 initial source-gate coverage miss are
preserved in review/t260_codec_saved_source/RECOVERY.md. Recovery changed only
logging/path/previous-attempt accounting, passed2a81c8b8, and immutable01795a5a
was publishedfaa10bc6 before execution. No producer replay or numeric changes.

Bounded T260 phase complete. Please register only this retained-pilot declared-code
result in INDEX. Full historical reproduction, optimal description length,
generalizable agreement and the main deductive accuracy challenge remain open.
