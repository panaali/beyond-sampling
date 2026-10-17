# Compiled guarded cofactor gate

## Question
Does a strictly compiled fast kernel preserve the existing reviewed enclosure and what arithmetic does it execute?

## Method
PROTOCOL.md freezes existing25+64 controls, build flags, fenv rejection and explicit fallback ownership. guard.c ports the separate binary64 trees. build.py saves compiler/source/library/disassembly pins without executing the kernel.

## Status
Bounded controls and saved-path contraction complete; final independent raw review pending. Claimf058ce6f. Independent source gates preceded build, execution and ledger contraction.

## Results
All81 fast inputs preserve point, outward endpoints, permanent and allowance bit for bit; five finite out-of-range inputs request the existing exact fallback and reproduce its saved outputs; three nonfinite inputs remain unknown. Four altered-environment probes reject and restore original rounding/MXCSR.93 kernel calls total, no new matrices or scientific sampling. Existing point cancellation failures and wide intervals are unchanged.

GCC14.2 build pins and disassembly are in build1/. The ordinary nonzero-allowance fast path executes31 scalarFP arithmetic operations. For dense eligible inputs it also executes37scalarFP comparisons and18packed bitmask instructions, plus2external nextafter calls and1fegetround call. These categories are not a complete cost; no speedup against the older104upper subset follows. Across89normal controls the separate counts are2507arithmetic,3017comparisons,1575bitmasks,158nextafter calls and89fegetround calls. Four environment probes are excluded from this ledger. ASSEMBLY.md and instruction_ledger.json preserve assumptions.

Commands, in a fresh output copy (scripts refuse to overwrite):
```sh
/workspace/clean-neurons/.venv/bin/python m16/approaches/cofactor3_guard/compiled/build.py
/workspace/clean-neurons/.venv/bin/python m16/approaches/cofactor3_guard/compiled/controls.py
/workspace/clean-neurons/.venv/bin/python m16/approaches/cofactor3_guard/compiled/instruction_ledger.py
```
Original runs used one thread,2GB/20CPU caps and `/workspace/work/codex-review.lock`. Binary is git-ignored; its source/flags/hash and exact assembly are committed for reproducibility. Parent Python theorem and exact control reviews cb751402/fd0dabc7 remain prerequisites.

## Compute accounting
Zero model/mass/MC/native determinant calls. Controls .023583684CPU, excluding imports and final serialization. Saved-only instruction ledger reports0CPU at clock resolution, not free work. One successful build; build CPU was not instrumented. Five charged Python exact fallbacks and64 Python comparator calls are explicit validation work. External nextafter/fenv, interpreter/control/memory and exact fallback costs are not automatically priced.

## Claims
- [measured] Fixed source/control scope frozen before build/testing.
- [measured] Existing89inputs and4environment probes pass the declared bit/status controls.
- [inferred] Bit parity plus strict assembly can support applicability of the reviewed binary64 theorem under explicit runtime assumptions.
- [speculative] A compiled enclosure could reduce primitive overhead; no speed or total-mass claim yet.

## Alternatives not ruled out
Caller environment changes, library behavior, cheaper native paths, fallback frequency and downstream interval propagation remain open. The104 native envelope is not a lower bound.

## Failure modes
Fast-math/reassociation/fusion, unsupported rounding/FTZ, undocumented fallback and ignored library work can invalidate stronger conclusions. Existing point cancellations are retained; enclosure is not an accurate-point acceptance test.
