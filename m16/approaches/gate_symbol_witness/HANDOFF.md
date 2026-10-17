# T138 bounded handoff

Claim a678be7 preceded implementation. Research chain899b535 ->5ae0493 ->b326469 ->ef7963a ->818f167 ->final handoff. Independent T55 check_gate_symbol_review.py/results reconstructs actual recurrence, global Lipschitz radius and margin coefficients by reverse adjoint (different from author basis propagation). All checks pass; reviewer commit notification pending.

One original H16T10 target pair, exact binary weights and rational inputs, proves top3 values/indices + full input rank + all160 gate bits cannot determine correctness. Same top3 at indices0,3,6 equals1,63/64,31/32; winners3 and6. Common positive open radius8.28669939158e-10 gives open Gaussian support and a formal positive Bayes-error floor for this summary only, not a useful numerical floor.

Constructive rescue: exact affine margin3-6 uses all seven lower inputs; a single weighted scalar determines correctness on the certified segment, switching once atlambda=.6988823149957404. Other outputs lie at least2.43219660145 below at crossing. Weighted/noisy residual summaries remain open; no global compression, estimator accuracy, cost win or method-family retirement.

All64 frozen anchors/640 LP statuses retained;8 floating positives, first certified per frozen stopping rule;56 null searches inconclusive. Synthetic analytic and LP positive/tie controls pass. Diagnostic search1.20394CPU, exact proof.031995CPU, margin.043878CPU, adapter controls.003999CPU. Initial4 tiny exact controls CPU unrecorded. Full counts in run_accounting.json, no MC/random inputs. Root/Claude DRI registry requested.

Next T141: conditional Gaussian integration along the same exact line, frozen intervals[0,1] and[1/2,1], to retain continuous uncertainty instead of replacing it with a mean. Distinguish conditional law from E27's actual omitted-input averaging; no global inference.
