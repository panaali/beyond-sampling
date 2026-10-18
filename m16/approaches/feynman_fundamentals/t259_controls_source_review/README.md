# T259 exact margin-loss controls: source/resource review

## Question
Can the frozen seventeen-case exact-rational suite distinguish common-design/readout and matched all-pair quadratic equivalence from its explicitly separating exceptions?

## Method
Read algebra.py, fixtures.py, run.py, launch.sh, PROTOCOL.md, protocol.json and README.md at initial06c7cf14 and documentation correction22b2f8e4. Inspect the prior written theorem3bc9e2c3 and independent593589de artifact and byte bindings. Check every fixture analytically, without importing or executing the candidate. Metadata checks hash sources, parse ASTs, check shell syntax and Git ancestry. No scientific payload, model, array, RNG or Gaussian calls.

## Status
Source/resource gate passed for final22b2f8e4/sharedc6812df6. The original numerical source and fixtures are unchanged; the correction specifies that solved square systems, rather than every rectangular query matrix, have dimension at most4. Gate publication and a separately published immutable execution binding must precede the sole run.56 owns the separate saved arithmetic review.

## Results
[inferred] Ordinary ridge constructs X'X+diag(penalty), X'Y and exact Gauss–Jordan elimination. A first nonzero pivot, exact row normalization and elimination give the unique result when nonsingular; each original matrix/RHS is preserved before elimination. The only intended singular system has second feature identically zero and zero penalty; it rejects at the second pivot, retaining the failed attempt.

[inferred] The pair solver constructs pair features directly in the zero-sum basis V=[I | -1], with d(K-1) unknowns. Its data weights use ordered/unordered sum or mean and optional sample mean. The shared penalty block is V V' times each feature penalty. Multiplying that penalty by alpha times the sample factor matches the ordinary ridge objective in the centered subspace. Alpha is K for unordered sum,2K for ordered sum, and2/(K-1) for either pair mean. The eight K=3 cases therefore have alpha3,6,1,1 crossed with sample factors1 or1/2. Their four-dimensional systems differ from the ordinary two-dimensional comparator; the implementation does not simply return a centered ordinary solution.

[inferred] Base ridge coefficients follow directly from diagonal(2,3): rows[1,1/2] and[1/3,1]. The rectangular readout produces rows[1,1/2,-1/2] and[1/3,1,2/3]; centering gives[2/3,1/6,-5/6] and[-1/3,1/3,0]. The rank-deficient readout, common shifts, shared intercept and fixed cell/pooled routing expectations also match direct substitution. A zero query owns the first-index tie. Common shifts add the same scalar per query, preserving class differences and ties.

[inferred] Separating fixtures are nonvacuous. Unmatched penalty yields query scores[-1/30,1/30], versus matched[1/16,-1/16], reversing the label. Output-specific scalar penalties yield1 and1/2, whose sum differs from direct zero-penalty summed-target fit2. Singular nullspace choices agree on training but yield different query values, including an arbitrary summed-target value7. ReLU clipping changes[-2,-1] to[0,0], changing label1 to first-index0. These exceptions are not evidence against the restricted equivalence theorem.

## Compute accounting
Planned36 solve attempts:4 readout,2 common-shift,1 intercept,6 routing,16 pair-normalization,2 unmatched,3 output-specific,1 singular and1 ReLU. Exactly35 return; the singular attempt is the one expected failure. Seventeen cases produce36 progress events. No malformed-domain sweep is promised. The largest square solved system is4-by-4; some query matrices have five rows.

One pinned /usr/bin/python3 process with exact Fraction arithmetic, codex-review.lock, one thread,15 internal CPU seconds,16soft/20hard OS CPU,2GiB/2GiB address space and25 wall seconds after lock. The binary digest is checked before candidate imports. Internal timing excludes top-level imports and final serialization; whole-process OS caps still apply. These are research call counts, not a native-FLOP cost or trained-fit benchmark.

Metadata reproduction: `/usr/bin/python3 check_metadata.py`. It never calls the algebra or fixture functions. No numerical suite has been run by this reviewer.

## Claims
[inferred] A passing executed suite would validate these finite implementations and examples, subject to independent saved review. It would not prove universal floating commutation, a99% ceiling, feature insufficiency, failure of weighted/top-rival/nonquadratic objectives, or any trained/HP1/HP2 performance result. The prior written theorem has its own exact-real/common-map assumptions and is not an execution authorization.

## Alternatives not ruled out
Output-specific features, target-refitted banks, nonlinear losses, nonnegative constraints and recurrent rollout can differ from this common-design quadratic fit. Floating solves/casts can differ even where exact algebra commutes. Different fixed linear conventions can handle singular systems; only an arbitrary inconsistent convention is excluded by the exhibited case.

## Failure modes
The runner verifies prior/source/reviewer/protocol pins and actual limits before creating output and importing candidate code. It refuses existing output; the launcher refuses an old log. Attempt ownership prevents overwriting prior evidence on rejected repeats. Each case and solve records its start; returned coefficients and derived outputs precede consistency assertions. Atomic replacement preserves the previous JSON on interruption, while a serialization/process failure can leave a partial temporary file or last started record. It cannot count as success. The one singular rejection is explicit and expected. No automatic retry, fixture substitution, cap increase or tolerance change is cleared.
