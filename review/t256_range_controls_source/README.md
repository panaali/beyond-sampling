# T256 affine-gap primitive source/resource gate

## Question
Do the frozen exact-rational controls test the reviewed affine range theorem, including physical units, with independent references and bounded durable execution?

## Method
Delegation ba6ac722; source f659a676/shared9260a784; written theorem independently a17014a6. Read all seven source/protocol files without author imports. Verify hashes, fixture AST lengths and theorem completion with `python review/t256_range_controls_source/check_source_metadata.py`; check launcher with `bash -n`. This metadata helper executes no primitive or reference.

## Status
Source/resource gate passed for one frozen synthetic attempt after immutable published gate binding. Independent saved reviewer recover_910 remains required. No candidate execution by this reviewer.

## Results
[inferred] The exact feasible second-coordinate interval and complete clipping-kink set match the theorem. Sign-selected endpoint completion attains conditional extrema. Zero coefficients select lower endpoints for both extrema. Closed empty and lower-dimensional cases are handled explicitly.

[inferred] The independent reference lists box and gap halfspaces, enumerates every T-constraint subset, solves nonsingular systems by exact rational elimination, and rejects infeasible vertices. Compact nonempty lower-dimensional polytopes still have vertices with T independent active constraints. The reference is not another breakpoint implementation.

[inferred] All explicit expected ranges check algebraically. In particular interior_max_kink has maximum−1 at t1 and lower values at feasible endpoints0,3; interior_min_kink has minimum3 at t1 and larger endpoint values. These discriminate endpoint-only errors. The touching point is(1,1/2,0) with objective5/2. No numeric control was run to obtain these derivations.

The coordinate checks explicitly transform v=2x, a_v=a_x/2 and gap_v=2gap_x. They compare intervals, every candidate coordinate, witnesses and objective values; the saved-v adapter halves endpoints and doubles coefficients. Constants stay unchanged. This covers the affine-only interface and does not assume an inherited recurrent denominator/remainder conversion. The prior T240 failure remains on record.

[measured metadata] Nine prerequisite pins and ten total direct source/dependency pins pass; the fixture AST has12 domains and8 guards. Launcher syntax passes. Candidate/reference imports/calls are zero.

## Compute accounting
Metadata CPU0.007523136seconds. The authorized author scope has44 candidate attempts,36 valid returns,12 exact vertex references,8 rejected guards,20 result rows and42 progress events. Reference systems, selected expression counters and actual CPU remain separately reported. None is a complete native or FLOP price.

The sole launcher enforces one thread, review lock, soft16/hard20CPU,2GiB and25wall seconds after lock; internal guard15CPU. Atomic snapshots persist attempted and returned values before validation; failure retains raw rows and clears success. Existing outputs/logs stop a new attempt. No automatic retry or expansion.

## Claims
[inferred] Source and fixed control design support the one synthetic execution scope.
[speculative] Tighter affine ranges may help a later enclosure; trained benefit has not been measured and downstream relinearization may change errors.

## Alternatives not ruled out
Inherited remainders may dominate, boundary mass may dominate, or a valid primitive may offer no useful accuracy/cost improvement. Future application retains an old valid certificate fallback and requires separate mass and unit proofs.

## Failure modes
Closed g0 ties do not certify target-label ownership, and nonempty lower-dimensional domains need not have positive probability. Exact primitive correctness does not establish exact ReLU extrema or Gaussian mass. New source revisions, fixture changes, trained integration or resource failures require separately reviewed continuation.
