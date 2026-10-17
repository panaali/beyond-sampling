# T163 independent dqds retry review

## Question
Does DLASQ3's uncapped retry jump admit a finite Q5-attempt bound for the small SVD path, and does the surrounding DLASQ2 source have other unbounded branches? Owner T156 requested this independent source-only review; TASKS-only36994f24 reserved it first. No owner files are changed.

## Method
Read and hash T156's public DLASQ2/3/4/5/6 sources and installed-environment manifest. Prove the retry transition bound under explicit active-array/IEEE conditions, separate those conditions from whole-SVD reachability and installed-source equivalence. Inspect other backwards jumps before treating outer iteration caps as a total termination proof. No numerical SVD, geometry or Gaussian calculation.

Two alternative explanations: Q3's failure counter may implicitly bound retries (test threshold transitions and zero-shift exit); outer iteration caps may still leave a cleanup loop without progress (inspect every backwards jump and loop variant). Invalid internal state is not itself an executable valid-input counterexample.

## Status
Bounded source review complete; independent proof review and owner integration pending.

## Results
Under the stated zero-shift sign invariant, Q3 uses at most4 Q5 calls and at most1 Q6 fallback per invocation. This counts attempts, not only retries. A NaN does not add a fifth attempt because it forces zero shift earlier. See PROOF.md for initial TTYPE and signed-zero qualifications.

A distinct source-level non-progress state exists in Q2's INFO=2 cleanup at label145: I0=I1=2 and Z(3)<0 leaves I1 and the previous split marker unchanged forever. This does NOT establish reachability from a valid scaled SVD input or equivalence to the installed binary. It prevents deriving a universal full failure-path bound solely from the displayed outer caps. Successful observed calls and abstract worst-case control paths must remain distinct.

Reproduce: inspect pinned files against source_pins.json and follow PROOF.md line references. No numerical outputs or timing claimed.

## Compute accounting
Read-only source reasoning plus file hashing; zero model, SVD or mass evaluations. No benchmark budget modification. Installed dispatch/binary proof remains T156 owner scope.

## Claims
[inferred] Conditional four-attempt retry theorem and the explicit cleanup-state non-progress witness. Neither a universal installed-SVD termination proof nor a discovered reachable model failure is claimed. Independent review pending; Claude DRI classifies.

## Alternatives not ruled out
A stronger invariant may prove the cleanup state unreachable for supported inputs; installed code may differ; a runtime finite-call census can bound the actual frozen run without a universal source theorem. Numerical examples cannot by themselves prove a universal cap.

## Failure modes
Fortran NaN MIN semantics, signed zero, non-nearest rounding, trapping arithmetic, compiler fast-math, invalid active markers and source/binary mismatch. The zero-shift invariant must be justified for the invocation population before using the conditional bound in a complete cost claim.
