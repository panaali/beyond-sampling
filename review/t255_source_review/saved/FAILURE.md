# Preserved reviewer provenance failure

The sole first saved-review attempt stopped before block/result arithmetic at a historical written-gate README hash. Recorded review CPU was 0.00399439900000001 seconds; no block record or scientific array was decoded. `check.py`, `results.json`, `run1.log`, `scope.json` and `gate.json` are retained unchanged.

Exactly one historical reviewer pin differs from the present filesystem: the author README was updated after the original source gate. The original SOURCE_REVIEW, manifest and all twenty frozen source/JSON payloads remain unchanged. Exact Git bytes from b8d5cb1b are saved as `historical/README_at_source_review.md` and match the original pinned SHA. This is a documentation-version binding mismatch, not an arithmetic failure.

Any recovery must use new source/gate/output names, pin the failed attempt and exact historical README, and resolve only this historical pin to those original bytes. All numerical formulas, coverage and tolerances stay unchanged. No author replay or automatic retry is authorized by this note.
