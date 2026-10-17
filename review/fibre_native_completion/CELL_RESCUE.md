# Frozen separate-process cell continuation

Board41c0b794 authorizes completed component6 as the control, then failed component12 only. The original timeout and its114.875CPU overrun remain unchanged. All new processes enforce OS soft105/hard110; there is no live-process cap reset.

1. Enumerate original cells_on_face for each of the two original polys in source order using the unchanged tracer. Save every cell's float64 vertices exactly, charge this geometry once, and pin its artifact.
2. In a fresh process for each cell, import the unchanged reviewed source/tracer. Select that cell's original poly and return the saved cell from the cells_on_face call, whose cost was counted in step1. Replace only the original face_err assignment tot=0.0 by tot=_resume_total. The resume value is the exact binary output from the preceding cell (zero for first); original inner += operations execute in the same order. Each cell has its own cache in the original function already.
3. Recompute the same clamp forms at each call. Their repeated numerical work is charged, so the final result is the actual cell-adapter cost and a conservative source-arithmetic upper bound, not an assumed equal cost. Legacy parity removes only the exactly known repeated setup charge (Ncells−1)*16*len(clamps), while actual executed legacy total is also retained.
4. On completed component6, require original estimate hex, stats and corrected legacy parity. Verify every native site difference against the repeated clamp preparation, with no unexplained differences. Independent review gates component12; no source tuning or later-model expansion.

Saved cells are computed from this model and counted; they are not free advice. Process imports, JSON/state transport and memory/control remain outside the named numerical convention. Any individual cell cap or unexplained mismatch is preserved and halts this finite rescue. Independent-subtotal summation is never used.

Control6 passed all31 cells, hex/legacy/stats and exact repeated-setup site differences, but has no error pieces. Board d4bfd2af adds fixed nonzero component2 (first completed nonzero component under10CPU) before12. cell_rescue_nonzero.py reverses exactly to the original source after its three index/path-only substitutions; retain both controls and require actual nonzero carried totals.

Both controls pass. Component6:31 cells, zero output,383 traces/0pieces,5.962562139CPU. Component2:30 cells, final0x1.4125ee2a14d06p-17 exactly,789 traces/536pieces,10.655092788CPU. In both, every native site delta equals only repeated _fn preparation; corrected legacy matches while executed extra480/464 is retained. Failed12 remains unexecuted under this adapter until independent saved-control review.
