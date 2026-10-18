# T261 fixed serial batch-tile rescue

Original whole-batch B1024 compiler metadata requests332800 shared bytes, beyond SM80. Preserve original source/results. New tiled_step.py fixes internal BB32/warps8 for every B and accumulates full-batch gradients/loss across serial tiles before the single AdamW update. This changes reduction order; no prior parity transfers.

Run the identical six configurations/eight reference+kernel calls and unchanged thresholds via tiled_validation.py once, with the same explicit CPU libdevice shim and local-update scope. Separate outputs/gates; all attempted work retained. Runtime caps105/110CPU2GiB120wall primarylock unchanged. Both56kernel and910harness completed published gates required.

Only after this new suite passes, compile tiled source exactly twice SM80 B32/B1024, BB32/warps8, unchanged IEEE/FMAfalse under same caps and workspace cache. No device calls, tuning or GPU speed/equivalence claim. Require independent completed source gate for tiled_compile.py/scope/source and published immutable gate binding validation. Retain any resource failure. Interpreter and compile metadata do not establish hardware occupancy or speed.
