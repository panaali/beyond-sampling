# Narrow reviewer logging recovery

Original source gate5020286e missed the `event(name, **kw)` collision with
`name=artifact_name` at the first artifact start. This is a reviewer/source-review
coverage failure, including h4's gate, not a codec experiment discrepancy.
The original source gate remains unchanged as part of the record.

56 preserved attempt1 at5f596edb, complete=false/pass_all=false,32 control checks,
68 array bindings,three progress events and zero artifact frame decodes.
CPU1.0331012650000002 is retained; it was not a zero-work failure.

Frozen45b5079e recovery changes only the positional event argument to `kind`,
fresh file/gate/log names and explicit previous work/CPU binding. Reader is byte
identical; six top-level numerical/control helpers have identical ASTs. The main
diff and launcher substitutions were read directly. No producer source or data,
fixtures, tolerance, bit grammar or cap changes. Full reviewer control/array work
will be repeated and charged; no producer computation is repeated.

recovery_gate.json permits one separately published/bound105/110CPU2GiB120wall
recovery, with all prior failures preserved. No numerical parser or array decode
was executed in this gate review. Its numerical outcome remains pending.
