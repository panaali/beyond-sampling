# Preserved reviewer-only progress failure

The sole gated launch after immutable211132d1/shared6b2f6ee4 stopped with
`TypeError: main.<locals>.event() got multiple values for argument 'name'`.
The event helper's positional parameter collided with artifact `name` metadata.
Original checker, reader, protocol, gate, completed-input hashes, output/results,
three progress records and run1.log are immutable and retained.

Before failure,32 saved controls and68 arrays in23 archives passed their checks.
No artifact frame/payload/AST file was decoded. The producer results JSON was read
inside the gated attempt; its row values are now exposed to the checker, but there
is no change to predeclared cases, arithmetic, source bindings or tolerances.
Recorded reviewer CPU1.0331012650000002s, RSS157284KiB; external log1.22wall,
0.89user/0.32system. No author/predictor/model/fit/RNG calls occurred.

A distinct source-gated recovery is proposed: rename only the helper event
parameter, fresh checker/protocol/gate/output/log paths and preserve/sum the prior
CPU. Repeated reviewer prerequisites, controls and array binding are charged as
recovery work; no completed producer or scientific computation is repeated. No
recovery execution is authorized by this note alone.
