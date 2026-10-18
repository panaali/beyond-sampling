# T236 independent proof and interpretation review

## Question
Do the frozen scaling, depth-two count and arrangement arguments support the interpretation proposed for T232, and does the source inventory describe only what is actually available?

## Method and status
PASS for f20b54be, shared3228e8aa, under delegation84f348a5. This is written algebra plus four file-byte/hash comparisons, original README blob comparison and a pinned Git tree listing. source_review.json records the bindings. No author program, array, model, geometry, numerical library or random generator was invoked. This is not execution clearance or a review of random-control results.

## Independent reasoning
For positive finite a,b, substitute z_t=x_t/b^(t-1). The induction step factors a*b^(t-1) from both recurrent and input terms before ReLU. Thus every gate and its zero boundary correspond under an invertible linear input map. Open nonempty sets and dimensions are preserved. The cardinality identity holds for each draw even when a,b depend on its norms. It does not identify the Gaussian measures or true rank labels under a nonorthogonal time rescaling.

At depth two, fixing the sign and magnitude scale of x1 reduces the second layer to thresholds in x2/|x1|. For a nonzero first-state vector, each threshold is an independent continuous row projection divided by a fixed nonzero input weight. Every threshold flips its own coordinate once, so all H+1 intervals have different gate words. If all input weights share a sign, the opposite first half is identically OFF and contributes only two intervals. Mixed-sign and same-sign halves therefore give the claimed 2H+2 and H+3. The Gaussian sign probabilities follow by counting the two constant-sign patterns. These arguments assume the independent continuous row laws conditional on w; the intended independent Gaussian construction satisfies this. Independent Gaussian row directions normalized to positive target lengths retain atom-free projections for H>=2. Zero target lengths or a different dependent generator need separate treatment.

For the stronger upper bound, fix a full-dimensional parent history. One nonzero input coefficient, selected at every time, supplies triangular signed preactivation normals with nonzero diagonal. Their span is the whole past-input space. All inward forms are nonnegative on the closed cone, so their sum is strictly positive off the origin. The section of the extruded cone where this sum equals one is convex and has dimension t. Any open child history has a point with nonzero past and therefore meets that section. H affine hyperplanes give at most the standard affine arrangement count; convex restriction cannot split a cell. Applying this to each parent proves the product bound. It supplies neither a future growth factor nor an algorithmic lower bound, and does not establish floating enumeration completeness. The w=0 case is separately trivial.

A consistent hidden permutation or positive diagonal state reparameterization preserves the represented function and gate regions at the same input. Independently permuting entries within each recurrent row is not such a relabeling. Replacing the readout can affect accuracy with unchanged prefix gates. All-OFF output ties also prevent an unconditional exchangeable-readout chance assertion without additional conditioning.

## Source and protocol findings
The manifest correctly binds the 3001-byte README at author revision1d6f4db9, and that folder contains only this file at that revision. This is a local historical inventory, not global absence of an implementation. The five global-normalized and five row-normalized random draws define different ensembles. The combined median is a stated descriptive rule, not a confidence guarantee; per-ensemble reporting and explicit mass-coverage aggregation remain useful. Optional k5 inclusion depends on runtime and is not a random subset. Diagnostic accuracy needs hits, trials, streams and ties, while numerical mass coverage needs separate mass/completeness qualifications. These are interpretation prerequisites, not changes to Claude's experiment or its budget.

## Results and claims
[inferred, exact-real proof] Positive global input/recurrent scaling preserves complete prefix-history cardinality via time-diagonal input mapping.

[inferred, conditional distribution theorem] At H16 the usual mixed-sign Gaussian depth-two count is34; the exceptional same-sign count is19. The target count34 alone does not demonstrate training-created complexity reduction.

[inferred, upper bound] The pointed-parent affine-section product is valid and tighter than the full-space per-parent bound; no later-step equality or lower growth rate follows.

[measured, metadata only] All four frozen source files and original README blob/tree listing match their recorded pins. There are no experimental measurements in this review.

## Alternatives not ruled out and failure modes
Training can affect deeper counts, mass concentration or output behavior. Aggregation need not enumerate every gate history. Global norm changes can affect mass without count changes; rowwise matching is different. Degenerate weights, biased recurrences, rank-conditioned inputs, finite arithmetic, missing histories, mass error, dependent generators and selected cheap-depth subsets all require their own analysis. A finite descriptive comparison cannot establish a universal training-effect or scaling theorem.

## Compute accounting
Document/Git metadata comparison and written proof only. Zero author, model, RNG, numerical-library, geometry, mass, array-decoding or training calls. No source-runtime or FLOP claim. Reproducible provenance is the pinned source bytes and original Git blob, not a new numerical command.
