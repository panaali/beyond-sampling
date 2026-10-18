# T173 independent source and metadata audit

## Question
Does the published sensitivity inventory faithfully describe its historical
sources and saved evidence, without treating defective-wrapper summaries as
validated numerical performance?

## Method
Claim5c6d2cb4 delegates this complementary review after T55 yields it. Read ASTs
and saved JSON only. check_saved.py independently extracts class defaults and
variant keyword dictionaries, verifies summary coverage/fields and file hashes,
and binds the two saved diagnostic rows. No author imports or numerical calls.

Two alternatives are explicitly separated: more individual evidence may exist
outside the pinned summaries (only local absence is checked), and the equality
repair may behave differently on other variants (no numerical conclusion follows
from the repaired frozen controls). The cheapest tests here are saved field
coverage and source branch inventory; a new variant experiment needs DRI priority
and a separately frozen protocol.

## Status
Complete: one saved metadata run passed in0.008005001CPU after independent
source gatec236f110 was published6e49bc7c. Checker e708070e, results.json and
immutable run1.log preserve the source and outcome. No failure/retry occurred.
This is an open review, not permission to restart the frozen cost chain.

## Results
All12 historical dependency hashes,16 variants,320 aggregate groups,
noise/source contracts and both diagnostic rows pass. Source inspection supports the stated
16 variants, five seeds, four widths and clean plus16 noisy copies design.
The 5,440 count covers recipe evaluations only: the original complete design
also calls340 exact truths and four sampler-counter setups. No such calls are
made in this review. 340 model slots by source design does not prove every
numerical model is distinct or recover missing historical byte identities.

## Compute accounting
Text/AST/JSON/hash inspection only; one <=20CPU/2GB one-thread review-lock run.
Zero estimator, model, noise, truth, mass, training or native-cost calls.

## Claims
[inferred, source-only] Aggregate summaries cannot reconstruct missing paired
per-copy truths, estimates and costs. Existing frozen diagnostic noise seed89
is distinct from the sensitivity design seed86. No winning/failed variant or
historical panel identity is certified here.

## Alternatives not ruled out
Individual artifacts elsewhere; variant-specific semantic differences;
checkpoint/library-default changes; plain-float/reference tie defects; better
repair implementations. None is settled by summary coverage.

## Failure modes
Importing author code during metadata review; substituting noise panels;
selecting variants from invalid summaries; treating source call counts as full
native cost; granting numerical replay permission through a metadata pass.

Reproduction: `python review/t173_inventory_review/check_saved.py` in a fresh
exclusive output folder under the declared review lock/caps. Existing results
refuse overwrite. See results.json for complete input/source hashes.
