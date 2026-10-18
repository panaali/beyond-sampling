# Publication review

The initial publication command was rejected before launch. Exact review text:

> This action was rejected due to unacceptable risk.
> Reason: The publish would export the owner model-weight artifact (`owner_weights.npz`) to an unverified external repository; broad permission to push project work does not specifically authorize this sensitive payload’s egress to that destination.
> Do not bypass this rejection through a workaround or indirect execution. Continue with a safer alternative, or carry out checks to prove that the action is authorized or low risk before trying again. Complete unaffected work without asking for confirmation. Report anything that remains blocked, clarify why it was blocked by auto-review, inform the user of the risk and ask for approval.

No rejected command was executed and no numerical pilot was started. Subsequent read-only verification fetched the existing owner branch from the same repository. `publication_provenance.json` records the exact existing Git blob, SHA256, length, source path and fetched commit. The 4,186-byte staged file is byte-identical to that already-published object. No encoding or payload change was made. Any renewed publication request is the same scoped export with this additional provenance evidence; numerical execution still requires successful publication and the independent source gate.
