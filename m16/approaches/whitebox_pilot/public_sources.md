# Public Phase 2 source review, 21 October 2026

Independent read-only review by Codex child `whitebox_public_review`, requested
by the owner while T473 ran. No external messages or submissions. Author-reported
scores and resource figures are registered in the [T473 ledger entry](../../../INDEX.md);
none was numerically reproduced in this review.

## Best starting points

- [504aldo V29, pinned](https://github.com/504aldo/whest-p2-cumulant-k3/tree/a038d17924cd2df82dbf3a0d42da38fee48aa5a5):
  readable factorized third-cumulant propagation, source-memory compression,
  approximate fourth cumulants and extensive negative experiments. README flags
  outdated ARC term tables (generated at `6e80f7f`; recommends `93d091a`) and an
  orientation issue relevant to asymmetric fourth-cumulant replacements. Preserve
  the submitted version for reproduction; correct tables in a separate variant.
- [e30_a, pinned](https://github.com/siming767-creator/whestbench-lordprocciuto/tree/3d038e9f48a97671878ee7bbef02e193f128b61c):
  an additional public baseline missing from the initial list. A V29 descendant
  with pruning, fixed-basis compression and a learned mean correction. MIT code
  and submitted archive; estimator SHA256
  `6ddd97a96c30397683815c530bfb422f0a49ffdea49e4f7827149ed0a14729b0`
  matches the README and another participant's report.
- [logxio, pinned](https://github.com/logxio/fourth-cumulant/tree/90b6d5fb3e49d7b49c7e0b070afcefe45e3785bf):
  distinguish the submitted pruned K3/corrector from the much larger full-K4
  research implementation. Estimator SHA256
  `253796045ba6bdf2bdb0e33bc4b67721f8afb081d75e106b439bd89aebcea669`
  matches its manifest. “Best public write-up” is not an overall leaderboard rank.

Submitted numerical paths inspected use flopscope; research tools also use
NumPy/Torch. Static inspection is not a complete compliance audit. In particular,
setup behavior, residual time and peak memory still need native validation. The
reviewed full-width implementations exceed our local memory cap. Do not run
them unmodified under the T473 resource authorization.

## Prior evidence on the pilot's family

- [Submission 335413 report, E46, p.32](https://discourse.aicrowd.com/uploads/short-url/9NvFSYXNhGpw4QTKxs6dsd6AR1F.pdf):
  an exact first-layer mean control variate transported through **hard** mean-path
  gates amplified variance on its tested Phase 2 networks. T473 instead uses soft
  Gaussian gains; that difference merits a bounded test, not a novelty claim.
- [V29 findings F8/F13/F62](https://github.com/504aldo/whest-p2-cumulant-k3/blob/a038d17924cd2df82dbf3a0d42da38fee48aa5a5/docs/findings_log.md):
  Phase 1 soft-affine and fitted first-layer controls; a Phase 2 regression
  control reduced MC variance modestly but remained far behind cumulant methods.
  T473 evaluates its linear correction after aggregation, avoiding an extra
  per-sample surrogate pass. These earlier finite tests do not prove all
  sampling/structural hybrids impossible.

## Recommendation

Reproduce the public cumulant lineage before more undirected control-variate
sweeps. Start at small width, profile live buffers, then stream or chunk the
source-memory contractions while checking predictions and metered FLOPs against
the frozen original. Any accuracy-changing compression is a new variant, not a
memory-only reproduction. Follow-up is tracked on the task board.

Current submission eligibility remains unverified. Forum date statements and
later leaderboard entries conflict; individual submission pages were inaccessible
or empty. All external-source exposure is logged in [lit/web_log.md](../../../lit/web_log.md).
