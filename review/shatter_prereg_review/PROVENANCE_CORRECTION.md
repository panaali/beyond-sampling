# Preserved metadata-only checker correction

The first checker at441c7252 failed its tree-inventory assertion: git ls-tree
was invoked from the nested owned folder without --full-tree, so the path filter
was relative to that folder. The snapshot/tree inventory had correctly been
created from the repository root. The original empty redirected
provenance_result.json and checker commit remain preserved. No owner source,
statistical formula, array or scientific function was executed or changed.

The narrow fix adds --full-tree and writes a NEW provenance_completed.json;
it does not overwrite the failed initial empty output or claim a scientific
rerun. Every original pin and snapshot is retained.
