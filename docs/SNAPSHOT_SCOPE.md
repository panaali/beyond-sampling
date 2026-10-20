# Historical review snapshots and current extracts

Six historical mixed-scope documents now contain ARC-only editorial extracts. [SNAPSHOT_SCOPE.json](SNAPSHOT_SCOPE.json) records both original Git blobs and current extract hashes. Existing review manifests, saved results and their hashes remain historical evidence; they have not been relabelled as checks of edited bytes.

To repeat the original reviews, restore their original snapshots from the commit and path in the scope manifest into a disposable historical worktree. The pinned original review commits and source hashes in each review's manifest remain authoritative. Some original scripts (including `review/t201_metadata_review/check_saved.py`) hardcode their original workspace and require fresh output paths; they are historical replay scripts, not current-tree verification commands. Do not overwrite saved review results.

Current extraction integrity can be checked with `python tools/check_snapshot_scope.py`; it checks only editorial file hashes and performs no scientific rerun. ARC procedures, numerical results and original historical evidence remain available.
