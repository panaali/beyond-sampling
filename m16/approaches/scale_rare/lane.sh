#!/bin/bash
# T424 helper lane: for TAGS: K3 pass (push each), graded pass (push each), then truths (push each).
# Env: T424_R1C, T340_SRC, T421_DIR, T424_TRUTH_DIR (outside the repo), optional T424_CH; PY = python of the venv.
# Push only results/; commits end with the attribution line. usage: lane.sh LANE_NAME TAG [TAG ...]
set -u; LANE=$1; shift; cd "$(dirname "$0")"; PY=${PY:-$HOME/venv/bin/python}
push() { git add results/est results/truth 2>/dev/null; git commit -qm "T424 lane $LANE: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || return 0
  for i in 1 2 3 4 5; do git pull -q --rebase origin claude/t424-scale-rare && git push -q origin HEAD:claude/t424-scale-rare && return 0; sleep $((2**i)); done; echo "PUSH FAILED"; }
for pass in k3 graded; do for tag in "$@"; do
  T424_ONLY=$pass $PY est_run.py $tag && push "$tag $pass estimate (sealed)" || echo "FAILED $tag $pass"
done; done
for tag in "$@"; do
  for t0 in $($PY -c "import models as M, thresholds as T; r=M.by_tag('$tag'); print(*T.choose(r['T'], r['acc']))"); do
    $PY truth.py $tag $t0 && cp "$T424_TRUTH_DIR/$tag@$t0.json" results/truth/ && push "$tag@$t0 truth" || echo "FAILED truth $tag $t0"
  done
done
echo "LANE $LANE DONE"
