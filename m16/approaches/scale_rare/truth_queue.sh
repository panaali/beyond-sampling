#!/bin/bash
# usage: truth_queue.sh TAG [TAG...] : truths at each model's two pre-registered t0 (logs to $TRUTH_LOG)
for tag in "$@"; do
  for t0 in $(/home/user/venv/bin/python -c "import models as M, thresholds as T; r=M.by_tag('$tag'); print(*T.choose(r['T'], r['acc']))"); do
    /home/user/venv/bin/python truth.py $tag $t0 && echo "$tag $t0 done"
  done
done
