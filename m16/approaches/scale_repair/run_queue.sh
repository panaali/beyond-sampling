#!/bin/bash
# usage: run_queue.sh tag1 tag2 ...   design -> commit/push frozen edit -> verify -> commit/push (one heavy process per queue)
cd "$(dirname "$0")"; R=../../..; PY=$R/.venv/bin/python; L=$R/data/t425/logs; mkdir -p $L
push() { ( flock 9; cd $R; git add m16/approaches/scale_repair/results/$1; git commit -qm "T425: $1 $2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DR3S5STcPKr54qKn4pfgJz"
  for i in 1 2 3 4; do git pull -q --rebase origin claude/t425-scale-repair && git push -q origin HEAD:claude/t425-scale-repair && break; sleep $((2**i)); done ) 9>$R/data/t425/git.lock; }
for tag in "$@"; do
  [ -f results/$tag/edit.json ] || (ulimit -v 2500000; OMP_NUM_THREADS=1 $PY design.py $tag > $L/$tag.design.log 2>&1)
  [ -f results/$tag/edit.json ] || { push $tag "design (no edit)"; continue; }
  push $tag "design and frozen edit (before verification)"
  [ -f results/$tag/verify.json ] || (ulimit -v 2500000; OMP_NUM_THREADS=1 $PY verify.py $tag > $L/$tag.verify.log 2>&1)
  push $tag "verification"
done
