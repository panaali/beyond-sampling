#!/bin/bash
# Recover every job in this container that trained to the end but failed to save; commit + push each. Run from repo root.
DIR=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$DIR/../../.." && pwd); BR=claude/t421-scale-study
source "$REPO/.venv/bin/activate"
for d in "$REPO"/data/scale_study/runs/*_*_*; do
  IFS=_ read H T S <<< "$(basename "$d")"; TAG=m_${H}_${T}_cpu${S}
  [ -f "$d/2nd_argmax_${H}_${T}_1073741824_${S}.pth" ] || continue
  (flock 9; cd "$REPO"; git pull -q --rebase origin $BR) 9>"$REPO/data/scale_study/git.lock"
  [ -f "$DIR/results/models/$TAG.json" ] && continue
  (cd "$DIR" && python recover.py $H $T $S) || { echo "$TAG recover FAILED"; continue; }
  (flock 9; cd "$REPO"
   git add "$DIR/weights/$TAG.pth" "$DIR/weights/$TAG.npz" "$DIR/results/models/$TAG.json"
   git commit -q -m "T421: $TAG recovered from its final checkpoint

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
   for i in 1 2 3 4 5; do git pull -q --rebase origin $BR && git push -q -u origin $BR && break; sleep $((2**i)); done
  ) 9>"$REPO/data/scale_study/git.lock"
  echo "$TAG recovered"
done
