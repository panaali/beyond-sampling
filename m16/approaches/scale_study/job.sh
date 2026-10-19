#!/bin/bash
# One grid job (T421): skip if done or screened out, train under the memory cap, then commit + push its own files.
# usage: job.sh H T SEED     (run from any cwd; needs .venv per AGENTS.md)
H=$1; T=$2; S=$3; TAG=m_${H}_${T}_cpu${S}
DIR=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$DIR/../../.." && pwd); BR=claude/t421-scale-study
LOG=$REPO/data/scale_study/logs; mkdir -p "$LOG"
gitsync() { (flock 9; cd "$REPO"; for i in 1 2 3 4 5; do git pull -q --rebase origin $BR && break; sleep $((2**i)); done) 9>"$REPO/data/scale_study/git.lock"; }
gitsync
[ -f "$DIR/results/models/$TAG.json" ] && { echo "$TAG already done"; exit 0; }
python "$DIR/screen.py" $H $T $S || { echo "$TAG screened out"; exit 0; }
( ulimit -v 2500000; cd "$DIR" && python train.py $H $T $S ) > "$LOG/$TAG.log" 2>&1 || { echo "$TAG FAILED (see $LOG/$TAG.log)"; exit 1; }
(flock 9; cd "$REPO"
 git add "$DIR/weights/$TAG.pth" "$DIR/weights/$TAG.npz" "$DIR/results/models/$TAG.json"
 git commit -q -m "T421: $TAG trained ($(python -c "import json;print(round(json.load(open('$DIR/results/models/$TAG.json'))['acc'],5))"))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
 for i in 1 2 3 4 5; do git pull -q --rebase origin $BR && git push -q -u origin $BR && break; sleep $((2**i)); done
) 9>"$REPO/data/scale_study/git.lock"
echo "$TAG done"
