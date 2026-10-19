#!/bin/bash
# One training run under the memory cap, then commit + push its own files (weights/<name>.{pt,json,log}).
# usage: job.sh TAG SEED LR_BASE LOG2N THREADS
TAG=$1; S=$2; LR=$3; LG=$4; TH=$5
DIR=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$DIR/../../.." && pwd); BR=claude/t432-arch-transfer
NAME=${TAG}_s${S}; [ "$LG" != 30 ] && NAME=${NAME}_lr${LR}_n${LG}
[ "$S" -lt 100 ] && [ "$LG" != 30 ] && NAME=${TAG}_s${S}   # panel runs keep the short name at any length
mkdir -p "$REPO/data/arch_transfer"
[ -f "$DIR/weights/$NAME.json" ] && { echo "$NAME already done"; exit 0; }
( ulimit -v 2500000; cd "$DIR" && OMP_NUM_THREADS=$TH python train.py $TAG $S $LR $LG weights $NAME ) \
  > "$REPO/data/arch_transfer/$NAME.out" 2>&1 || { echo "$NAME FAILED"; exit 1; }
(flock 9; cd "$REPO"
 git add "$DIR/weights/$NAME".*
 git commit -q -m "T432: $NAME trained (val $(python -c "import json;print(round(json.load(open('$DIR/weights/$NAME.json'))['val_acc_2^20'],5))"))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
 for i in 1 2 3 4 5; do git pull -q --rebase origin $BR && git push -q -u origin $BR && break; sleep $((2**i)); done
) 9>"$REPO/data/arch_transfer/git.lock"
echo "$NAME done"
