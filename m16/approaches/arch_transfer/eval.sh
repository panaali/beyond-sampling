#!/bin/bash
# Evaluate one candidate (pre-registered README section 3), two jobs at a time, each job committing its own output:
# truths (3 seeds), repair design (seed 0), M1 at 64k / 1e6 / 4e6 FPE x 3 seeds x 4 rules, repair verify, line diagnostics.
# usage: eval.sh TAG [SEEDS]   (waits for the seeds' weights to exist locally after a pull)
TAG=$1; SEEDS=${2:-"0 1 2"}; DIR=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$DIR/../../.." && pwd); BR=claude/t432-arch-transfer
mkdir -p "$REPO/data/arch_transfer"
sync() { (flock 9; cd "$REPO"; for i in 1 2 3 4 5; do git pull -q --rebase origin $BR && break; sleep $((2**i)); done) 9>"$REPO/data/arch_transfer/git.lock"; }
for s in $SEEDS; do
  if [ "$TAG" != rnn32 ] && [ "$TAG" != m16 ]; then
    until [ -f "$DIR/weights/${TAG}_s$s.pt" ]; do sleep 600; sync; done
  fi
done
J=$REPO/data/arch_transfer/jobs_$TAG.txt; : > $J
for s in $SEEDS; do echo "truth.py $TAG $s|truths/${TAG}_s$s.json" >> $J; done
echo "repair.py design $TAG 0|repair/${TAG}_s0_design.json" >> $J
for B in 64000 1000000 4000000; do for s in $SEEDS; do for r in 0 1 2 3; do
  echo "score.py $TAG $s $B $r|runs/m1_${TAG}_s${s}_b${B}_r${r}.json" >> $J; done; done
  [ $B = 64000 ] && echo "repair.py verify $TAG 0|repair/${TAG}_s0_verify.json" >> $J; done
for s in $SEEDS; do echo "lines.py $TAG $s|results/lines/${TAG}_s$s.json" >> $J; done
run1() {
  CMD=${1%|*}; OUTF=${1#*|}; DIR=$2; REPO=$3; BR=$4
  [ -f "$DIR/$OUTF" ] && { echo "skip $OUTF"; return 0; }
  if [[ $CMD == "repair.py verify"* ]]; then until [ -f "$DIR/${OUTF/_verify/_design}" ] || [ -f "$REPO/data/arch_transfer/FAILED_design_$(basename ${OUTF%_s0_verify.json})" ]; do sleep 60; done; [ -f "$DIR/${OUTF/_verify/_design}" ] || { echo "FAILED (no design) $CMD"; return 0; }; fi
  ( ulimit -v 2500000; cd "$DIR" && OMP_NUM_THREADS=1 python $CMD ) > "$REPO/data/arch_transfer/$(echo $CMD | tr ' /' '__').out" 2>&1 \
    || { echo "FAILED $CMD"; [[ $CMD == "repair.py design"* ]] && touch "$REPO/data/arch_transfer/FAILED_design_$(echo $CMD | cut -d" " -f3)"; return 0; }
  (flock 9; cd "$REPO"; git add "$DIR/$OUTF"; git commit -q -m "T432: $CMD

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
   for i in 1 2 3 4 5; do git pull -q --rebase origin $BR && git push -q origin $BR && break; sleep $((2**i)); done
  ) 9>"$REPO/data/arch_transfer/git.lock"
  echo "done $CMD"
}
export -f run1
tr '\n' '\0' < $J | xargs -0 -P 2 -I{} bash -c 'run1 "$@"' _ {} "$DIR" "$REPO" "$BR"
echo "eval $TAG finished"
