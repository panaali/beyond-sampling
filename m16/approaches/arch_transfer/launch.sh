#!/bin/bash
# Start arch.sh pipelines ("TAG TH LG"), panel-only trainings ("jobs:TAG:LR:LG:TH") or "eval:TAG[:s,s]" evaluations detached (survive the tool call).
# Several pipelines given to one launch run one after another (each pipeline keeps <= 2 runs at a time).
ID=$1; shift; DIR=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$DIR/../../.." && pwd); mkdir -p "$REPO/data/arch_transfer"
source "$REPO/.venv/bin/activate"
CMD=""; for a in "$@"; do case $a in eval:*) e=${a#eval:}; CMD="$CMD bash $DIR/eval.sh ${e%%:*} \"$( [[ $e == *:* ]] && echo ${e#*:} | tr , " " || echo 0 1 2)\";";; jobs:*) IFS=: read -r _ t lr lg th <<< "$a"; CMD="$CMD printf '$t 0 $lr $lg $th\\n$t 1 $lr $lg $th\\n$t 2 $lr $lg $th\\n' | xargs -P 2 -L 1 bash $DIR/job.sh;";;
  *) CMD="$CMD bash $DIR/arch.sh $a;";; esac; done
setsid nohup bash -c "$CMD" > "$REPO/data/arch_transfer/queue_$ID.log" 2>&1 < /dev/null &
echo $! > "$REPO/data/arch_transfer/queue_$ID.pid"; echo $!
