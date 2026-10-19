#!/bin/bash
# Start arch.sh pipelines detached (survive the tool call). usage: launch.sh ID "TAG THREADS LOG2N" ["TAG THREADS LOG2N"]
# Several pipelines given to one launch run one after another (each pipeline keeps <= 2 runs at a time).
ID=$1; shift; DIR=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$DIR/../../.." && pwd); mkdir -p "$REPO/data/arch_transfer"
source "$REPO/.venv/bin/activate"
CMD=""; for a in "$@"; do CMD="$CMD bash $DIR/arch.sh $a;"; done
setsid nohup bash -c "$CMD" > "$REPO/data/arch_transfer/queue_$ID.log" 2>&1 < /dev/null &
echo $! > "$REPO/data/arch_transfer/queue_$ID.pid"; echo $!
