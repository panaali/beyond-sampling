#!/bin/bash
# Start a queue detached from the shell (survives the tool call); log in data/scale_study/logs/queue_<id>.log.
# usage: launch.sh <queue-id>  -> prints the PID of the queue runner
DIR=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$DIR/../../.." && pwd); mkdir -p "$REPO/data/scale_study/logs"
PF="$REPO/data/scale_study/queue_$1.pid"
[ -f "$PF" ] && kill -0 "$(cat "$PF")" 2>/dev/null && { echo "queue $1 already running (PID $(cat "$PF")); not starting a second runner"; exit 1; }
source "$REPO/.venv/bin/activate"
setsid nohup bash "$DIR/run_queue.sh" "$1" > "$REPO/data/scale_study/logs/queue_$1.log" 2>&1 < /dev/null &
echo $! > "$REPO/data/scale_study/queue_$1.pid"; echo $!
