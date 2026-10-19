#!/bin/bash
# Block until the queue runner exits or ~115 min pass (fits one background tool call); then print a compact status.
# usage: wait_queue.sh <queue-id>
REPO=$(cd "$(dirname "$0")/../../.." && pwd); PID=$(cat "$REPO/data/scale_study/queue_$1.pid")
timeout 6900 tail --pid=$PID -f /dev/null
if kill -0 $PID 2>/dev/null; then echo "queue $1 RUNNING"; else echo "queue $1 FINISHED"; fi
grep -E "done|FAILED|screened|already" "$REPO/data/scale_study/logs/queue_$1.log" | tail -20
