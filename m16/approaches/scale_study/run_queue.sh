#!/bin/bash
# Run a container's job list two at a time (AGENTS rule 8: <= 2 heavy processes per container).
# usage: run_queue.sh <queue-id>   (jobs in queues/<queue-id>.txt, one "H T SEED" per line, in order)
DIR=$(cd "$(dirname "$0")" && pwd)
grep -v '^#' "$DIR/queues/$1.txt" | xargs -P 2 -L 1 bash "$DIR/job.sh"
