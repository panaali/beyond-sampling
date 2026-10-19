#!/bin/bash
# usage: run_jobs.sh jobs.txt log  (2 parallel processes, each under ulimit -v 2500000)
cd "$(dirname "$0")"; . ../../../.venv/bin/activate
xargs -P 2 -L 1 bash -c 'ulimit -v 2500000; python s1train.py "$@"' _ < "$1" >> "$2" 2>&1
