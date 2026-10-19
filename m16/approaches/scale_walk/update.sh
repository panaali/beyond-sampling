#!/bin/sh
# Pick up new T421 grid points: refresh registry, gate, truths (2 workers), blind estimates, then score.
set -e
cd "$(dirname "$0")"; . /home/user/venv423/bin/activate; ulimit -v 2500000
./refresh.sh
python gate.py
python truths.py 0 2 & python truths.py 1 2; wait
python run.py
python analyse.py > analysis.log; cat analysis.log
