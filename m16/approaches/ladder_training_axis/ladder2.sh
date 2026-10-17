#!/bin/bash
# T101 driver v2: waits for the running k=30 job (PID given as $1), then k = 28, 27, ... with n_fib = 128;
# stops after the first rung where a recipe fails.
cd "$(dirname "$0")"; source ../../../.venv/bin/activate; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -v 2500000
while kill -0 $1 2>/dev/null; do sleep 20; done
python verdict.py 30 >> ladder.log 2>&1
for k in 28 27 26 25 24 23 22 21 20; do
  python run_rung.py $k 128 >> rung_k$k.log 2>&1
  python verdict.py $k >> ladder.log 2>&1 || { echo "STOP at k=$k" >> ladder.log; break; }
done
echo finished >> ladder.log
