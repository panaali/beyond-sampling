#!/bin/bash
# T281: HP3 confirmation run (DRI 2026-10-18). T317c exactly as frozen (@1c742430), no changes; control first.
# Prereq: data/hp3/{weights.npz,manifest.json,truths.json} with hashes entered in score.PANELS["hp3"].
set -e
cd "$(dirname "$0")"
export PANEL=hp3 PYTHONPATH=shim
ulimit -v 4000000
S=../../../data/subs/t317c_1c742430/m16/approaches/r1_cont/estimator_c.py
python3 -W ignore score.py control control est sampling 1000,4000,16000,64000,256000 | grep -A1 "B=1000\|B=16000"
python3 -W ignore score.py t317c_cont $S est deductive 0 > /dev/null
CI_CODE=216 python3 -W ignore cost_t317.py t317c_cont $S          # official cost (correct_interval 229)
python3 -W ignore sens_t317c.py                                    # every executed ndtr charged
