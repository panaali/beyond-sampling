#!/bin/bash
# T427 helper (child sessions): set up, then for each TAG run run_est.py then mc.py in its own process (<= 2 TAGs = 2 heavy
# processes, each under ulimit -v 2500000), then commit only results/{est,mc}_<TAG>.{json,log} and push.
# usage: bash m16/approaches/scale_rating/helper.sh TAG1 [TAG2]
set -e
REPO=$(cd "$(dirname "$0")/../../.." && pwd); cd "$REPO"
git fetch -q origin claude/t421-scale-study claude/t427-scale-rating
mkdir -p m16/approaches/scale_study && git show origin/claude/t421-scale-study:m16/approaches/scale_study/setup.sh > m16/approaches/scale_study/setup.sh
bash m16/approaches/scale_study/setup.sh; rm -rf m16/approaches/scale_study
source .venv/bin/activate; pip install -q scipy
cd m16/approaches/scale_rating; export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for TAG in "$@"; do
  (ulimit -v 2500000; python run_est.py $TAG > results/est_$TAG.log 2>&1 && python mc.py $TAG > results/mc_$TAG.log 2>&1) &
done
wait
for TAG in "$@"; do git add results/est_$TAG.json results/mc_$TAG.json results/est_$TAG.log results/mc_$TAG.log; done
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -qm "T427 helper: results for $*

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3 4 5; do git pull -q --rebase origin claude/t427-scale-rating && git push -q origin HEAD:claude/t427-scale-rating && break; sleep $((2**i)); done
echo HELPER_DONE
