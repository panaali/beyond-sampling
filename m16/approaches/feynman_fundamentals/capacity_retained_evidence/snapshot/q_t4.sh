# T234 queue on a T4 (2 vCPU): two runs at a time; archive results + weights + checkpoints after each run
C=/content/cn/m16/approaches/capacity_ceiling; cd $C; mkdir -p weights results
arch() { date > results/heartbeat; tar czf /content/t234.tmp.tgz -C /content/cn m16/approaches/capacity_ceiling/results m16/approaches/capacity_ceiling/weights data/capacity_ceiling 2>/dev/null && mv /content/t234.tmp.tgz /content/t234.tgz; }
r() { local t0=$(date +%s); python run.py "$@" > /content/t234_$1.log 2>&1; echo "$1 rc=$? wall=$(( $(date +%s)-t0 ))" >> /content/t234.done; arch; }
( python repro_zoo.py 0 > /content/t234_zoo0.log 2>&1; echo "zoo0 rc=$?" >> /content/t234.done; arch ) &
r a30 --log2n 30 --seeds 0 1 2 3 4
wait
r a31 --log2n 31 --seeds 0 1 2 3 4 &
r a32 --log2n 32 --seeds 0 1 2 3 4
wait
echo T4DONE >> /content/t234.done; arch
