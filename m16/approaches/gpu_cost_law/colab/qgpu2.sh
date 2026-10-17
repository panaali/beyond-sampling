cd /content/cn/m16/approaches/gpu_cost_law
while pgrep -f "estP.py 8 5 0 exact" >/dev/null; do sleep 5; done
run() { local t0=$(date +%s); python estP.py $1 $2 $3 $4 11 cuda > /content/g_$1_$2_$3_$4.log 2>&1; echo "H$1 T$2 $3 $4 rc=$? wall=$(( $(date +%s)-t0 ))" >> /content/qgpu.done; }
run 8 4 0 recipe108
for seed in 1 2 3 4; do run 8 4 $seed recipe108; done
run 8 5 0 recipe108
for seed in 1 2 3 4; do run 8 5 $seed exact; run 8 5 $seed recipe108; done
