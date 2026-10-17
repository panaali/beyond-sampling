cd /content/cn/m16/approaches/gpu_cost_law
run() { s=$(date +%s); python estP.py $1 $2 $3 $4 11 cuda > /content/g_$1_$2_$3_$4.log 2>&1; echo "H$1 T$2 $3 $4 rc=$? wall=$(( $(date +%s)-s ))" >> /content/qgpu.done; }
for s in 0 1 2 3 4; do run 8 4 $s exact; run 8 4 $s recipe108; done
run 8 5 0 exact
run 8 5 0 recipe108
for s in 1 2 3 4; do run 8 5 $s exact; run 8 5 $s recipe108; done
