cd /content/cn/m16/approaches/gpu_cost_law
while pgrep -f "bash qgpu3.sh" >/dev/null; do sleep 10; done
run() { local t0=$(date +%s); python estP.py $1 $2 $3 $4 11 cuda > /content/g_$1_$2_$3_$4.log 2>&1; echo "H$1 T$2 $3 $4 rc=$? wall=$(( $(date +%s)-t0 ))" >> /content/qgpu.done; }
run 4 6 0 exact
python mc_gpu.py 4 6 0 10 7006 >> /content/qmc.out 2>&1
for m in 0 1 2 3 4; do python mc_gpu.py 8 6 $m 10 7006 >> /content/qmc.out 2>&1; done
run 8 6 0 exact
run 8 6 0 recipe108
echo Q4DONE >> /content/qgpu.done
