cd /content/cn/m16/approaches/gpu_cost_law
while pgrep -f "bash qgpu3.sh" >/dev/null; do sleep 10; done
run() { local t0=$(date +%s); python estP.py "$@" > /content/g_$1_$2_$3_$4.log 2>&1; echo "H$1 T$2 $3 $4 ${@:5} rc=$? wall=$(( $(date +%s)-t0 ))" >> /content/qgpu.done; }
run 16 5 0 exact 11 cuda
run 16 5 0 recipe108 11 cuda
python mc_gpu.py 16 5 0 10 7165 >> /content/qmc.out 2>&1
run 4 6 0 exact 11 cuda
python mc_gpu.py 4 6 0 10 7046 >> /content/qmc.out 2>&1
for m in 0 1 2 3 4; do python mc_gpu.py 8 6 $m 10 $((8600+m)) >> /content/qmc.out 2>&1; done
run 8 6 0 exact 11 cuda
run 8 6 0 recipe108 11 cuda
echo Q4DONE >> /content/qgpu.done
