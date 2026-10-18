cd /content/cn/m16/approaches/gpu_cost_law
while pgrep -f "bash qgpu4.sh" >/dev/null; do sleep 10; done
run() { local t0=$(date +%s); python estP.py "$@" > /content/g_$1_$2_$3_$4.log 2>&1; echo "H$1 T$2 $3 $4 ${@:5} rc=$? wall=$(( $(date +%s)-t0 ))" >> /content/qgpu.done; }
for s in 0 1 2 3 4; do run 8 3 $s exact 11 cuda; run 8 3 $s recipe108 11 cuda; python mc_gpu.py 8 3 $s 10 7103 >> /content/qmc.out 2>&1; done
run 16 3 0 exact 11 cuda; run 16 3 0 recipe108 11 cuda; python mc_gpu.py 16 3 0 10 7163 >> /content/qmc.out 2>&1
run 16 4 0 exact 11 cuda --sample 0.1 --nomass
run 16 4 0 recipe108 11 cuda --sample 0.1 --nomass
echo Q5DONE >> /content/qgpu.done
