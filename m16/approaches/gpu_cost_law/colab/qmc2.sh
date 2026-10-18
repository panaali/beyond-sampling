# T193 recheck: independent MC stream per model (seed 9000 + 100 T + model index), results in results/mc2/
cd /content/cn/m16/approaches/gpu_cost_law
while pgrep -f "bash qmc.sh" >/dev/null; do sleep 10; done
i=0
for T in 4 5; do for m in 0 1 2 3 4 0:fresh1000 0:fresh1001 0:fresh1002 0:fresh1003; do
  MCDIR=mc2 python mc_gpu.py 8 $T $m 10 $((9000+100*T+i)) >> /content/qmc2.out 2>&1; i=$((i+1)); done; done
echo MC2DONE >> /content/qmc2.out
