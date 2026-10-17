cd /content/cn/m16/approaches/gpu_cost_law
for T in 4 5; do for m in 0 1 2 3 4 0:fresh1000 0:fresh1001 0:fresh1002 0:fresh1003; do python mc_gpu.py 8 $T $m 10 $((7000+T)) >> /content/qmc.out 2>&1; done; done
echo MCDONE >> /content/qmc.out
