cd /content/cn/m16/approaches/gpu_cost_law
run() { grep -qxF "H$1 T$2 $3 $4 ${*:5} rc=0" <(sed 's/ wall=.*//' /content/qgpu.done) && return
        local t0=$(date +%s); python estP.py "$@" > /content/g_$1_$2_$3_$4.log 2>&1; echo "H$1 T$2 $3 $4 ${*:5} rc=$? wall=$(( $(date +%s)-t0 ))" >> /content/qgpu.done; }
for s in 1 2 3 4; do run 16 3 $s exact 11 cuda; run 16 3 $s recipe108 11 cuda; done
for s in 0 1 2 3 4; do run 16 4 $s exact 11 cuda --start truth; run 16 4 $s exact 11 cuda; run 16 4 $s recipe108 11 cuda; done
# noisy-copy cost of exact enumeration (DRI): relative sigma x RMS (leaderboard) and ARC absolute 0.05
for H in 8 16; do run $H 4 0 exact 11 cuda --start truth --leafsum
  for nz in rel0.005 rel0.01 rel0.02 abs0.05; do for i in 0 1; do run $H 4 0:${nz}_$i exact 11 cuda --start truth --leafsum; done; done; done
for s in 0 1 2 3 4; do run 8 3 $s exact 11 cuda --start truth; done
for s in 1 2 3 4; do run 8 5 $s exact 11 cuda --start truth --leafsum; done
T192_RULES5=8,256 run 8 5 0 exact 11 cuda --start truth --leafsum
for s in 0 1 2 3 4; do python mc_gpu.py 16 4 $s 10 $((1640+s)) >> /content/qmc.out 2>&1; done
echo QL4CDONE >> /content/qgpu.done
