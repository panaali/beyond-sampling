# prefix regions of M_16,10 (geometry only, CPU): k = 2..5 complete, k = 6 capped at 1 h
source /content/cn/m16/approaches/gpu_cost_law/colab/runq.sh
for k in 2 3 4 5; do python prefix.py $k 12 >> /content/prefix.out 2>&1; archive; done
python prefix.py 6 12 3600 >> /content/prefix.out 2>&1; archive
echo PREFIXDONE >> /content/prefix.out; archive
