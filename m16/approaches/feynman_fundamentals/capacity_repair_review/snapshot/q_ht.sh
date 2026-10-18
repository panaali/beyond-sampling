# T234 add-on on an L4 (12 vCPU): 6 zoo-code trainings (T = 7, 9 x seeds 0-2) in parallel, then the T = 2..10 table
C=/content/cn/m16/approaches/capacity_ceiling; cd $C; mkdir -p weights results
arch() { date > results/heartbeat_ht; tar czf /content/t234.tmp.tgz -C /content/cn m16/approaches/capacity_ceiling/results m16/approaches/capacity_ceiling/weights 2>/dev/null && mv /content/t234.tmp.tgz /content/t234.tgz; }
python ht_table.py 1e8 > /content/ht_zoo.log 2>&1; cp results/ht_table.json results/ht_table_zoo_only.json; arch
for T in 7 9; do for s in 0 1 2; do python train_zoo.py $T $s > /content/tz_${T}_${s}.log 2>&1 & done; done; wait; arch
python ht_table.py 1e8 > /content/ht.log 2>&1; arch
echo HTDONE >> /content/ht.log; arch
