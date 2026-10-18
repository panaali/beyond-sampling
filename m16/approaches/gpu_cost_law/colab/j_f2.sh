# t192f2 (replacement for reclaimed t192fu): resume T270 shard 0 from the synced chunks + restart grid3
G=/content/cn/m16/approaches/gpu_cost_law; mkdir -p $G/results/prefix6 && tar xf /content/p6.tar -C $G/results/prefix6
cp /content/cn/m16/approaches/gpu_cost_law/colab/keepalive.py /content/ 2>/dev/null
( cd $G && python prefix6.py 0 2 11 40 >> /content/t270_shard0.log 2>&1; echo T270DONE >> /content/t270_shard0.log ) &
sleep 120; bash /content/cn/m16/approaches/capacity_ceiling/q_grid3.sh
