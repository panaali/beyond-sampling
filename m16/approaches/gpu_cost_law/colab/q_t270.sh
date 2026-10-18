# T270 shard runner: waits for the VM's current job (pattern $2), then runs prefix6 shard $1 of 2 (11 workers)
G=/content/cn/m16/approaches/gpu_cost_law; cd $G
while pgrep -f "$2" >/dev/null; do sleep 20; done
python prefix6.py $1 2 11 40 >> /content/t270_shard$1.log 2>&1
echo T270DONE >> /content/t270_shard$1.log
