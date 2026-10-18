cd /content/cn/m16/approaches/gpu_cost_law
while pgrep -f "bash qgpu2.sh" >/dev/null; do sleep 10; done
run() { local t0=$(date +%s); python estP.py $1 $2 $3 $4 11 cuda > /content/g_$1_$2_$3_$4.log 2>&1; echo "H$1 T$2 $3 $4 rc=$? wall=$(( $(date +%s)-t0 ))" >> /content/qgpu.done; }
for k in 1000 1001 1002 1003; do run 8 4 0:fresh$k exact; run 8 4 0:fresh$k recipe108; done
for k in 1000 1001 1002 1003; do run 8 5 0:fresh$k exact; run 8 5 0:fresh$k recipe108; done
echo Q3DONE >> /content/qgpu.done
