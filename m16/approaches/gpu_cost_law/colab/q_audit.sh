# T196 nominal FLOP audit of unchanged estT recipe108 at (8,5), seeds 0-4 (CPU runtime, 2 at a time)
source /content/cn/m16/approaches/gpu_cost_law/colab/runq.sh
for s in 0 1 2 3 4; do echo $s; done | xargs -P2 -I{} sh -c 'python audit.py 8 5 {} recipe108 > /content/audit_{}.log 2>&1'
archive; echo AUDITDONE >> /content/audit_done
