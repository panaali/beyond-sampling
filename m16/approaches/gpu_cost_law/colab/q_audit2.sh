# T196 (8,5) audit, seeds 1-4 (relaunch: the first queue was terminated after seed 0), 2 at a time, CPU runtime
cd /content/cn/m16/approaches/gpu_cost_law
for s in 1 2 3 4; do echo $s; done | xargs -P2 -I{} sh -c 'python audit.py 8 5 {} recipe108 > /content/audit_{}.log 2>&1'
echo AUDIT2DONE >> /content/audit_done
