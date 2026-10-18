# T270 mass-deficit diagnosis on t192f2 (GPU masses): per-subtree leaf vs root for shard1 chunk 1 (largest roots first)
cd /content/cn/m16/approaches/gpu_cost_law
DIAG_DEV=cuda python diag6.py /content/shard1_chunk0001.npz 2 > /content/diag6.log 2>&1
