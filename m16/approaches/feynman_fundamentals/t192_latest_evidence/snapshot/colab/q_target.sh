# M_16,10 high-precision MC truth: 1000 blocks x 1e9 (float64), archived every 10 blocks (resumable)
source /content/cn/m16/approaches/gpu_cost_law/colab/runq.sh
while pgrep -f "estP.py" >/dev/null; do sleep 15; done; archive
for i in $(seq 1 100); do python mc_target.py $((10*i)) cuda >> /content/target.out 2>&1; archive; done
echo TARGETDONE >> /content/target.out; archive
