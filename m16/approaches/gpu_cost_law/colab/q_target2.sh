# M_16,10 MC truth capped at 300 blocks (3e11 samples; DRI). Resumes from blocks.jsonl; seeds SEED0 + block.
source /content/cn/m16/approaches/gpu_cost_law/colab/runq.sh
while pgrep -f "mc_target.py" >/dev/null; do sleep 5; done
for i in $(seq 1 30); do python mc_target.py $((10*i)) cuda >> /content/target.out 2>&1; archive; done
echo TARGETDONE >> /content/target.out; archive
