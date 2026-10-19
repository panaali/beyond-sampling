mkdir -p /content/t399 && for f in /content/t399_*; do mv "$f" /content/t399/"${f#/content/t399_}"; done
cd /content/t399 && WEIGHTS=/content/t399/m16_weights.npz python gpu_bench.py bench_182_k4.npz bench_182_k4_local.npz bab_d33_a.json,bab_d33_b.json cuda > /content/t399bench.log 2>&1; echo "rc=$?" >> /content/t399bench.log
