# T348 census relaunch 2 (balanced partitions, suffix patterns k = 7..10), P = 8
cp /content/census.py /content/t348/census.py && cd /content/t348 && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python census.py 1e9 8 > /content/census.log 2>&1; echo "rc=$?" >> /content/census.log
