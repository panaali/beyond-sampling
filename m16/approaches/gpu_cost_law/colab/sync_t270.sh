#!/bin/bash
# pull new T270 chunk files (and roots/logs) from both shard VMs: sync_t270.sh SESSION...
C=/home/user/colabcli-venv/bin/colab; HERE=$(cd "$(dirname "$0")/.." && pwd); D=$HERE/results/prefix6; mkdir -p $D
for S in "$@"; do
  for f in $(timeout 120 $C ls -s $S content/cn/m16/approaches/gpu_cost_law/results/prefix6 2>/dev/null | grep -oE "(shard[0-9]_chunk[0-9]+|roots)\.npz"); do
    [ -f $D/$f ] || timeout 300 $C download -s $S /content/cn/m16/approaches/gpu_cost_law/results/prefix6/$f $D/$f >/dev/null 2>&1
  done
  for k in 0 1; do timeout 60 $C download -s $S /content/t270_shard$k.log $D/${S}_shard$k.log >/dev/null 2>&1; done
done; ls $D | grep -c chunk
