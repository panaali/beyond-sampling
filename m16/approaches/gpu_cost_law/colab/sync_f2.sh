#!/bin/bash
# reclaim insurance for t192f2/t192lb: every 8 min pull grid3's state.pt (resumable via fused_run.py) and the T270 chunks
C=/home/user/colabcli-venv/bin/colab; HERE=$(cd "$(dirname "$0")/.." && pwd); D=$HERE/../../../data/capacity_ceiling/grid3
mkdir -p $D
for i in $(seq 1 15); do
  timeout 300 $C download -s t192f2 /content/cn/data/capacity_ceiling/grid3/state.pt $D/state.pt.new >/dev/null 2>&1 && mv $D/state.pt.new $D/state.pt
  $HERE/colab/sync_t270.sh t192f2 t192lb >/dev/null 2>&1
  echo "$(date -u +%H:%M) $(ls -la $D/state.pt 2>/dev/null | awk '{print $5}') chunks $(ls $HERE/results/prefix6 | grep -c chunk)"; sleep 480
done
