#!/bin/bash
# create a Colab session and install code + deps: newsession.sh NAME [GPU]   (GPU: T4|L4|A100|...; none = CPU)
S=$1; GPU=$2; C=/home/user/colabcli-venv/bin/colab; ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
cd $ROOT && tar czf /tmp/claude-0/t192code.tgz --exclude=results m16/*.py m16/truth.txt \
  m16/approaches/{estT_fix,understand_m43,arc_m43_repro,gpu_cost_law,capacity_ceiling} || exit 1
timeout 900 $C new -s $S ${GPU:+--gpu $GPU} 2>&1 | tail -1
timeout 300 $C upload -s $S /tmp/claude-0/t192code.tgz /content/t192code.tgz 2>&1 | tail -1
printf 'import subprocess as s\nprint(s.run("cd /content && mkdir -p cn && tar xzf t192code.tgz -C cn && mkdir -p cn/m16/approaches/gpu_cost_law/results && pip install -q numba git+https://github.com/alignment-research-center/alg-zoo 2>&1|tail -1; nproc; nvidia-smi -L 2>/dev/null",shell=True,capture_output=True,text=True).stdout)\n' | timeout 900 $C exec -s $S 2>&1 | tail -3
