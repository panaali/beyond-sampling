#!/bin/bash
# pull results/ from the Colab session (default t192gpu) into this folder
S=${1:-t192gpu}; C=/home/user/colabcli-venv/bin/colab; HERE=$(cd "$(dirname "$0")/.." && pwd)
printf 'import subprocess as s\ns.run("cd /content/cn/m16/approaches/gpu_cost_law && cp /content/qgpu.done /content/qmc.out results/ 2>/dev/null; cp /content/qmc2.out results/ 2>/dev/null; tar czf /content/res.tgz results",shell=True)\n' | timeout 120 $C exec -s $S >/dev/null 2>&1
timeout 300 $C download -s $S /content/res.tgz /tmp/claude-0/res.tgz >/dev/null 2>&1 && tar xzf /tmp/claude-0/res.tgz -C "$HERE" && echo synced
