#!/bin/bash
# T352 width ladder insurance: every 8 min pull qat_h32/qat_h64 checkpoints + logs from t352c (resumable via qtrain.py)
C=/home/user/colabcli-venv/bin/colab; D=/home/user/clean-neurons/data/quantised; Q=/content/cn/m16/approaches/quantised; mkdir -p $D
for i in $(seq 1 15); do
  for t in qat_h32 qat_h64; do
    timeout 300 $C download -s t352c $Q/$t.pt $D/$t.pt.new >/dev/null 2>&1 && mv $D/$t.pt.new $D/$t.pt
    timeout 60 $C download -s t352c /content/$t.log $D/$t.log >/dev/null 2>&1
  done
  echo "$(date -u +%H:%M) $(tail -n1 $D/qat_h32.log 2>/dev/null | cut -c1-50) | $(tail -n1 $D/qat_h64.log 2>/dev/null | cut -c1-50)"; sleep 480
done
