#!/bin/bash
# T352 reclaim insurance: every 8 min pull the QAT checkpoints (resumable via qtrain.py) and logs to data/quantised/
C=/home/user/colabcli-venv/bin/colab; D=/home/user/clean-neurons/data/quantised; mkdir -p $D; Q=/content/cn/m16/approaches/quantised
for i in $(seq 1 15); do
  timeout 300 $C download -s t352a $Q/qat.pt $D/qat.pt.new >/dev/null 2>&1 && mv $D/qat.pt.new $D/qat.pt
  for t in qat_ht3 qat_tanh; do timeout 300 $C download -s t352b $Q/$t.pt $D/$t.pt.new >/dev/null 2>&1 && mv $D/$t.pt.new $D/$t.pt; done
  for f in qat qat_eval; do timeout 60 $C download -s t352a /content/$f.log $D/$f.log >/dev/null 2>&1; done
  for f in qat_ht3 qat_tanh qat_ht3_eval qat_tanh_eval; do timeout 60 $C download -s t352b /content/$f.log $D/$f.log >/dev/null 2>&1; done
  echo "$(date -u +%H:%M) $(tail -n1 $D/qat.log 2>/dev/null | cut -c1-40) | $(tail -n1 $D/qat_ht3.log 2>/dev/null | cut -c1-40)"; sleep 480
done
