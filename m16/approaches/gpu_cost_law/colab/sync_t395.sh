#!/bin/bash
# T395 insurance: every 6 min pull the per-partition checkpoint + log from t395b (census10.py resumes from it)
C=/home/user/colabcli-venv/bin/colab; D=/home/user/clean-neurons/data/t395; mkdir -p $D
for i in $(seq 1 40); do
  timeout 300 $C download -s t395b /content/census10.json.partial $D/census10.json.partial.new >/dev/null 2>&1 && mv $D/census10.json.partial.new $D/census10.json.partial
  timeout 60 $C download -s t395b /content/t395.log $D/t395.log >/dev/null 2>&1
  echo "$(date -u +%H:%M) $(tail -n1 $D/t395.log 2>/dev/null | cut -c1-80)"; grep -q "rc=" $D/t395.log 2>/dev/null && break; sleep 360
done
