#!/bin/bash
# Pause (SIGSTOP) my own queue children while the machine is overloaded; resume when it calms down.
# Only children of my queue scripts (PIDs 8004 and 8236) are touched.
while kill -0 8004 2>/dev/null || kill -0 8236 2>/dev/null; do
  l=$(cut -d' ' -f1 /proc/loadavg)
  kids=$(pgrep -P 8004; pgrep -P 8236)
  if awk "BEGIN{exit !($l > 4.5)}"; then for p in $kids; do kill -STOP $p 2>/dev/null; done
  elif awk "BEGIN{exit !($l < 3.0)}"; then for p in $kids; do kill -CONT $p 2>/dev/null; done; fi
  sleep 60
done
