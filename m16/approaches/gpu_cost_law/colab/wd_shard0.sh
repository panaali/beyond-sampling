# t192f2 watchdog for T270 shard 0 (its main process RSS grew ~20 GB/h): when a new chunk line appears in the log and
# the main RSS exceeds LIM KB, kill that prefix6 (main + pool workers, by PID) and restart it (resumes from the chunks).
# Ends when a prefix6 it started exits by itself with rc 0 (writes T270DONE_REAL); the old wrapper's "T270DONE" after a
# kill is bogus. The VM runs only this project's processes, so pgrep -f is safe here.
G=/content/cn/m16/approaches/gpu_cost_law; L=/content/t270_shard0.log; LIM=${LIM:-22000000}; cd $G
M=""; for p in $(pgrep -f "prefix6.py 0 2 11 40"); do [ "$(ps -o comm= -p $(ps -o ppid= -p $p))" != "python3" ] && M=$p; done
while true; do
  if [ -z "$M" ] || ! kill -0 $M 2>/dev/null; then
    if [ -n "$MINE" ]; then wait $M; rc=$?; [ $rc -eq 0 ] && { echo T270DONE_REAL >> $L; exit 0; }; fi
    echo "WD $(date -u +%H:%M) start prefix6 shard 0" >> $L
    python prefix6.py 0 2 11 40 >> $L 2>&1 & M=$!; MINE=1; sleep 60; continue
  fi
  n=$(grep -c chunk $L); sleep 20
  if [ "$(grep -c chunk $L)" != "$n" ] && [ "$(ps -o rss= -p $M)" -gt $LIM ]; then
    echo "WD $(date -u +%H:%M) kill main $M rss $(ps -o rss= -p $M)" >> $L
    for c in $(pgrep -P $M); do kill $c; done; kill $M; sleep 10; wait $M 2>/dev/null; MINE=""
  fi
done
