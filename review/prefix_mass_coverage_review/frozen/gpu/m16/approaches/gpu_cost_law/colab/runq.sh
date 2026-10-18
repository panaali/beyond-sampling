# sourced by queue scripts on a Colab VM. run H T name mode [args...]: skip if already done, then run estP and
# re-archive results (/content/res.tgz) after EVERY run, so a reclaimed runtime loses at most the run in flight.
G=/content/cn/m16/approaches/gpu_cost_law
cd $G
run() {
  grep -qxF "H$1 T$2 $3 $4 ${*:5} rc=0" <(sed 's/ wall=.*//' /content/qgpu.done 2>/dev/null) && return
  local t0=$(date +%s); python estP.py "$@" > /content/g_$1_$2_$3_$4.log 2>&1
  echo "H$1 T$2 $3 $4 ${*:5} rc=$? wall=$(( $(date +%s)-t0 ))" >> /content/qgpu.done
  archive
}
mc() { python mc_gpu.py "$@" >> /content/qmc.out 2>&1; archive; }
archive() { for f in qgpu.done qmc.out; do cp /content/$f $G/results/$(hostname)_$f 2>/dev/null; done
            date > $G/results/heartbeat_$(hostname); tar czf /content/res.tmp.tgz -C $G results && mv /content/res.tmp.tgz /content/res.tgz; }
