# T270 remass queue on t192lb: chunks in descending deficit D (colab/remass_order.txt), 11 workers each, until the
# deadline (epoch seconds, $1) -> results/prefix6/remass_<chunk>.npz; shard-0 chunks must be uploaded to results/prefix6.
G=/content/cn/m16/approaches/gpu_cost_law; cd $G; END=$1
for c in $(cat colab/remass_order.txt); do
  [ $(date +%s) -ge $END ] && { echo "deadline" >> /content/remass.log; break; }
  [ -f results/prefix6/remass_$c ] && continue
  [ -f results/prefix6/$c ] || { echo "missing $c" >> /content/remass.log; continue; }
  python remass6.py results/prefix6/$c 11 cuda >> /content/remass.log 2>&1
done
echo REMASSDONE >> /content/remass.log
