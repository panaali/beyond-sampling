cd /content/cn/m16/approaches/estT_fix
for s in 1 2 3 4; do for m in exact recipe108; do echo "zoo$s $m"; done; done | \
 xargs -P2 -L1 sh -c 'start=$(date +%s); python run5.py $0 $1 > /content/q85_$0_$1.log 2>&1; echo "$0 $1 rc=$? wall=$(( $(date +%s)-start ))" >> /content/q85.done'
