while kill -0 25041 2>/dev/null; do sleep 20; done
python mc.py > out/mc.log 2>&1
python count.py > out/count.log 2>&1
python adv.py split > out/adv_split.log 2>&1
python adv.py climb > out/adv_climb.log 2>&1
python adv.py diag > out/adv_diag.log 2>&1
python fresh.py 16 abs 0.05 32 80 > out/fresh_H16.log 2>&1
python fresh.py 3 abs 0.05 32 80 > out/fresh_H3.log 2>&1
python fresh.py 2 abs 0.05 32 80 > out/fresh_H2.log 2>&1
python audit.py > out/audit.log 2>&1
echo QA_DONE > out/queueA.done
