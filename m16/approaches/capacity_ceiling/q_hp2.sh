# T284 HP2 on t192lb: bit-identity test of fused_gid.py, then (only if every check is True) the 100-model HP2 retrain
cd /content/cn/m16/approaches/capacity_ceiling
python test_gid.py > /content/test_gid.log 2>&1
if grep -q "False" /content/test_gid.log || ! grep -q "alone vs packed" /content/test_gid.log; then echo TESTFAIL >> /content/hp2.log; exit 1; fi
python hp2_run.py >> /content/hp2.log 2>&1
