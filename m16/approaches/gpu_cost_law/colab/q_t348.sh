# T348 census on t192f2, after the T340 truths and DEV3 training finish (DRI order)
until grep -q "rc=" /content/rare_mc.log 2>/dev/null && grep -q DEV3DONE /content/dev3.log 2>/dev/null; do sleep 60; done
mkdir -p /content/t348 && cp /content/census.py /content/t348_weights.npz /content/t348/ && cd /content/t348 && mv t348_weights.npz weights.npz
python census.py 1e9 4 > /content/census.log 2>&1; echo "rc=$?" >> /content/census.log
