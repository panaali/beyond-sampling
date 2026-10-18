# T348 census relaunch (all sets partitioned, P = 8) after the T358 counts finish
until grep -q "rc=" /content/t358.log 2>/dev/null; do sleep 30; done
cp /content/census.py /content/t348/census.py && cd /content/t348 && python census.py 1e9 8 > /content/census.log 2>&1; echo "rc=$?" >> /content/census.log
