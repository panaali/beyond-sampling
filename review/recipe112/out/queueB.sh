while pgrep -f "fresh.py 4 abs|fresh.py 5 abs|fresh.py 6 abs" -u $(id -u) > /dev/null || ! grep -q "zoo 4" out/fresh_H456.log 2>/dev/null || [ $(grep -c "zoo 4" out/fresh_H456.log) -lt 3 ]; do sleep 30; done
for H in 4 5 6 8; do python sens.py $H > out/sens_H$H.log 2>&1; done
for H in 4 5 6 8; do
  python fresh.py $H abs 0.02 32 84 --light; python fresh.py $H abs 0.1 32 85 --light
  python fresh.py $H rel 0.0025 32 90 --light; python fresh.py $H rel 0.005 32 91 --light
  python fresh.py $H rel 0.01 32 92 --light; python fresh.py $H rel 0.02 32 93 --light
done > out/noise.log 2>&1
echo QB_DONE > out/queueB.done
