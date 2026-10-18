# prefix k = 6 of M_16,10, capped at 40 min, with region masses (seeded random subtree sample)
source /content/cn/m16/approaches/gpu_cost_law/colab/runq.sh
python prefix.py 6 11 2400 masses cuda >> /content/k6.out 2>&1; archive
echo K6DONE >> /content/k6.out; archive
