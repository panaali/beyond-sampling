# T221: exact Gaussian masses of the M_16,10 prefix regions at k = 5 (shares the A100 with the MC truth)
source /content/cn/m16/approaches/gpu_cost_law/colab/runq.sh
python prefix.py 5 11 0 masses cuda >> /content/pmass.out 2>&1; archive
echo PMASSDONE >> /content/pmass.out; archive
