"""Reference MC (independent of the constraint algebra: membership by the forward pass common.run) for the event
mass of step-10 regions: importance sampling N(x*, s^2 I) at the float dominating point x*.
usage: python mc_ref.py TRI_FILE N s i0 i1 ... -> prints i, mean, se, hits"""
import sys

import numpy as np

import mc

d = np.load(sys.argv[1]); tri = tuple(int(v) for v in d["tri"]); n = int(float(sys.argv[2])); s = float(sys.argv[3])
for i in map(int, sys.argv[4:]):
    m, se, h = mc.mc_mass(tri, d["gates"][i], n=n, s=s, seed=i)
    print(i, m, se, h, flush=True)
