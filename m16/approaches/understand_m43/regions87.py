"""Region counts for the two rungs: number of linear regions (distinct gate patterns, each a convex cone) of
zoo_2nd_argmax(4,3) (exact, by enumeration) and (4,4) (lower bound: distinct patterns among 2e6 Gaussian inputs, and
the exact cell count of estT exact mode, which also splits by chamber halves). usage: python regions87.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "arc_m43_repro")); sys.path.insert(0, os.path.join(HERE, "..", "understand_ladder"))
import numpy as np
from exact3 import count_regions
from count_regions import patterns
from zoo44 import zoo
for s in range(5):
    r3 = count_regions(zoo(4, 3, s)); p3 = patterns(zoo(4, 3, s), 2_000_000, np.random.default_rng(1))
    p4 = patterns(zoo(4, 4, s), 2_000_000, np.random.default_rng(1))
    print(f"seed {s}: (4,3) linear regions exact {r3} (sampled patterns {p3}); (4,4) sampled patterns {p4}", flush=True)
