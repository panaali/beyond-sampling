"""T398b check: every line-walk event prefix (k = 10, i.e. full gate history) of (1,8,2) must be a k = 10 survivor;
also MC: sample far-gap (g23 >= DELTA ||x||inf) points in the triple's cone, compute their full histories, and count how
many event points fall outside the survivors (must be 0) and how many survivors are hit by an event point."""
import glob

import numpy as np

import common_path  # noqa: F401
from bnb import event_prefixes, gates_of

tri = (1, 8, 2)
surv = np.concatenate([np.load(f)["gates"] for f in sorted(glob.glob("certs/surv_d0.333_1-8-2_k10_s*.npz"))])
S = {g.tobytes() for g in surv}
pre, ev = event_prefixes(tri, 10, n_lines=200_000)
P = [gates_of(p, 10, {}).tobytes() for p in pre[10]]
miss = sum(p not in S for p in P)
print(f"k=10 survivors {len(S)}; line-walk event points {ev}, distinct full histories {len(P)}, NOT in survivors: {miss}")
