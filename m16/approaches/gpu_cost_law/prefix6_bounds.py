"""T270 early-stop bounds from the synced shard chunks (merged by full 12-byte region key).
Unprocessed mass U = sum of exact root masses of unprocessed subtrees. Partial region masses m_i (processed pieces only)
are <= true masses, and any n regions gain at most U from unprocessed pieces, so
  lower bound N_delta >= min n : top_n(m) + U >= 1 - delta;  upper bound N_delta <= min n : top_n(m) >= 1 - delta.
Geometry defect (T270 diagnosis, cx6.py): some leaf cells miss vertices / triangulation pieces, so their masses are
UNDERestimates (never over: hull of a subset of the true rays). The exact shortfall per chunk is
D_c = sum(root masses of its subtrees) - leaf sum (roots verified against MC, mc6.py); the bounds stay valid with
U -> U + D (any n regions gain at most D from the missing pieces). Bounds are provisional until the defect is fixed.
g = N_delta(6) / N_delta(5) with N_delta(5) from results/prefix/k5_regionmass.npz. usage: python prefix6_bounds.py"""
import glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "results", "prefix6")
roots = np.load(os.path.join(D, "roots.npz")); rm = roots["mass"]
files = sorted(glob.glob(os.path.join(D, "shard*_chunk*.npz")))
ka, kb, m, pos, cells, leaf = [], [], [], [], 0, 0.0; Dc = {}
nfix = 0
for f in files:
    rf = os.path.join(D, "remass_" + os.path.basename(f))          # remass6.py repair of this chunk, if present
    if os.path.exists(rf):
        f = rf; nfix += 1
    z = np.load(f); ka.append(z["key_a"]); kb.append(z["key_b"]); m.append(z["mass"]); pos.append(z["positions"])
    cells += int(z["cells"]); leaf += float(z["leaf"]); Dc[os.path.basename(f).replace("remass_", "")] = float(rm[z["positions"]].sum() - z["leaf"])
ka, kb, m = np.concatenate(ka), np.concatenate(kb), np.concatenate(m); done = np.concatenate(pos)
o = np.lexsort((ka, kb)); ka, kb, m = ka[o], kb[o], m[o]
new = np.r_[True, (ka[1:] != ka[:-1]) | (kb[1:] != kb[:-1])]
reg = np.add.reduceat(m, np.flatnonzero(new)); reg = np.sort(reg)[::-1]; cum = np.cumsum(reg)
mask = np.ones(len(rm), bool); mask[done] = False; U = float(rm[mask].sum())
Ddef = float(sum(abs(v) for v in Dc.values()))           # |.|: conservative for tiny negative rounding
m5 = np.sort(np.load(os.path.join(HERE, "results", "prefix", "k5_regionmass.npz"))["mass"])[::-1]; c5 = np.cumsum(m5)
out = dict(subtrees_done=int(len(done)), subtrees=int(len(rm)), cells=cells, regions_partial=int(len(reg)),
           processed_mass=leaf, unprocessed_mass_U=U, deficit_D=Ddef, chunks_remassed=nfix,
           worst_chunks=sorted(Dc.items(), key=lambda kv: -kv[1])[:10], roots_sum=float(rm.sum()), g={})
for dl in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
    n5 = int(np.searchsorted(c5, c5[-1] * (1 - dl)) + 1)
    lb = int(np.searchsorted(cum, 1 - dl - U - Ddef) + 1) if 1 - dl - U - Ddef > 0 else 0
    ub = int(np.searchsorted(cum, 1 - dl) + 1) if cum[-1] >= 1 - dl else None
    out["g"][f"{dl:g}"] = dict(N5=n5, N6_lower=lb, N6_upper=ub, g_lower=lb / n5, g_upper=(ub / n5 if ub else None), provisional=True)
json.dump(out, open(os.path.join(D, "bounds.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
