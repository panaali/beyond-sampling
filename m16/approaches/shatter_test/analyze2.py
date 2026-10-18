"""T239 follow-up: (a) correct / mixed / wrong leaves by tie-distance stratum; (b) fraction of leaves adjacent to a leaf
with a different predicted class. Adjacency = sharing >= d-1 rays (unit rays rounded to 1e-7): exact for face-to-face
neighbours, missing neighbours across non-matching subdivisions (so (b) is a lower bound).
A leaf "touches another class" if its own rays disagree (a class boundary inside) or a neighbour's ray classes include a
class other than its own. -> summary2.json"""
import glob, json, os
from collections import Counter, defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); EDGES = (0.0, 0.02, 0.05, 0.1, 0.2, 9.0); OUT = {}
for f in sorted(glob.glob(os.path.join(HERE, "out", "rays_*.npz"))):
    tag = os.path.basename(f)[5:-4]; z = np.load(f, allow_pickle=True)
    err, mass, rays, cls = z["err"], z["mass"], z["rays"], z["cls"]; n = len(err); d = rays[0].shape[1]
    kind = np.where(err <= 0, 0, np.where(err >= mass * (1 - 1e-9), 2, 1))          # 0 correct, 1 mixed, 2 wrong
    U = np.array([(lambda u: u / np.linalg.norm(u))(R.mean(0)) for R in rays]); Us = np.sort(U, 1)[:, ::-1]
    tie = np.minimum(Us[:, 0] - Us[:, 1], Us[:, 1] - Us[:, 2]) / np.sqrt(2); blind = Us[:, 1] <= 0
    strat = np.where(blind, 5, np.digitize(tie, EDGES[1:-1]))
    names = ["<.02", ".02-.05", ".05-.1", ".1-.2", ">=.2", "blind"]
    tab = {names[s]: {k: int(((strat == s) & (kind == i)).sum()) for i, k in enumerate(("correct", "mixed", "wrong"))} for s in range(6)}
    for s in range(6):
        sel = strat == s; tab[names[s]].update(mass=float(mass[sel].sum()), err=float(err[sel].sum()))
    # adjacency
    keyof = lambda r: tuple(np.round(r, 7)); bucket = defaultdict(list)
    for i, R in enumerate(rays):
        for r in R:
            bucket[keyof(r)].append(i)
    shared = Counter()
    for ids in bucket.values():
        if 1 < len(ids) < 2000:
            ids = sorted(set(ids))
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    shared[(ids[a], ids[b])] += 1
    nb = defaultdict(set)
    for (a, b), c in shared.items():
        if c >= d - 1:
            nb[a].add(b); nb[b].add(a)
    S = [set(c.tolist()) for c in cls]; own = [next(iter(s)) if len(s) == 1 else None for s in S]
    touch = np.array([own[i] is None or any(S[j] - {own[i]} for j in nb[i]) for i in range(n)])
    has_nb = np.array([len(nb[i]) > 0 for i in range(n)])
    bad = kind > 0
    touch_err = np.array([bad[i] or any(bad[j] for j in nb[i]) for i in range(n)])
    # connected components of fully-correct leaves (graph restricted to correct leaves)
    comp = -np.ones(n, int); sizes = []; cm = []
    for i in range(n):
        if kind[i] or comp[i] >= 0:
            continue
        st = [i]; comp[i] = len(sizes); cnt = 0; ms = 0.0
        while st:
            a = st.pop(); cnt += 1; ms += mass[a]
            for b in nb[a]:
                if not kind[b] and comp[b] < 0:
                    comp[b] = comp[i]; st.append(b)
        sizes.append(cnt); cm.append(ms)
    cm = np.array(cm); order = np.argsort(cm)[::-1]
    OUT[tag] = dict(touch_error_frac=float(touch_err.mean()), correct_clean_mass=float(mass[(kind == 0) & ~touch_err].sum()),
                    correct_clean_cells=int(((kind == 0) & ~touch_err).sum()), correct_components=len(sizes),
                    top_components=[dict(cells=int(sizes[j]), mass=float(cm[j])) for j in order[:5]],
                    components_for_90pct_correct_mass=int(np.searchsorted(np.cumsum(cm[order]) / cm.sum(), 0.9) + 1),leaves=n, by_stratum=tab, multiclass_frac=float(np.mean([o is None for o in own])),
                    touch_frac=float(touch.mean()), touch_mass_frac=float(mass[touch].sum() / mass.sum()),
                    touch_frac_by_kind={k: float(touch[kind == i].mean()) if (kind == i).any() else None for i, k in enumerate(("correct", "mixed", "wrong"))},
                    correct_far_untouched_mass=float(mass[(kind == 0) & ~touch].sum()), no_neighbour_frac=float(1 - has_nb.mean()),
                    mean_neighbours=float(np.mean([len(nb[i]) for i in range(n)])))
    o = OUT[tag]; print(tag, "touch-error %.3f clean-correct cells %d mass %.3f; correct components %d, top %s, comps for 90%% %d" % (o["touch_error_frac"], o["correct_clean_cells"], o["correct_clean_mass"], o["correct_components"], o["top_components"][:3], o["components_for_90pct_correct_mass"]))
    print(tag, n, "touch %.3f (mass %.3f) by kind %s multi %.3f no-nb %.3f nb %.1f" % (o["touch_frac"], o["touch_mass_frac"],
          {k: round(v, 3) for k, v in o["touch_frac_by_kind"].items() if v is not None}, o["multiclass_frac"], o["no_neighbour_frac"], o["mean_neighbours"]))
    for s, v in tab.items():
        print("   ", s, v)
json.dump(OUT, open(os.path.join(HERE, "summary2.json"), "w"), indent=1)
