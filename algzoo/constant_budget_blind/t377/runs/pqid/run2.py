from fit import *
import probe, sys, json
rng = np.random.default_rng(5)
x = rng.standard_normal((40000, 10)); F = feats(x); h = probe.hidden(x)
xt = np.random.default_rng(6).standard_normal((40000, 10)); Ft = feats(xt); ht = probe.hidden(xt)
jobs = json.loads(sys.argv[1])
for job in jobs:
    i, t = job[0], job[1]
    if t == 'W': SETS['j'] = job[2]
    if t == 'S':
        sg = np.zeros(10); sg[job[2]] = 1; sg[job[3]] = -1; SETS['cur'] = sg
    p0 = np.array(T[t][1], float); inits = []
    for c in range(0, 10, 2):
        q = p0.copy()
        if t == 'W' or t[0] == 'Q': q[1] = c
        elif t != 'S': q[2] = c
        else: q[1] = (c - 4) / 2
        inits.append(q)
    p = fitone(h[:, i], F, t, inits); pr = T[t][0](p, Ft)
    print(job, 'R2test %.4f' % (1 - ((ht[:, i] - pr) ** 2).mean() / ht[:, i].var()), np.round(p, 4).tolist(), flush=True)
