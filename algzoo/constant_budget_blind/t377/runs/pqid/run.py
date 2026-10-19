from fit import *
if True:
    import probe
    jobs = json.loads(sys.argv[1])  # {neuron: [templates]}
    rng = np.random.default_rng(5)
    x = rng.standard_normal((40000, 10)); F = feats(x); h = probe.hidden(x)
    xt = np.random.default_rng(6).standard_normal((40000, 10)); Ft = feats(xt); ht = probe.hidden(xt)
    for i, tl in jobs.items():
        i = int(i)
        for t in tl:
            p0 = np.array(T[t][1], float)
            inits = []
            for c in range(0, 9, 2):
                q = p0.copy()
                if t[0] in "GK": q[1 if t[0]=="G" else 2] = c
                else: q[4] = c * 0.7
                inits.append(q)
            p = fitone(h[:, i], F, t, inits)
            pr = T[t][0](p, Ft)
            r2 = 1 - ((ht[:, i] - pr) ** 2).mean() / ht[:, i].var()
            print(i, t, 'R2test %.4f' % r2, np.round(p, 4).tolist(), flush=True)
