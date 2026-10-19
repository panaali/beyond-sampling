"""Independent judging rescoring + verdicts for T24."""
import json, os, glob, re
from collections import defaultdict, Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
key = json.load(open(os.path.join(BASE, 'judging/key.json')))
recon = json.load(open(os.path.join(HERE, 'recon.json')))['runs']
SIZE = {c: v['H'] for c, v in recon.items()}

def parse(path, ncol):
    d = {}
    for ln in open(path):
        ln = ln.strip()
        if not ln or '|' not in ln: continue
        parts = [p.strip() for p in ln.split('|')]
        if len(parts) < ncol: print('SHORT', path, ln); continue
        if parts[0] in d: print('DUP', path, parts[0])
        d[parts[0]] = parts[1:]
    return d

issues = []
# ---- pair ----
pscore = {'SAME': 1.0, 'PARTIAL': 0.5, 'DIFFERENT': 0.0}
pair_items = {}
for b, batch in enumerate(key['pair']):
    ans = {j: parse(os.path.join(BASE, f'judging/raw/pair_{b:02d}_j{j}.txt'), 2) for j in (1, 2)}
    for it in batch:
        vals = []
        for j in (1, 2):
            a = ans[j].get(it['id'])
            if a is None: issues.append(f"pair missing {it['id']} j{j}"); continue
            v = a[0].upper()
            if v not in pscore: issues.append(f"pair bad verdict {it['id']} j{j} {v}"); continue
            vals.append(pscore[v])
        extra = set(ans[1]) | set(ans[2])
        pair_items[it['id']] = dict(kind=it['kind'], H=it['H'], score=float(np.mean(vals)) if vals else None)
    for j in (1, 2):
        for k in set(ans[j]) - {it['id'] for it in batch}: issues.append(f'pair extra id {k} batch {b} j{j}')
S = {}
for H in (8, 16, 32):
    S[H] = {}
    for kind in ('matched', 'control'):
        sc = [v['score'] for v in pair_items.values() if v['H'] == H and v['kind'] == kind and v['score'] is not None]
        S[H][kind] = float(np.mean(sc)); S[H][kind + '_n'] = len(sc)
    S[H]['D'] = S[H]['matched'] - S[H]['control']
kinds = Counter(v['kind'] for v in pair_items.values())

# ---- role ----
role = {}
for b, batch in enumerate(key['role']):
    ans = {j: parse(os.path.join(BASE, f'judging/raw/role_{b:02d}_j{j}.txt'), 4) for j in (1, 2)}
    for it in batch:
        r = dict(code=it['code'], neuron=it['neuron'])
        for j in (1, 2):
            a = ans[j].get(it['id'])
            if a is None: issues.append(f"role missing {it['id']} j{j}"); r[f'cat{j}'] = r[f'role{j}'] = None; continue
            r[f'cat{j}'] = a[0].upper(); r[f'role{j}'] = a[1].upper()
        role[it['id']] = r
    for j in (1, 2):
        for k in set(ans[j]) - {it['id'] for it in batch}: issues.append(f'role extra id {k} batch {b} j{j}')
pairs = [(r['role1'], r['role2']) for r in role.values() if r['role1'] and r['role2']]
n = len(pairs); po = sum(a == b for a, b in pairs) / n
c1 = Counter(a for a, _ in pairs); c2 = Counter(b for _, b in pairs)
pe = sum(c1[k] * c2[k] for k in set(c1) | set(c2)) / n ** 2
kappa = (po - pe) / (1 - pe)
# coverage check
cov = Counter((r['code'], r['neuron']) for r in role.values())
for c, H in SIZE.items():
    for i in range(H):
        if cov[(c, i)] != 1: issues.append(f'role coverage {c} {i} = {cov[(c,i)]}')
leg_run = {}
for c, v in recon.items():
    fr = []
    for j in (1, 2):
        ok = [1.0 if (r[f'cat{j}'] in ('A', 'B') and v['r2'][r['neuron']] >= 0.8) else 0.0
              for r in role.values() if r['code'] == c]
        fr.append(float(np.mean(ok)))
    leg_run[c] = fr
leg = {H: float(np.mean([x for c in leg_run if SIZE[c] == H for x in leg_run[c]])) for H in (8, 16, 32)}
# self-label vs judge cat agreement (extra)
G = {H: float(np.median([recon[c]['gap'] for c in recon if SIZE[c] == H])) for H in (8, 16, 32)}
recon_v = 'holds' if (G[32] >= .9 and G[32] >= G[8] - .05) else ('degrades' if (G[32] < .9 and G[32] < G[8] - .05) else 'mixed')
lab_v = 'holds' if S[32]['D'] >= S[8]['D'] - .15 else 'degrades'
overall = 'holds up' if (recon_v == 'holds' and lab_v == 'holds') else ('does not hold up' if 'degrades' in (recon_v, lab_v) else 'mixed/inconclusive')
res = dict(seed=json.load(open(os.path.join(HERE, 'recon.json')))['seed'], N=200000,
           runs={c: {k: recon[c][k] for k in ('H', 'model_acc', 'repl_acc', 'gap', 'mean_r2', 'r2')} for c in recon},
           G=G, pair=S, pair_kind_counts=kinds, kappa_primary_role=kappa, kappa_po=po, kappa_pe=pe, kappa_n=n,
           legibility_by_run=leg_run, legibility=leg,
           verdicts=dict(reconstruction=recon_v, labels=lab_v, overall=overall), issues=issues)
json.dump(res, open(os.path.join(HERE, 'rescore.json'), 'w'), indent=1)
print(json.dumps({k: res[k] for k in ('G', 'pair', 'pair_kind_counts', 'kappa_primary_role', 'kappa_n', 'legibility_by_run', 'legibility', 'verdicts', 'issues')}, indent=1))
