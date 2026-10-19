import json, glob, os, re
import numpy as np
from scipy.stats import spearmanr
D = '/home/user/clean-neurons/algzoo/constant_budget_blind/t377'
r = json.load(open(os.path.join(D, 'independent/stage1.json')))
REG = ['M8', 'M16', 'M16_s0', 'M16_s1', 'M16_example', 'M32', 'M32_s1', 'M32_s2']
M = r['models']

def loo(names, feat):
    errs = []
    for k in names:
        tr = [n for n in names if n != k]
        A = np.array([[1, feat(n)] for n in tr]); b = np.array([M[n]['g_bar'] for n in tr])
        c = np.linalg.lstsq(A, b, rcond=None)[0]
        errs.append(c[0] + c[1] * feat(k) - M[k]['g_bar'])
    return float(np.sqrt(np.mean(np.square(errs))))

def verdict(rs, rf):
    if rf <= 0.7 * rs: return 'fragility dominates'
    if rs <= 0.7 * rf: return 'size dominates'
    return 'not separable'

reg = {}
for lab, names in [('H16_32_7models', [n for n in REG if M[n]['H'] in (16, 32)]), ('all8', REG)]:
    rs = loo(names, lambda n: np.log2(M[n]['H'])); rf = loo(names, lambda n: M[n]['F'])
    reg[lab] = {'models': names, 'rmse_size': rs, 'rmse_F': rf, 'verdict': verdict(rs, rf)}
sp = {}
for H in (16, 32):
    names = [n for n in REG if M[n]['H'] == H]
    rho = spearmanr([M[n]['F'] for n in names], [M[n]['g_bar'] for n in names]).correlation
    sp['H%d' % H] = {'models': names, 'rho': float(rho)}

key = json.load(open(os.path.join(D, 'judging/key.json')))
items = {it['id']: it for batch in key['pair'] for it in batch}
SC = {'SAME': 1.0, 'PARTIAL': 0.5, 'DIFFERENT': 0.0}
votes = {}
for fn in sorted(glob.glob(os.path.join(D, 'judging/raw/pair_*_j[12].txt'))):
    for line in open(fn):
        m = re.match(r'\s*([A-Z0-9]+)\s*\|\s*(SAME|PARTIAL|DIFFERENT)\b', line.strip())
        if m: votes.setdefault(m.group(1), []).append(SC[m.group(2)])
missing = [i for i in items if len(votes.get(i, [])) != 2]
extra = [i for i in votes if i not in items]
per = {}
for i, it in items.items():
    if i in votes:
        per.setdefault(it['model'], {}).setdefault(it['kind'], []).append(np.mean(votes[i]))
judge = {}
for mod, d in sorted(per.items()):
    mt, ct = np.mean(d.get('matched', [np.nan])), np.mean(d.get('control', [np.nan]))
    judge[mod] = {'n_matched': len(d.get('matched', [])), 'n_control': len(d.get('control', [])),
                  'mean_matched': float(mt), 'mean_control': float(ct), 'D': float(mt - ct)}
out = dict(r); out['loo_regression'] = reg; out['spearman_within_size'] = sp
out['judging'] = {'per_model': judge, 'n_items': len(items), 'items_without_2_votes': missing, 'unknown_ids': extra}
json.dump(out, open(os.path.join(D, 'independent/rescore.json'), 'w'), indent=1)
print(json.dumps({'reg': reg, 'sp': sp, 'judge': out['judging']}, indent=1))
