"""Compare independent/rescore.json against results/summary.json (run only after rescore.json was written)."""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
me = json.load(open(os.path.join(HERE, 'rescore.json'))); s = json.load(open(os.path.join(BASE, 'results/summary.json')))
rows = []
def cmp(name, a, b, tol=0.01): rows.append((name, a, b, abs(a - b), abs(a - b) > tol))
for H in ('8', '16', '32'):
    sz = s['sizes'][H]
    for c, v in sz['investigators'].items():
        m = me['runs'][c]; r2 = np.array(m['r2'])
        cmp(f'{c} model_acc', m['model_acc'], v['model_acc']); cmp(f'{c} repl_acc', m['repl_acc'], v['replacement_acc'])
        cmp(f'{c} gap', m['gap'], v['gap_recovered']); cmp(f'{c} n_r2>=0.8', int((r2 >= .8).sum()), v['n_r2_ge_0.8'], 0)
        cmp(f'{c} median_r2', float(np.median(r2)), v['median_r2'])
        cmp(f'{c} legible_judge_frac', float(np.mean(me['legibility_by_run'][c])), v['legible_judge_frac'])
    cmp(f'G_{H}', me['G'][H], sz['G']); cmp(f'S_matched_{H}', me['pair'][H]['matched'], sz['S_matched'])
    cmp(f'S_control_{H}', me['pair'][H]['control'], sz['S_control']); cmp(f'D_{H}', me['pair'][H]['D'], sz['D'])
    cmp(f'legibility_{H}', me['legibility'][H], sz['legible_judge_frac_mean'])
cmp('role_kappa_judges', me['kappa_primary_role'], s['judge_reliability']['role_kappa_judges'])
for r in rows: print(f"{r[0]:28s} mine={r[1]:.6f} theirs={r[2]:.6f} diff={r[3]:.6f}{'  <-- DIFF' if r[4] else ''}")
print('verdicts mine', me['verdicts'], 'theirs', s['verdict'], 'SAME' if me['verdicts'] == s['verdict'] else 'DIFFERENT')
