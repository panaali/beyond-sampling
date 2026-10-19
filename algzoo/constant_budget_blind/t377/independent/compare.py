import json
D = '/home/user/clean-neurons/algzoo/constant_budget_blind/t377'
me = json.load(open(D + '/independent/rescore.json'))
s = json.load(open(D + '/results/summary377.json')); fr = json.load(open(D + '/fragility.json'))
MAP = {'M8': 'M8_s4(default)', 'M16': 'M16_s2(default)', 'M32': 'M32_s0(default)', 'M16_s0': 'M16_s0',
       'M16_s1': 'M16_s1', 'M16_example': 'M16_example', 'M32_s1': 'M32_s1', 'M32_s2': 'M32_s2'}
# account order in summary gaps lists (verified by value): T24 order as given; T377 sorted alphabetically
ORDER = {'M8': ['agsc','gcap','wfpn'], 'M16': ['hnrt','wduk','yyuc'], 'M32': ['dncx','krvc','sfbp'],
         'M16_s0': ['bnic','rtiv'], 'M16_s1': ['briy','kzuy'], 'M16_example': ['phwb','pqid'],
         'M32_s1': ['htmm','rdaq'], 'M32_s2': ['incg','vbya']}
rows, big = [], []
for m, sm in MAP.items():
    a, b = me['models'][m], s['models'][sm]
    for code, g in zip(ORDER[m], b['gaps']):
        d = a['accounts'][code]['gap'] - g; rows.append((m, code, 'gap', a['accounts'][code]['gap'], g, d))
    rows.append((m, '', 'g_bar', a['g_bar'], b['g_bar'], a['g_bar'] - b['g_bar']))
    rows.append((m, '', 'F', a['F'], b['F90'], a['F'] - b['F90']))
    rows.append((m, '', 'acc_model', a['acc_model'], fr[sm]['model_acc'], a['acc_model'] - fr[sm]['model_acc']))
for acc in s['arm160']['accounts']:
    g = me['models']['M32_160call']['accounts'][acc['code']]['gap']
    rows.append(('M32_160call', acc['code'], 'gap', g, acc['gap'], g - acc['gap']))
for r in rows: print('%-12s %-5s %-9s mine=%.4f theirs=%.4f diff=%+.4f' % r)
print('max |diff|:', max(abs(r[5]) for r in rows), ' >0.02:', [r[:3] for r in rows if abs(r[5]) > 0.02])
TM = {'excl_M8': 'H16_32_7models', 'all8': 'all8'}
for k, mk in TM.items():
    t, mm = s['loo'][k], me['loo_regression'][mk]
    print(k, 'RMSE_size %.4f/%.4f RMSE_F %.4f/%.4f verdict %s/%s' % (mm['rmse_size'], t['RMSE_size'], mm['rmse_F'], t['RMSE_F'], mm['verdict'], t['verdict']))
for H in ('16', '32'):
    print('spearman H', H, me['spearman_within_size']['H' + H]['rho'], s['within_size'][H]['spearman_F_g'])
for m, j in me['judging']['per_model'].items():
    t = s['labels'][m]; print(m, 'D mine %.6f theirs %.6f; matched %.4f/%.4f control %.4f/%.4f' % (j['D'], t['D'], j['mean_matched'], t['S_matched'], j['mean_control'], t['S_control']))
