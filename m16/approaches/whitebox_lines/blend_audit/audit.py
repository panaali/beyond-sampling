"""Read frozen predictions only; never import an estimator."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
PRED = BASE / 'whitebox_stronger' / 'results'
REF = BASE / 'whitebox_cumulants' / 'results' / 'references'
rows = []
for label, ref_label in [('mini0', 'mini_0'), ('full0', 'full_0'), ('full1', 'full_1')]:
    refmeta = json.loads((REF / f'{ref_label}.json').read_text())
    metas = [json.loads((PRED / f'{name}_{label}.json').read_text()) for name in ['e30','logxio']]
    assert all(m['mlp_seed'] == refmeta['mlp_seed'] for m in metas)
    assert metas[0]['weights_sha256'] == metas[1]['weights_sha256']
    assert all(m['local_contract_pass'] for m in metas)
    truth = np.load(REF / f'{ref_label}.npz')['final_means'].astype('float64')
    e, l = [np.load(PRED / f'{name}_{label}.npy')[-1].astype('float64') - truth for name in ['e30','logxio']]
    v_e, v_l, cross = float(np.mean(e*e)), float(np.mean(l*l)), float(np.mean(e*l))
    cost_e, cost_l = [m['cost_fraction'] for m in metas]
    denom = v_e + v_l - 2*cross
    a = (v_l-cross)/denom
    score_e = v_e*max(.1,cost_e)
    def blend(a):
        raw = float(np.mean((a*e+(1-a)*l)**2))
        return {'a':float(a),'mse':raw,'adjusted':raw*max(.1,cost_e+cost_l),'ratio_to_e30':raw*max(.1,cost_e+cost_l)/score_e}
    rows.append({'row':ref_label,'mse_e30':v_e,'mse_logxio':v_l,'cross_error':cross,
                 'error_cosine':cross/(v_e*v_l)**.5,'cost_e30':cost_e,'cost_logxio':cost_l,
                 'score_e30':score_e,'fixed_half':blend(.5),'oracle':blend(a)})
# Optimize a common coefficient for mean cost-adjusted score, correctly weighting
# rows by the cost of both full computations.
num = sum((r['cost_e30']+r['cost_logxio'])*(r['mse_logxio']-r['cross_error']) for r in rows)
den = sum((r['cost_e30']+r['cost_logxio'])*(r['mse_e30']+r['mse_logxio']-2*r['cross_error']) for r in rows)
a = num/den
mean_e = sum(r['score_e30'] for r in rows)/len(rows)
score = sum((r['cost_e30']+r['cost_logxio'])*(a*a*r['mse_e30']+(1-a)**2*r['mse_logxio']+2*a*(1-a)*r['cross_error']) for r in rows)/len(rows)
out = {'rows':rows,'pooled_oracle_a':a,'pooled_oracle_mean_adjusted':score,
       'e30_mean_adjusted':mean_e,'pooled_oracle_ratio_to_e30':score/mean_e,
       'mean_rowwise_oracle_ratio_to_e30':sum(r['oracle']['adjusted'] for r in rows)/len(rows)/mean_e,
       'scope':'Three reused DEV rows, finite-noise truth, both full costs charged; no shared compute and no runtime test.'}
(HERE / 'results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
