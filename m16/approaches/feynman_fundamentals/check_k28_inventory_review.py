"""Saved k28 inventory/source audit only; no author imports or models."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import ast, hashlib, json, math, time
P=Path('/workspace/work/recover-small_h4_4/review/margin_k28_native')
D=Path('/workspace/work/recover-small_h2_78/m16/approaches/margin_training_recovery')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
start=time.process_time();r=json.loads((P/'inventory.json').read_text())
assert r['complete'] and r['saved_noisy_models']==5120
for p,h in r['pins'].items():assert sha(Path(p))==h
src=Path('/workspace/work/recover-small_h2_910/review/recipe112_recovery/dependencies/understand_m43/est75.py')
assert ast.dump(ast.parse(src.read_text()))==ast.dump(ast.parse((D/'frozen/est75_t112.py').read_text()))
m=json.loads((D/'frozen/margin_k28.json').read_text());t=json.loads((D/'frozen/rung_k28.json').read_text())
assert m['k']==t['k']==28 and set(m['seeds'])==set(t['seeds'])==set(map(str,range(5)))
rows=[]
for seed,saved in enumerate(r['seed_records']):
    a=m['seeds'][str(seed)];b=t['seeds'][str(seed)];truth=b['panel']['truths']
    assert saved['seed']==seed and saved['truth_count']==len(truth)==1024
    assert all(math.isfinite(p) and 0<=p<=1 for p in truth)
    assert saved['clean_truth']==b['clean']['truth']
    assert saved['old_truth_spotcheck_maxdiff']==a['truth_spotcheck_maxdiff']
    assert saved['canonical_checkpoint_url']==f'https://storage.googleapis.com/arc-ml-public/alg/zoo/2nd_argmax_4_3_268435456_{seed}.pth'
    for recipe in ('r85b','r112'):
        q=a['p20'][recipe];sr=saved['recipes'][recipe]
        assert len(q['errs'])==sr['error_count']==1024 and all(math.isfinite(x) for x in q['errs'])
        assert sr['mean_legacy_ops']==q['ops'] and sr['mean_cells']==q['cells']
        assert sr['saved_clean']==a['clean'][recipe] and sr['old_ratio']==q['ratio']
        assert 'model_hashes' not in q and 'ops_by_copy' not in q
        pq=sum((F(p)*(1-F(p)) for p in truth),F())
        ratio=sum((F(e)**2 for e in q['errs']),F())*F(q['ops'])/(72*pq)
        assert math.isclose(float(ratio),q['ratio'],rel_tol=2e-14,abs_tol=1e-28)
        rows.append(dict(seed=seed,recipe=recipe,saved_mean_cost_ratio=float(ratio)))
for k in ('model_identity_verified','old_checkpoint_hashes_present','old_per_copy_model_hashes_present','old_per_copy_costs_present'):assert r[k] is False
out=dict(complete=True,inventory_sha256=sha(P/'inventory.json'),source_pins=r['pins'],saved_models=5120,recipe_groups=10,rows=rows,review_cpu_seconds=time.process_time()-start,new_estimator_or_reference_or_download_calls=0,scope='Source AST, all saved truths/errors, mean-counter ratios and URL derivation checked. Original checkpoint/per-copy identities and per-copy costs remain unverified; rounded saved errors cannot recover original output bits. Canonical object reacquisition is a separate qualified provenance step.')
Path(__file__).with_name('k28_inventory_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
