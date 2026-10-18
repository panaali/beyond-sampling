"""Saved-source k28 provenance inventory only: no models, estimators or references."""
import ast,hashlib,json,math,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
DATA=Path('/workspace/work/recover-small_h2_78/m16/approaches/margin_training_recovery')
TRACER=Path('/workspace/work/recover-small_h2_910/review/recipe85b_native_cost')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 start=time.process_time();target=HERE/'inventory.json';assert not target.exists()
 manifest=json.loads((DATA/'manifest.json').read_text());pins={str(DATA/'manifest.json'):sha(DATA/'manifest.json')}
 names=['margin_k28.json','rung_k28.json','run_margin.py','run_rung.py','lt.py','est75_t112.py']
 for name in names:
  p=DATA/'frozen'/name;assert sha(p)==manifest[name]['sha256'];pins[str(p)]=sha(p)
 noise=DATA/'dependencies/zoo43.py';assert sha(noise)==manifest['m16/approaches/understand_m43/zoo43.py']['sha256'];pins[str(noise)]=sha(noise)
 source=TRACER.parent/'recipe112_recovery/dependencies/understand_m43/est75.py'
 assert ast.dump(ast.parse(source.read_text()),include_attributes=False)==ast.dump(ast.parse((DATA/'frozen/est75_t112.py').read_text()),include_attributes=False)
 for p in [source,TRACER/'native.py',TRACER/'inventory.json',Path('/workspace/dependencies/alg-zoo/alg_zoo/zoo.py'),Path(__file__)]:pins[str(p)]=sha(p)
 margin=json.loads((DATA/'frozen/margin_k28.json').read_text());rung=json.loads((DATA/'frozen/rung_k28.json').read_text());assert margin['k']==rung['k']==28 and set(margin['seeds'])==set(rung['seeds'])==set(map(str,range(5)))
 seeds=[]
 for seed in range(5):
  m=margin['seeds'][str(seed)];r=rung['seeds'][str(seed)];truth=r['panel']['truths'];assert r['k']==28 and r['seed']==seed and r['panel']['n']==1024 and len(truth)==1024 and all(math.isfinite(v) and 0<=v<=1 for v in truth)
  assert m['truth_spotcheck_maxdiff']<1e-12 and set(m['p20'])=={'r85b','r112'}
  recipes={}
  for name in ('r85b','r112'):
   d=m['p20'][name];assert len(d['errs'])==1024 and all(math.isfinite(v) for v in d['errs']) and d['ops']>0 and d['cells']>0
   assert 'ops_by_copy' not in d and 'model_hashes' not in d
   recipes[name]=dict(error_count=len(d['errs']),mean_legacy_ops=d['ops'],mean_cells=d['cells'],old_ratio=d['ratio'],saved_clean=m['clean'][name])
  seeds.append(dict(seed=seed,truth_count=len(truth),clean_truth=r['clean']['truth'],old_truth_spotcheck_maxdiff=m['truth_spotcheck_maxdiff'],recipes=recipes,canonical_checkpoint_url=f'https://storage.googleapis.com/arc-ml-public/alg/zoo/2nd_argmax_4_3_{2**28}_{seed}.pth'))
 out=dict(complete=True,claim='4dbeebbc',saved_noisy_models=5120,seed_records=seeds,pins=pins,source_ast_equal=True,model_identity_verified=False,old_checkpoint_hashes_present=False,old_per_copy_model_hashes_present=False,old_per_copy_costs_present=False,noise_law='Original lt.panel: seeds20..23 each256, abs.05, NumPy default_rng reset per noise seed; Wi,Wh,Wo order; float64 conversion from original zoo state.',gaps=['No original checkpoint byte hashes recovered from T101/T115 artifacts.','No original per-copy model hashes or per-copy legacy costs/cells saved; only means.','Saved per-copy errors are rounded subtraction results, not original estimate bit patterns.','Canonical checkpoint reacquisition could establish current object identity and model-source consistency, but cannot retroactively prove historical byte identity.'],next_gate='Provenance source review first; separately freeze canonical checkpoint reacquisition/metadata and unchanged estimator-error parity controls if proceeding. No new truth/model selection.',cpu=time.process_time()-start,scope='Saved source/data inventory only. No estimator, model-forward, checkpoint download or reference call.')
 target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('complete','saved_noisy_models','source_ast_equal','model_identity_verified','cpu')}))
if __name__=='__main__':main()
