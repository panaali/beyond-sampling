"""Fixed25 saved ledger contraction, one pilot reuse, no estimator calls."""
import collections,gzip,hashlib,json,resource,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(19,20));dst=HERE/'panel_summary.json';assert not dst.exists();start=time.process_time()
 out={'complete':False,'rows':[],'source_sha256':sha(__file__)}
 def save():dst.write_text(json.dumps(out,indent=2)+'\n')
 save()
 try:
  scope=json.loads((HERE/'summary_scope.json').read_text())
  for p,h in scope['pins'].items():assert sha(p)==h,p
  models=json.loads((HERE/'models.json').read_text());assert models['complete'] and len(models['models'])==25
  for i,model in enumerate(models['models']):
   seed,copy=divmod(i,5);assert (seed,copy)==(model['seed'],model['copy'])
   meta=HERE/('pilot.json' if i==0 else f'panel/{i:02d}.json');raw=HERE/('pilot_ledger.json.gz' if i==0 else f'panel/{i:02d}_ledger.json.gz')
   r=json.loads(meta.read_text());assert r['complete'] and (r['seed'],r['copy'],r['nu'],r['nd'])==(seed,copy,64,128)
   assert r['source_sha256']==sha(HERE/('pilot.py' if i==0 else 'panel.py'))
   assert r['model_sha256']==model['model_sha256'] and sha(raw)==r['ledger_sha256']
   assert r['value_hex']==r['baseline_hex']==r['saved_hex']==model['saved_h64_hex']==float(r['value']).hex()
   assert r['legacy']==r['baseline_legacy']==r['saved_legacy']==model['saved_legacy']
   data=json.loads(gzip.decompress(raw.read_bytes()));report,startup=data['report'],data['startup']
   for d in (report,startup):
    total=collections.Counter()
    for cats in d['sites'].values():total.update(cats)
    assert dict(total)==d['totals'] and sum(total.values())==d['inclusive_explicit_total']
    assert d['named_numeric_total']+d['integer_control_total']==d['inclusive_explicit_total']
    assert not d.get('complete_implementation_cost',False)
   closures=collections.Counter()
   for cl in report['source_closures']:closures[cl['name']]+=cl['numerical_allowance']
   assert set(closures)<={'np.lexsort','np.searchsorted','np.argsort'}
   assert sum(closures.values())==report['source_closure_allowance']
   assert sum(x['count'] for x in report['opaque_calls'])==len(report['source_closures'])==r['source_closure_count']
   explicit=report['inclusive_explicit_total']+startup['inclusive_explicit_total'];conditional=explicit+sum(closures.values())
   assert explicit==r['explicit_with_constant_setup'] and conditional==r['conditional_source_with_constant_setup']
   out['rows'].append(dict(index=i,seed=seed,copy=copy,model_sha256=r['model_sha256'],value=r['value'],value_hex=r['value_hex'],legacy=r['legacy'],explicit=explicit,conditional=conditional,closures=dict(closures),closure_count=len(report['source_closures']),cpu_seconds=r['cpu_seconds'],baseline_cpu_seconds=r['baseline_cpu_seconds'],max_rss_kib=r['max_rss_kib'],meta_sha256=sha(meta),ledger_sha256=sha(raw),pilot_reused_once=i==0));save()
  assert len({r['model_sha256'] for r in out['rows']})==25
  out.update(complete=True,totals={k:sum(r[k] for r in out['rows']) for k in ('legacy','explicit','conditional','closure_count','cpu_seconds')},new24_cpu_seconds=sum(r['cpu_seconds'] for r in out['rows'][1:]),scope='All25 original H5 saved h64 models; pilot reused once, actual per-model source costs. Known mask defect and conditional sort/search/native exclusions remain; no new truth or general runtime claim.')
 except BaseException as e:out['failure']=repr(e);raise
 finally:out['contraction_cpu_seconds']=time.process_time()-start;save()
 print(json.dumps({'complete':out['complete'],'totals':out.get('totals')}))
if __name__=='__main__':main()
