"""Losslessly archive complete original ledgers; retain local raw copies."""
import gzip,hashlib,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
 dst=HERE/'model_h64_summary.json';assert not dst.exists();out={'arms':[],'archives':[],'scope':'One existing clean seed0 h64 case; no panel transfer. Source closure allowances are mixed-category and conditional on pinned native paths, not a complete runtime budget.'};start=time.process_time()
 for arm in ('original','tile'):
  p=HERE/f'model_h64_{arm}.json';raw=p.read_bytes();d=json.loads(raw);assert d['status']=='complete'
  rep=d['ledger'];sums={}
  for r in rep['sites'].values():
   for k,v in r.items():sums[k]=sums.get(k,0)+v
  assert sums==rep['totals'];assert sum(sums.values())==rep['inclusive_explicit_total']
  by={}
  for r in rep['source_closures']:by[r['name']]=by.get(r['name'],0)+r['numerical_allowance']
  assert sum(by.values())==rep['source_closure_allowance']
  z=p.with_suffix('.json.gz');assert not z.exists();packed=gzip.compress(raw,compresslevel=9,mtime=0);assert gzip.decompress(packed)==raw;z.write_bytes(packed)
  out['archives'].append({'path':z.name,'raw_sha256':hashlib.sha256(raw).hexdigest(),'compressed_sha256':hashlib.sha256(packed).hexdigest(),'raw_bytes':len(raw),'compressed_bytes':len(packed)})
  out['arms'].append({k:d[k] for k in ('arm','value','legacy','explicit_with_constant_setup','conditional_source_with_constant_setup','cpu_seconds','maxrss_kib','model_sha256')});out['arms'][-1]['closures_by_native']=by;out['arms'][-1]['opaque_calls']=sum(r['count'] for r in rep['opaque_calls'])
 assert all(out['arms'][0][k]==out['arms'][1][k] for k in ('value','legacy','explicit_with_constant_setup','conditional_source_with_constant_setup','closures_by_native','opaque_calls'))
 out['archive_cpu_seconds']=time.process_time()-start;dst.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['arms']))
if __name__=='__main__':main()
