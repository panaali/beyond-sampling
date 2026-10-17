"""All fixed rows, raw-ledger conservation and exact stored-float comparisons."""
import gzip,importlib.util,json,time
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
from provenance import HERE,DATA,ROOT,sha
HELPER=ROOT/'recipe112_recovery/summarize_direct.py'
spec=importlib.util.spec_from_file_location('t186_full_summary_helper',HELPER);S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
def metrics(rows):
 out={}
 for convention,sampler,key in [('named_numeric',111,'numeric'),('historical',72,'legacy')]:
  values=[dict(estimate=r['value'],truth=r['truth'],ops=r[key],parity_difference=0.,cell_count_equal=True,parity_pass=True) for r in rows]
  d=S.summarize(values,sampler);ratios=[]
  for r in rows:
   n=r[key]//sampler;v=F(r['truth'])*(1-F(r['truth']));assert n>0 and v>0
   ratios.append((F(r['value'])-F(r['truth']))**2*n/v)
  worst=max(range(len(rows)),key=lambda i:ratios[i]);d.update(individual_losses=sum(q>1 for q in ratios),max_individual_ratio=float(ratios[worst]),worst_copy={k:rows[worst][k] for k in ('seed','noise','copy')})
  out[convention]=d
 return out
def main():
 start=time.process_time();target=HERE/'full_summary.json';assert not target.exists()
 pins=json.loads((HERE/'provenance.json').read_text());assert pins['complete']
 for p,d in pins['pins'].items():assert sha(Path(p))==d
 assert sha(HELPER)==json.loads((HERE/'pilot_summary.json').read_text())['helper_sha256']
 out=dict(complete=False,rows=0,groups=[],noise_groups=[],individual=[],files={},numeric=0,integer=0,legacy=0,research_cpu=0.,max_shard_cpu=0.,max_rss_kib=0,source_sha256=sha(Path(__file__)),helper_sha256=sha(HELPER),provenance_sha256=sha(HERE/'provenance.json'),pilot_gate_sha256=sha(HERE/'pilot_review.json'),scope='Original4096 missing-k30 noisy copies and RECIPE85B only. Named-source111 and historical72 budgets separate. Conditional bootstrap1000 RNG0; excludes saved-truth/geometry uncertainty and full runtime/native/control/memory/setup costs.')
 hashes=set();reuses=[];all_totals=Counter()
 for seed in range(1,5):
  seedrows=[]
  for noise in range(20,24):
   prior=json.loads((DATA/f'k30_s{seed}_noise{noise}.json').read_text());noiserows=[]
   for block in range(4):
    path=HERE/('pilot.json' if (seed,noise,block)==(1,20,0) else f'panel/s{seed}_n{noise}_b{block}.json');d=json.loads(path.read_text())
    assert d['complete'] and 'failure' not in d and (d['seed'],d['noise'],d['block'])==(seed,noise,block)
    assert d['provenance_sha256']==sha(HERE/'provenance.json') and d['sampler_numeric']==111 and d['sampler_legacy']==72
    expected=HERE/('pilot.py' if path.name=='pilot.json' else 'shard.py');assert d['driver_sha256']==sha(expected)
    if path.name!='pilot.json':assert d['pilot_gate_sha256']==sha(HERE/'pilot_review.json')
    assert [r['copy'] for r in d['rows']]==list(range(64*block,64*(block+1)))
    archive=path.with_suffix('.jsonl.gz');assert sha(archive)==d['archive_sha256'];out['files'][str(path.relative_to(HERE))]=sha(path)
    with gzip.open(archive,'rt') as f:
     for row in d['rows']:
      raw=json.loads(next(f));j=row['copy'];saved=prior['rows'][j];assert raw['copy']==j and raw['model_sha256']==row['model_sha256']==saved['model_hash'] and row['model_sha256'] not in hashes
      hashes.add(row['model_sha256']);assert row['truth']==saved['saved_truth']
      b,t=raw['baseline'],raw['traced']
      for key in ('hex','legacy','stats','info'):assert b[key]==t[key]
      assert row['hex']==b['hex'] and row['value']==b['value']==saved['recipes']['r85b']['value']
      assert row['legacy']==b['legacy']==saved['recipes']['r85b']['native_ops'] and row['cells']==b['stats']['cells']==saved['recipes']['r85b']['cells']
      ledger=t['ledger'];total=Counter()
      for site in ledger['sites'].values():total.update(site)
      assert dict(total)==ledger['totals'] and not ledger['complete_implementation_cost']
      numeric=sum(v for k,v in total.items() if not k.startswith('integer_control'));integer=sum(v for k,v in total.items() if k.startswith('integer_control'))
      assert numeric==ledger['named_numeric_total']==row['numeric'] and integer==ledger['integer_control_total']==row['integer'];all_totals.update(total)
      reuse=(seed,noise,j) in ((1,20,0),(4,23,255));assert row['reused_control']==reuse
      if reuse:
       control=json.loads((ROOT/f'recipe85b_native_cost/trained_seed{seed}.json').read_text());assert b==control['baseline'] and t==control['traced'];reuses.append((seed,noise,j))
      noiserows.append(dict(row,seed=seed,noise=noise))
     assert next(f,None) is None
    out['research_cpu']+=d['cpu'];out['max_shard_cpu']=max(out['max_shard_cpu'],d['cpu']);out['max_rss_kib']=max(out['max_rss_kib'],d['rss_kib'])
   assert len(noiserows)==256;out['noise_groups'].append(dict(seed=seed,noise=noise,metrics=metrics(noiserows)));seedrows.extend(noiserows)
  assert len(seedrows)==1024;out['groups'].append(dict(seed=seed,metrics=metrics(seedrows)))
  for row in seedrows:
   individual=dict(seed=seed,noise=row['noise'],copy=row['copy'],model_sha256=row['model_sha256'],ratios={})
   for name,sampler,key in [('named_numeric',111,'numeric'),('historical',72,'legacy')]:
    ratio=(F(row['value'])-F(row['truth']))**2*(row[key]//sampler)/(F(row['truth'])*(1-F(row['truth'])));individual['ratios'][name]=float(ratio)
   out['individual'].append(individual)
  out['numeric']+=sum(r['numeric'] for r in seedrows);out['integer']+=sum(r['integer'] for r in seedrows);out['legacy']+=sum(r['legacy'] for r in seedrows)
 assert len(hashes)==4096 and reuses==[(1,20,0),(4,23,255)] and len(out['files'])==64
 out.update(complete=True,rows=len(hashes),reused_controls=reuses,totals=dict(all_totals),cpu_summary=time.process_time()-start,research_cpu_scope='Current pilot plus63 shards, including parity and ledger I/O. Historical2-control CPU separately recorded; no numerical reruns of those controls.',historical_control_cpu=sum(json.loads((ROOT/f'recipe85b_native_cost/trained_seed{s}.json').read_text())['cpu'] for s in (1,4)))
 target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('complete','rows','numeric','integer','research_cpu','max_shard_cpu','max_rss_kib','cpu_summary')}))
if __name__=='__main__':main()
