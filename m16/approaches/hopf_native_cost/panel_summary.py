"""Saved original-source coverage/cost contraction; no model or reference runs."""
import collections,gzip,hashlib,json,time
from pathlib import Path
from tracer import HERE,ROOT

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 dst=HERE/'panel_summary.json';assert not dst.exists();start=time.process_time()
 models=json.loads((HERE/'panel_models.json').read_text());saved=json.loads((ROOT/'frozen/ev_44.json').read_text());rows=[]
 for model in models['rows']:
  s,c=model['seed'],model['copy'];p=HERE/('model_h64_original.json.gz' if (s,c)==(0,0) else f'panel_s{s}_c{c}.json.gz')
  raw=gzip.decompress(p.read_bytes());r=json.loads(raw);assert r['status']=='complete' and r['arm']=='original' and r['nu']==64 and r['nd']==128
  assert r['model_sha256']==model['model_sha256'];old=saved[str(s)]['rows'][c];assert r['value']==old['h64'] and r['legacy']==old['h64_ops']
  assert r['baseline_value']==r['value'] and r['baseline_legacy']==r['legacy']
  for name,h in r['pins'].items():assert sha(ROOT/'frozen/ev_44.json' if name=='frozen_ev_44.json' else HERE/name)==h,(s,c,name)
  for report in (r['ledger'],r['startup']):
   total=collections.Counter()
   for site,cats in report['sites'].items():total.update(cats)
   assert dict(total)==report['totals'];assert sum(total.values())==report['inclusive_explicit_total']
   assert report['named_numeric_total']+report['integer_control_total']==report['inclusive_explicit_total']
   assert not report['complete_implementation_cost']
  d=r['ledger'];closures=collections.Counter()
  for cl in d['source_closures']:closures[cl['name']]+=cl['numerical_allowance']
  assert sum(closures.values())==d['source_closure_allowance'];assert sum(o['count'] for o in d['opaque_calls'])==len(d['source_closures'])
  cold=r['startup']['inclusive_explicit_total'];explicit=d['inclusive_explicit_total']+cold;conditional=explicit+d['source_closure_allowance']
  assert explicit==r['explicit_with_constant_setup'] and conditional==r['conditional_source_with_constant_setup']
  rows.append(dict(seed=s,copy=c,model_sha256=r['model_sha256'],value=r['value'],legacy=r['legacy'],explicit=explicit,conditional=conditional,closures=dict(closures),opaque_calls=len(d['source_closures']),cpu_seconds=r['cpu_seconds'],maxrss_kib=r['maxrss_kib'],file=p.name,compressed_sha256=sha(p),raw_sha256=hashlib.sha256(raw).hexdigest(),reused_prior_record=(s,c) in ((0,0),(0,1))))
 assert {(r['seed'],r['copy']) for r in rows}=={(s,c) for s in range(5) for c in range(5)} and len(rows)==25
 out=dict(claim='304a1ddb',rows=rows,totals={k:sum(r[k] for r in rows) for k in ('legacy','explicit','conditional','cpu_seconds','opaque_calls')},new_continuation_cpu_seconds=sum(r['cpu_seconds'] for r in rows if not r['reused_prior_record']),contraction_cpu_seconds=time.process_time()-start,scope='All25 original saved H4T4 h64 values/counters preserved. Direct per-row source ledgers; original mask defect remains, source closures conditional, no full runtime/reference or performance claim.')
 dst.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'rows':len(rows),'totals':out['totals']}))
if __name__=='__main__':main()
