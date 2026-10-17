"""Saved-only fixed64 ledger reconciliation and unchanged matched-budget summaries."""
import gzip,importlib.util,json,time
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
from inventory import HERE,TRACER,sha
HELPER=TRACER.parent/'recipe112_recovery/summarize_direct.py'
def main():
 start=time.process_time();target=HERE/'pilot_summary.json';assert not target.exists();d=json.loads((HERE/'pilot.json').read_text());assert d['complete'] and len(d['rows'])==64 and [r['copy'] for r in d['rows']]==list(range(64))
 archive=HERE/'pilot.jsonl.gz';assert sha(archive)==d['archive_sha256'];pairs={};baselines={}
 with gzip.open(archive,'rt') as f:
  for line in f:
   r=json.loads(line);key=(r['copy'],r['recipe'])
   if r['stage']=='baseline':assert key not in baselines;baselines[key]=r['baseline']
   else:assert r['stage']=='paired_before_validation' and key not in pairs;pairs[key]=r
 assert len(pairs)==128 and len(baselines)==126
 totals={name:Counter() for name in ('r85b','r112')}
 for row in d['rows']:
  for name in ('r85b','r112'):
   key=(row['copy'],name);raw=pairs[key];b,t=raw['baseline'],raw['traced'];assert row['model_sha256']==raw['model_sha256'] and raw['reused_control']==row['reused_control']==(row['copy']==0)
   if row['copy']!=0:assert baselines[key]==b
   for field in ('hex','legacy','stats','info','recipe'):assert b[field]==t[field]
   value=row['recipes'][name];assert value['value']==b['value'] and value['hex']==b['hex'] and value['legacy']==b['legacy'] and value['cells']==b['stats']['cells'] and b['value']-row['truth']==value['old_error']
   ledger=t['ledger'];sums=Counter()
   for site in ledger['sites'].values():sums.update(site)
   assert dict(sums)==ledger['totals'];integer=sum(v for k,v in sums.items() if k.startswith('integer_control'));numeric=sum(sums.values())-integer
   assert integer==value['integer']==ledger['integer_control_total'] and numeric==value['numeric']==ledger['named_numeric_total'];totals[name].update(sums)
 spec=importlib.util.spec_from_file_location('t115_k28_summary_helper',HELPER);S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S);groups={}
 for name in ('r85b','r112'):
  groups[name]={}
  for convention,sampler,key in [('named_numeric',111,'numeric'),('historical',72,'legacy')]:
   rows=[dict(estimate=r['recipes'][name]['value'],truth=r['truth'],ops=r['recipes'][name][key],parity_difference=0.,cell_count_equal=True,parity_pass=True) for r in d['rows']]
   g=S.summarize(rows,sampler);ratios=[(F(r['estimate'])-F(r['truth']))**2*(r['ops']//sampler)/(F(r['truth'])*(1-F(r['truth']))) for r in rows];g.update(individual_losses=sum(x>1 for x in ratios),max_individual_ratio=float(max(ratios)));groups[name][convention]=g
 out=dict(complete=True,groups=groups,ledger_totals={k:dict(v) for k,v in totals.items()},rows=64,pilot_cpu=d['cpu'],projected_80_shard_cpu=80*d['cpu'],cpu_summary=time.process_time()-start,pins={str(p):sha(p) for p in (HERE/'pilot.json',archive,HELPER,Path(__file__))},scope='Fixed64 only. Resource projection is not runtime guarantee. Both unchanged recipes, current canonical/original-error parity, numeric111 vs historical72; saved-truth conditional bootstrap, no historical-byte/fullnative/fullpanel guarantee.')
 target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(complete=True,rows=64,groups=groups,projected_cpu=out['projected_80_shard_cpu'])))
if __name__=='__main__':main()
