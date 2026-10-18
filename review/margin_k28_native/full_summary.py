"""Saved-only full k28 audit and exact stored-float source-budget contractions."""
import gzip,importlib.util,json,time
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
from inventory import HERE,DATA,TRACER,sha
HELPER=TRACER.parent/'recipe112_recovery/summarize_direct.py'
NAMES=('r85b','r112')
def metrics(rows):
 spec=importlib.util.spec_from_file_location('k28_fixed_budget_helper',HELPER);S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S);out={}
 for convention,sampler,key in [('named_numeric',111,'numeric'),('historical',72,'legacy')]:
  values=[dict(estimate=r['value'],truth=r['truth'],ops=r[key],parity_difference=0.,cell_count_equal=True,parity_pass=True) for r in rows]
  d=S.summarize(values,sampler);ratios=[(F(r['value'])-F(r['truth']))**2*(r[key]//sampler)/(F(r['truth'])*(1-F(r['truth']))) for r in rows]
  worst=max(range(len(rows)),key=lambda i:ratios[i]);d.update(individual_losses=sum(q>1 for q in ratios),max_individual_ratio=float(ratios[worst]),worst_copy={k:rows[worst][k] for k in ('seed','noise','copy')});out[convention]=d
 return out
def checked_pair(b,t,row,truth,old_error):
 for k in ('hex','legacy','stats','info','recipe'):assert b[k]==t[k]
 assert b['value']==t['value']==row['value'] and b['hex']==row['hex'] and row['legacy']==b['legacy'] and row['cells']==b['stats']['cells']
 assert b['value']-truth==old_error==row['old_error']
 ledger=t['ledger'];total=Counter()
 for site in ledger['sites'].values():total.update(site)
 assert dict(total)==ledger['totals'] and not ledger['complete_implementation_cost']
 assert all(isinstance(v,int) and v>=0 for v in total.values())
 integer=sum(v for k,v in total.items() if k.startswith('integer_control'));numeric=sum(total.values())-integer
 assert integer==row['integer']==ledger['integer_control_total'] and numeric==row['numeric']==ledger['named_numeric_total']
 return total
def main():
 start=time.process_time();target=HERE/'full_summary.json';assert not target.exists()
 pins={str(HERE/f):sha(HERE/f) for f in ('inventory.json','reconstruction.json','controls.json','pilot.json','pilot_summary.json','pilot_gate.json','continuation_gate.json','shard.py','pilot.py','controls.py','FULL_PROTOCOL.md')};pins[str(HELPER)]=sha(HELPER);pins[str(Path(__file__))]=sha(Path(__file__))
 assert sha(HELPER)==json.loads((HERE/'pilot_summary.json').read_text())['pins'][str(HELPER)]
 for name in ('inventory.json','pilot_gate.json','continuation_gate.json'):
  gate=json.loads((HERE/name).read_text());assert gate['complete']
  for p,digest in gate['pins'].items():assert sha(Path(p))==digest
 identities=json.loads((HERE/'reconstruction.json').read_text());controls=json.loads((HERE/'controls.json').read_text());assert identities['complete'] and controls['complete'] and len(controls['rows'])==15
 cmap={(r['seed'],r['noise'],r['copy']):r for r in controls['rows']};assert len(cmap)==15
 margin=json.loads((DATA/'frozen/margin_k28.json').read_text());rung=json.loads((DATA/'frozen/rung_k28.json').read_text())
 out=dict(complete=False,pins=pins,files={},groups=[],noise_groups=[],individual=[],clean=[],research_cpu=controls['cpu'],max_process_cpu=controls['cpu'],max_rss_kib=controls['rss_kib'],historical_identity_verified=False,scope='Current-canonical five-seed original k28 specifications, unchanged85B/112 and original saved errors. Historical byte identity unproved. Numeric111/historical72 and conditional pointwise bootstrap; no full native/runtime/control/memory/setup/truth certificate.')
 totals={n:Counter() for n in NAMES};clean_totals={n:Counter() for n in NAMES};hashes=set();reuses=[];pair_count=baseline_count=0
 for seed in range(5):
  seedrows={n:[] for n in NAMES};identity=identities['records'][seed];assert identity['seed']==seed
  for noise in range(20,24):
   noiserows={n:[] for n in NAMES};noise_identity=identity['noises'][noise-20];assert noise_identity['noise']==noise
   for block in range(4):
    pilot=(seed,noise,block)==(0,20,0);path=HERE/('pilot.json' if pilot else f'panel/s{seed}_n{noise}_b{block}.json');d=json.loads(path.read_text())
    assert d['complete'] and 'failure' not in d and (d['seed'],d['noise'],d['block'])==(seed,noise,block)
    assert d['driver_sha256']==sha(HERE/('pilot.py' if pilot else 'shard.py')) and d['gate_sha256']==sha(HERE/('pilot_gate.json' if pilot else 'continuation_gate.json'))
    assert d['controls_sha256']==sha(HERE/'controls.json') and d['reconstruction_sha256']==sha(HERE/'reconstruction.json') and d['sampler_numeric']==111 and d['sampler_legacy']==72 and not d['historical_identity_verified']
    assert [r['copy'] for r in d['rows']]==list(range(block*64,(block+1)*64))
    archive=path.with_suffix('.jsonl.gz');assert sha(archive)==d['archive_sha256'];out['files'][str(path.relative_to(HERE))]=sha(path);out['files'][str(archive.relative_to(HERE))]=sha(archive)
    pairs={};baselines={}
    with gzip.open(archive,'rt') as f:
     for line in f:
      raw=json.loads(line);key=(raw['copy'],raw['recipe'])
      if raw['stage']=='baseline':assert key not in baselines;baselines[key]=raw['baseline']
      else:assert raw['stage']=='paired_before_validation' and key not in pairs;pairs[key]=raw
    assert set(pairs)=={(j,n) for j in range(block*64,(block+1)*64) for n in NAMES}
    for row in d['rows']:
     j=row['copy'];digest=row['model_sha256'];assert digest==noise_identity['hashes'][j] and digest not in hashes;hashes.add(digest)
     truth=rung['seeds'][str(seed)]['panel']['truths'][(noise-20)*256+j];assert row['truth']==truth
     control=cmap.get((seed,noise,j));reuse=control is not None;assert row['reused_control']==reuse
     if reuse:reuses.append((seed,noise,j));assert control['truth']==truth and control['model_sha256']==digest
     for name in NAMES:
      raw=pairs[(j,name)];assert raw['model_sha256']==digest and raw['reused_control']==reuse;b,t=raw['baseline'],raw['traced']
      if reuse:assert (j,name) not in baselines and b==control['recipes'][name]['baseline'] and t==control['recipes'][name]['traced']
      else:assert baselines.pop((j,name))==b
      r=row['recipes'][name];old=margin['seeds'][str(seed)]['p20'][name]['errs'][(noise-20)*256+j];totals[name].update(checked_pair(b,t,r,truth,old));noiserows[name].append(dict(r,seed=seed,noise=noise,copy=j,truth=truth,model_sha256=digest))
    assert not baselines;pair_count+=len(pairs);baseline_count+=128-2*sum(r['reused_control'] for r in d['rows'])
    out['research_cpu']+=d['cpu'];out['max_process_cpu']=max(out['max_process_cpu'],d['cpu']);out['max_rss_kib']=max(out['max_rss_kib'],d['rss_kib'])
   assert all(len(rows)==256 for rows in noiserows.values());out['noise_groups'].append(dict(seed=seed,noise=noise,recipes={n:metrics(noiserows[n]) for n in NAMES}))
   for name in NAMES:seedrows[name].extend(noiserows[name])
  means={}
  for name in NAMES:
   rows=seedrows[name];assert len(rows)==1024;means[name]={}
   for key,oldkey in [('legacy','ops'),('cells','cells')]:
    value=F(sum(r[key] for r in rows),1024);assert value==F(margin['seeds'][str(seed)]['p20'][name][oldkey]);means[name][key]=float(value)
   for r in rows:
    ratios={c:float((F(r['value'])-F(r['truth']))**2*(r[k]//s)/(F(r['truth'])*(1-F(r['truth'])))) for c,s,k in [('named_numeric',111,'numeric'),('historical',72,'legacy')]}
    out['individual'].append(dict(seed=seed,noise=r['noise'],copy=r['copy'],recipe=name,ratios=ratios))
  out['groups'].append(dict(seed=seed,recipes={n:metrics(seedrows[n]) for n in NAMES},original_mean_counter_parity=means))
  control=cmap[(seed,None,None)];assert control['model_sha256']==identity['base_sha256'];truth=rung['seeds'][str(seed)]['clean']['truth'];assert truth==control['truth'];clean={}
  for name in NAMES:
   b,t=control['recipes'][name]['baseline'],control['recipes'][name]['traced'];old=margin['seeds'][str(seed)]['clean'][name]
   assert b['legacy']==old['ops'] and b['stats']['cells']==old['cells']
   r=dict(value=b['value'],hex=b['hex'],legacy=b['legacy'],cells=b['stats']['cells'],numeric=t['ledger']['named_numeric_total'],integer=t['ledger']['integer_control_total'],old_error=old['err'])
   clean_totals[name].update(checked_pair(b,t,r,truth,old['err']));clean[name]=metrics([dict(r,seed=seed,noise=None,copy=None,truth=truth)])
  out['clean'].append(dict(seed=seed,recipes=clean))
 assert len(hashes)==5120 and len(out['files'])==160 and pair_count==10240 and baseline_count==10220 and reuses==[(s,n,j) for s in range(5) for n,j in ((20,0),(23,255))]
 assert out['research_cpu']<=3000 and out['max_process_cpu']<=110
 out.update(complete=True,models=5120,clean_models=5,reused_controls=reuses,paired_records=pair_count,baseline_records=baseline_count,ledger_totals={n:dict(totals[n]) for n in NAMES},clean_ledger_totals={n:dict(clean_totals[n]) for n in NAMES},cpu_summary=time.process_time()-start,research_cpu_scope='Controls once, pilot once and79 shards, including baseline/traced evaluations and ledger I/O; no truth/reference calls.')
 target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('complete','models','clean_models','research_cpu','max_process_cpu','cpu_summary')}))
if __name__=='__main__':main()
