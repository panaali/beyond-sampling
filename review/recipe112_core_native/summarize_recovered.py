"""Complete saved metadata/budget contraction after reviewed counter-provenance recovery."""
from pathlib import Path
from decimal import Decimal as D,localcontext
import gzip,json,time
from summarize import group,dec,sha,HERE,NOISES

def main():
 start=time.process_time();out=HERE/'summary_recovered.json';assert not out.exists()
 gate=json.loads((HERE/'summary_recovery_gate.json').read_text())
 for p,s in gate['pins'].items():assert sha(Path(p))==s,p
 recovery=json.loads((HERE/'recovery_gate.json').read_text());failed=json.loads((HERE/'panel/H4_z2_n80.json').read_text())
 failrows=json.loads((HERE/'recovery_evaluations.json').read_text())['rows'];pilots=json.loads((HERE/'pilots.json').read_text());prov=json.loads((HERE/'provenance.json').read_text())
 assert not failed['complete'] and failed['source_calls_started']==failed['source_calls_completed']==4
 paths=[];groups=[];blocks=[];allrows=[];losses=[];topology=[];calls=0;reuse=0;failreuse=0;numeric=0;integer=0;total_cpu=D(0)
 for h in (4,5,6):
  for z in range(5):
   selected=[]
   for noise in NOISES:
    stem=f'H{h}_z{z}_n{noise}';isold=stem in recovery['completed_shards'];folder='panel' if isold else 'panel_v2'
    assert isold or stem in recovery['remaining_shards'];path=HERE/folder/(stem+'.json');d=json.loads(path.read_text())
    expected_gate=HERE/('panel_gate.json' if isold else 'recovery_gate.json')
    assert d['complete'] and (d['H'],d['zoo'],d['noise'])==(h,z,noise) and d['gate_sha256']==sha(expected_gate)
    assert d['sampler_numeric']==4*h*h+12*h-1 and [r['copy'] for r in d['rows']]==list(range(64))
    archive=path.with_suffix('.jsonl.gz');assert sha(archive)==d['archive_sha256']
    original=HERE/'frozen/core'/('core_'+stem+'.json');assert sha(original)==d['prior_shard_sha256'];old=json.loads(original.read_text());assert len(old['rows'])==64
    # Conserve raw semantic metadata only. Independent reviewer owns full site-ledger cost audit.
    metadata={}
    with gzip.open(archive,'rt') as f:
     for line in f:
      ev=json.loads(line)
      if ev['event']!='evaluations':continue
      i=ev['copy'];assert i not in metadata;b,t=ev['baseline'],ev['traced']
      for key in ('hex','legacy','stats','info'):assert b[key]==t[key],(stem,i,key)
      ispilot=z==0 and noise==80 and i==0
      if ispilot:
       p=next(r for r in pilots['rows'] if r['H']==h);assert b==p['baseline'] and t==p['traced']
      isfailed=stem==recovery['accepted_failure'] and i in (0,1)
      if isfailed:assert b==failrows[i]['baseline'] and t==failrows[i]['traced']
      metadata[i]={'hex':b['hex'],'legacy':b['legacy'],'stats':b['stats'],'numeric':t['ledger']['named_numeric_total'],'integer':t['ledger']['integer_control_total'],'pilot':ispilot,'failed':isfailed}
    assert sorted(metadata)==list(range(64))
    for r,s in zip(d['rows'],old['rows'],strict=True):
     i=r['copy'];m=metadata[i]
     assert r['stage']=='complete' and r['parity'] and r['hex']==float(r['estimate']).hex()==float(s['saved_estimate']).hex()==m['hex']
     assert (r['truth'],r['model_sha256'],r['legacy'])==(s['truth'],s['model_sha256'],s['saved_ops']) and r['legacy']==m['legacy']
     assert r['cells']==m['stats']['cells'] and r['named_numeric']==m['numeric'] and r['integer_control']==m['integer']
     if isold:assert r['reused_pilot']==m['pilot'] and not m['failed']
     else:
      expect='pilot' if m['pilot'] else ('failed_v1_saved_evaluations' if m['failed'] else None)
      assert r['reused_from']==expect and r['native_stats']==m['stats'] and r['saved_scalar_stats']==s['stats'] and r['saved_scalar_cells']==s['saved_cells']
      assert r['scalar_native_cells_equal']==(m['stats']['cells']==s['saved_cells'])
     if m['stats']!=s['stats']:topology.append({'H':h,'zoo':z,'noise':noise,'copy':i,'model_sha256':r['model_sha256'],'native':m['stats'],'scalar':s['stats']})
     r=dict(r,H=h,zoo=z,noise=noise);selected.append(r);allrows.append(r)
     with localcontext() as ctx:
      ctx.prec=80;p=dec(r['truth']);e=dec(r['estimate'])-p;n=r['named_numeric']//d['sampler_numeric'];ratio=e*e/(p*(1-p)/D(n))
      if ratio>=1:losses.append({'H':h,'zoo':z,'noise':noise,'copy':i,'model_sha256':r['model_sha256'],'ratio':str(ratio),'tie':ratio==1})
    b=group(d['rows'],d['sampler_numeric']);b.update(H=h,zoo=z,noise=noise);blocks.append(b)
    pr=sum(x['pilot'] for x in metadata.values());fr=sum(x['failed'] for x in metadata.values())
    assert d['pilot_reuses']==pr and d.get('failed_attempt_reuses',0)==fr
    assert d['source_calls_started']==d['source_calls_completed']==2*(64-pr-fr)
    total_cpu+=dec(d['cpu_seconds']);calls+=d['source_calls_completed'];reuse+=pr;failreuse+=fr
    numeric+=sum(r['named_numeric'] for r in d['rows']);integer+=sum(r['integer_control'] for r in d['rows'])
    paths.append({'path':str(path),'sha256':sha(path),'archive_sha256':d['archive_sha256']})
   g=group(selected,4*h*h+12*h-1);g.update(H=h,zoo=z);groups.append(g)
 assert len(paths)==60 and len(allrows)==3840 and calls==7670 and reuse==3 and failreuse==2
 assert len({(r['H'],r['zoo'],r['noise'],r['copy']) for r in allrows})==len({r['model_sha256'] for r in allrows})==3840
 assert sum(g['individual_losses']+g['individual_ties'] for g in groups)==len(losses)
 actual=total_cpu+dec(failed['cpu_seconds'])+dec(pilots['cpu_seconds'])+dec(prov['cpu_seconds'])
 charged=total_cpu+D(failed['reservation_cpu'])+dec(pilots['cpu_seconds'])+dec(prov['cpu_seconds'])
 result={'complete':True,'groups':groups,'noise_blocks':blocks,'individual_losses_or_ties':losses,'scalar_native_topology_differences':topology,'files':paths,
  'rows':3840,'new_calls_in_completed_shards':calls,'saved_failed_attempt_calls':4,'panel_calls_total':calls+4,'source_calls_total_including_pilots':calls+10,
  'pilot_reuses':reuse,'failed_attempt_case_reuses':failreuse,'named_numeric_total':numeric,'integer_control_total':integer,
  'completed_shard_cpu_seconds':str(total_cpu),'failed_attempt_actual_cpu_seconds':str(dec(failed['cpu_seconds'])),'failed_attempt_charged_reserve':failed['reservation_cpu'],
  'provenance_pilots_panel_measured_cpu_seconds':str(actual),'research_cap_charged_cpu_seconds':str(charged),
  'worst_group_ratio':str(max(D(g['ratio_per_model_floor']) for g in groups)),
  'largest_group_bootstrap95_upper':max(g['bootstrap95_per_model_floor'][1] for g in groups),
  'scope':'Original fixed noisy panel and saved numerical truths. 80-digit Decimal contractions, conditional paired-row bootstrap, named source convention; no full native/library/hardware/truth-uncertainty/generalization claim. Semantic archive metadata conserved; full site-cost audit independent.',
  'counter_provenance':'Original fresh.py saves scalar R.rec112 topology but native E75 estimate/ops. Historical scalar/native equality is not assumed; all current native baseline/traced metadata remain equal.',
  'allocation_note':'Per-model floors use each cost; uniform floor uses floor(mean cost/sampler). Continuous budgets and historical proxy means are separate.',
  'summary_gate_sha256':sha(HERE/'summary_recovery_gate.json'),'cpu_seconds':time.process_time()-start}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('groups','noise_blocks','individual_losses_or_ties','scalar_native_topology_differences','files')}))
if __name__=='__main__':main()
