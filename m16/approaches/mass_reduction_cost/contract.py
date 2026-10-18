"""Replace matching inherited terms once; saved records only, no kernels."""
from pathlib import Path
import json,hashlib,time
HERE=Path(__file__).resolve().parent
def main():
 out=HERE/'summary.json';assert not out.exists();start=time.process_time();pins={}
 def read(name):
  p=HERE/name;pins[name]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
 prior=read('frozen/prior_summary.json');census=read('frozen/full_summary.json');controls=read('controls.json');assert controls['cases']==172 and controls['failures']==[]
 assert prior['source_sha256']=='b8d3baf0d85d47bf4a7a204f82cdfc648de580b47baea5f42a4c77c49beb906a'
 assert census['thin']==17955 and census['wide']==17143 and census['cases']==35098
 dispatch=[json.loads(s) for s in (HERE/'dispatch_offset0.jsonl').read_text().splitlines()]
 pins['dispatch_offset0.jsonl']=hashlib.sha256((HERE/'dispatch_offset0.jsonl').read_bytes()).hexdigest()
 assert dispatch[-1]['stage']=='complete'
 assert [r['legacy_candidate_offset'] for r in dispatch if r['stage']=='aux']==['0x5462e0']*2
 assert [r['offset'] for r in dispatch if r['stage']=='dot']==['0xa3d7c0']
 sites={
 'thin_row4':dict(branch='thin',multiplicity=1000,old_per=3,new_per=4),
 'thin_sum1000':dict(branch='thin',multiplicity=1,old_per=999,new_per=1000),
 'wide_norm4':dict(branch='wide',multiplicity=16,old_per=8,new_per=10),
 'wide_sum128':dict(branch='wide',multiplicity=6,old_per=127,new_per=128),
 'wide_python_sum6':dict(branch='wide',multiplicity=1,old_per=5,new_per=61)}
 for r in sites.values():
  r['copies']=census[r['branch']];r['removed']=r['copies']*r['multiplicity']*r['old_per'];r['replacement']=r['copies']*r['multiplicity']*r['new_per'];r['delta']=r['replacement']-r['removed']
 branches={}
 for name in ['thin','wide']:
  delta=sum(r['multiplicity']*(r['new_per']-r['old_per']) for r in sites.values() if r['branch']==name)
  branches[name]=dict(copies=census[name],prior=prior['branches'][name]['revised_hybrid'],delta=delta,new_partial_hybrid=prior['branches'][name]['revised_hybrid']+delta)
 delta=sum(r['delta'] for r in sites.values());total=prior['revised_hybrid_subtotal']+delta
 assert total==sum(r['copies']*r['new_partial_hybrid'] for r in branches.values())
 result=dict(sites=sites,branches=branches,prior_partial_hybrid=prior['revised_hybrid_subtotal'],removed_once=sum(r['removed'] for r in sites.values()),replacement=sum(r['replacement'] for r in sites.values()),delta=delta,revised_partial_hybrid=total,model_free_constants_unchanged=prior['model_free_runtime_constants_unchanged'],new_model_mass_geometry_calls=0,cpu_seconds=time.process_time()-start,pins=pins,complete_native_cost=False,sampling_ratio_not_computed=True,scope='Conditional frozen layout/dispatch native numeric subset replacing only listed terms. Python exact-float packed abs lanes charged; generic control/index/memory/casts/runtime and unpriced libm/elementwise/orientation-dot/geometry/cold remain. Current metadata is not proof of historical binding; no complete native upper/lower or budget-comparable MSE.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['delta','revised_partial_hybrid','complete_native_cost']}))
if __name__=='__main__':main()
