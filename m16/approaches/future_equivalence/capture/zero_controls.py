"""Two synthetic zero models; universal initial cones retained once."""
from pathlib import Path
import json,time,hashlib
import numpy as np
from common import H,load_engine,Observer,cone_pack

def main():
    out=H/'zero_controls.json';progress=H/'zero_progress.jsonl';assert not out.exists() and not progress.exists()
    def record(r):
        with progress.open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
    record(dict(stage='source_import_attempt'));p=json.loads((H/'protocol.json').read_text());start=time.process_time();E=load_engine(p);import_cpu=time.process_time()-start;O=Observer(E,record);all_halves={};rows=[]
    for T,count in [(3,24),(4,96)]:
        record(dict(stage='initial_cones_attempt',T=T));before=O.proxy;halves=E.halves(T);assert len(halves)==count
        saved=[dict(sigma=h['sigma'],perm=list(h['perm']),truth=int(h['truth']),n_true=[float(x).hex() for x in h['n_true']],cone=cone_pack(h['cone'])) for h in halves];all_halves[str(T)]=saved
        record(dict(stage='initial_cones_result',T=T,count=count,source_proxy=O.proxy-before));model=(np.zeros(4),np.zeros((4,4)),np.zeros((T,4)))
        begin=len(O.states);parentrows=[O.run_parent(i,h,model) for i,h in enumerate(halves)];states=O.states[begin:]
        checks=dict(one_state_per_chart=len(states)==count and all(r['states']==1 for r in parentrows),zero_states=all(all(float.fromhex(x)==0 for r in s['state'] for x in r) for s in states),zero_histories=all(not any(g) for s in states for g in s['history']),depth=all(s['depth']==T-1 and len(s['history'])==T-1 for s in states),first_index_truth='all future margins exactly zero => label0; truth remains per-chart')
        row=dict(T=T,checks=checks,parents=parentrows,passed=all(v for v in checks.values() if isinstance(v,bool)));rows.append(row);record(dict(stage='model_result',row=row));assert row['passed']
    result=dict(complete=True,rows=rows,all_pass=all(r['passed'] for r in rows),initial_cones=all_halves,observer=O.summary(),source_import_setup_cpu=import_cpu,cpu_seconds=time.process_time()-start,new_synthetic_models=2,new_trained_capture_reference_MC_calls=0,scope=p['scope'])
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(all_pass=result['all_pass'],observer=result['observer'],cpu_seconds=result['cpu_seconds'])))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (H/'zero_failures.jsonl').open('a') as f:f.write(json.dumps(dict(type=type(e).__name__,message=str(e)))+'\n')
        raise
