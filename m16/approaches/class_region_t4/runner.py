"""Pinned T4 family comparison with exact partial-progress and cost reporting."""
import time
IMPORT_START=time.process_time()
import argparse,hashlib,json,resource,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).parent
SOURCE=Path('/workspace/clean-neurons/m16/approaches/class_region_families')
HASHES={'families.py':'e1c93ac9096bcb37f9f9f39bbb79b1c2b1d65314000a3029dc34919768ac6a6a','tie_repair.py':'11eaa7354fb864a3d880727dcbf8b53fa5fa4cc22c5c0c7fd3138d78f81a4de8','estT_snapshot.py':'0a2db2eb14e37e0d6575630c15b145d41cdedba5e977e603662576fd3700c5cb'}
for name,h in HASHES.items():assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==h
sys.path.insert(0,str(SOURCE))
from families import engine,quadratic,nearest
from tie_repair import activate
activate()
IMPORT_CPU=time.process_time()-IMPORT_START
CHECKPOINT=Path('/workspace/clean-neurons/work/checkpoints/H4_T4_seed0.pth')
CHECKHASH='83ddefeaee737c73613cdc786b01b3bb79c0007bd6c1e42a454302c4801dd146'
REFERENCE=.477013125994119

class Limit(RuntimeError):pass
class Ops:
    def __init__(self,cpu=50,cap=200000000):self.n=0;self.start=time.process_time();self.cpu=cpu;self.cap=cap
    def add(self,n):
        self.n+=n
        if self.n>self.cap:raise Limit('Named-operation cap reached')
        if time.process_time()-self.start>self.cpu:raise Limit('CPU cap reached')

def families(kind,ops):
    return engine.halves(4) if kind=='old' else (quadratic if kind=='quadratic' else nearest)(4,ops)

def partial_run(model,kind,mode,output=None,cpu=50):
    start=time.process_time();ops=Ops(cpu);rows=[];error=0.;hs=[];setup=None
    out=dict(kind=kind,mode=mode,complete=False,source_sha256=HASHES,checkpoint_sha256=CHECKHASH,import_cpu=IMPORT_CPU,partial_named_costs=True,portable_cost_certified=False,mass_roundoff_certified=False)
    def save():
        out.update(completed_families=len(rows),partial_completed_error=error,rows=rows,named_ops=ops.n,stats=engine.STATS.copy(),cpu_seconds=time.process_time()-start,maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if output:output.write_text(json.dumps(out,indent=2)+'\n')
    try:
        setup_start=time.process_time();hs=families(kind,ops);out.update(families=len(hs),setup_cpu=time.process_time()-setup_start,setup_ops=None if kind=='old' else ops.n,old_setup_ops_unknown=kind=='old')
        engine.STATS.update({k:0 for k in engine.STATS});engine.MODE[0]=mode
        wi,wh,wo=model;eye=np.eye(4)
        for i,h in enumerate(hs):
            old=ops.n;ct=time.process_time();stats=engine.STATS.copy()
            g=(h['sigma']*wi>0).astype(float);ops.add(len(wi));A=np.outer(wi,eye[0])
            value=engine._finish(h['cone'],g,A,1,model,h['truth'],h['n_true'],.1,ops,mode=='exact')
            error+=float(value);rows.append(dict(index=i,truth=int(h['truth']),sigma=int(h['sigma']),error=float(value),ops=ops.n-old,cpu=time.process_time()-ct,stats={k:engine.STATS[k]-stats[k] for k in stats}))
            save()
        out.update(complete=True,estimate=1-error,reference=REFERENCE,signed_reference_error=1-error-REFERENCE)
    except Exception as exc:
        out.update(failure_type=type(exc).__name__,failure=str(exc),cost_is_partial=True,unfinished_family_index=len(rows),no_partial_accuracy_reported=True)
    save();return out

def controls():
    start=time.process_time();out=dict(rows=[],analytic_reference=.25)
    for kind in ('old','quadratic','nearest'):
        ops=Ops(15);hs=families(kind,ops);assert len(hs)=={'old':96,'quadratic':24,'nearest':72}[kind]
        mass=[sum(engine.mass(h['cone'],ops) for h in hs if h['truth']==j) for j in range(4)]
        assert max(abs(v-.25) for v in mass)<1e-8
        m=(np.array([1.]),np.zeros((1,1)),np.zeros((4,1)))
        run=partial_run(m,kind,'exact',cpu=15)
        if not run['complete'] or abs(run.get('estimate',0)-.25)>=1e-8:
            out['failed_control']=run;out['cpu_seconds']=time.process_time()-start
            (HERE/'initial_control_failure.json').write_text(json.dumps(out,indent=2)+'\n')
            raise AssertionError('Constant-label control failed; preserved full partial ledger')
        out['rows'].append(dict(kind=kind,families=len(hs),class_masses=mass,total_mass=sum(mass),constant_label_estimate=run['estimate'],named_ops=ops.n+run['named_ops']))
    out.update(complete=True,cpu_seconds=time.process_time()-start,import_cpu=IMPORT_CPU)
    (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--controls',action='store_true');parser.add_argument('--kind',choices=['old','quadratic','nearest']);parser.add_argument('--mode',choices=['exact','recipe']);a=parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU,(55,57))
    if a.controls:return controls()
    assert a.kind and a.mode and hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==CHECKHASH
    start=time.process_time();state=torch.load(CHECKPOINT,weights_only=True,map_location='cpu');model=(state['rnn.weight_ih_l0'].numpy()[:,0].astype(float),state['rnn.weight_hh_l0'].numpy().astype(float),state['linear.weight'].numpy().astype(float));load=time.process_time()-start
    output=HERE/(a.kind+'_'+a.mode+'.json');out=partial_run(model,a.kind,a.mode,output);out['checkpoint_load_cpu']=load;output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
