"""Fixed analytic tie controls. No checkpoint or stochastic inputs."""
from pathlib import Path
import hashlib,json,time,traceback,warnings,math
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def main():
    assert not any((H/n).exists() for n in ['results.json','progress.jsonl','failure.json']) and not list(H.glob('row_*.json'))
    gate=read(H/'gate.json')
    for p,h in gate['pins'].items():assert sha(p)==h,p
    assert gate['complete'] is True
    for name in ['t204_saved_review','t210_source_review']:
        r=gate[name];assert r['complete'] is True and r['commit'] and r['shared_commit']
        assert gate['pins'][r['artifact']]==r['sha256']==sha(r['artifact'])
    prior=read(gate['t204_saved_review']['artifact']);assert prior['complete'] is True and prior['pass_all'] is True
    proto=read(H/'protocol.json');assert proto['original_saved_review']=='40ea5646'
    priorpins=read(proto['original_review_pins']);assert sha(proto['original_review_pins'])==prior['pins_sha256']
    assert priorpins['pins'][proto['original_results']]==sha(proto['original_results'])
    start=time.process_time();rows=[];calls={'original_Q_batches':0,'repaired_Q_batches':0,'original_conditional_prefixes':0,'repaired_conditional_prefixes':0}
    def emit(stage,**d):
        with (H/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,**d))+'\n');f.flush()
    emit('started',gate_sha256=sha(H/'gate.json'))
    try:
        import numpy as np,mpmath as mp
        from adapter import load
        mp.mp.dps=80;original,repaired,meta=load();fixtures=read(H/'fixtures.json');proto=read(H/'protocol.json')
        inherited=read(proto['original_results'])['rows'][0];oldfixture=read(proto['original_fixtures'])[0]
        assert inherited['name']==oldfixture['name']==fixtures[0]['name']
        assert all(fixtures[0][k]==oldfixture[k] for k in ['Wi','Wh','Wo','x'])
        def cdf(y,radial):
            if mp.isinf(y):return mp.mpf(0) if y<0 else mp.mpf(1)
            if not radial:return (1+mp.erf(y/mp.sqrt(2)))/2
            # F_t9(3y), density128/(35*pi)*(1+y²)^-5.
            integral=mp.atan(y)
            for n in range(2,6):integral=y/(2*(n-1)*(1+y*y)**(n-1))+(mp.mpf(2*n-3)/(2*(n-1)))*integral
            return mp.mpf('.5')+128*integral/(35*mp.pi)
        def reference(f,x,radial):
            x=list(map(mp.mpf,x))
            if radial:
                norm=mp.sqrt(sum(v*v for v in x));x=[v/norm for v in x]
            order=sorted(range(9),key=lambda j:(-x[j],j));q1,q2=order[:2];m1,m2=x[q1],x[q2]
            edges=sorted(set([-mp.inf,mp.mpf(0),m1,m2,mp.inf]));total=mp.mpf(0);pieces=[]
            for lo,hi in zip(edges[:-1],edges[1:],strict=True):
                y=hi-1 if mp.isinf(lo) else lo+1 if mp.isinf(hi) else (lo+hi)/2
                target=q1 if y>m1 else 9 if y>m2 else q2
                kind=f['prediction'];pred=int(kind[-1]) if kind.startswith('constant') else (1 if y<0 else 0) if kind=='negative1_positive0' else (0 if y<0 else 5)
                if pred==target:
                    mass=cdf(hi,radial)-cdf(lo,radial);total+=mass;pieces.append([str(lo),str(hi),pred,str(mass)])
            return dict(value=str(total),pieces=pieces)
        def run(fn,label,model,x):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');out=fn(model,x,radial=radial)
            calls[label+'_Q_batches']+=1;calls[label+'_conditional_prefixes']+=len(x)
            return dict(values=out.tolist(),hex=[float(v).hex() for v in out],warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught])
        for f in fixtures:
            model=tuple(np.array(f[k],dtype=float) for k in ['Wi','Wh','Wo']);x=np.array(f['x'],dtype=float)
            assert x.shape==(2,9) and all(len(set(row))==9 for row in f['x'])
            for radial in [False,True]:
                emit('case_started',name=f['name'],radial=radial)
                if f['reuse_original']:
                    # Only Gaussian baseline exists; radial original is not replayed or invented.
                    baseline=None if radial else dict(values=[float.fromhex(v) for v in inherited['runs'][0]['original']['hex']],hex=inherited['runs'][0]['original']['hex'],warnings=inherited['runs'][0]['original_warnings'],reused=True)
                else:baseline=run(original,'original',model,x)
                patched=run(repaired,'repaired',model,x);refs=[reference(f,row,radial) for row in f['x']]
                r=dict(name=f['name'],radial=radial,model={k:f[k] for k in ['Wi','Wh','Wo']},prefixes=f['x'],original=baseline,repaired=patched,reference=refs)
                rows.append(r);save(H/f"row_{len(rows)-1:02d}.json",r)
                assert all(math.isfinite(v) and abs(mp.mpf(v)-mp.mpf(ref['value']))<=mp.mpf('3e-13') for v,ref in zip(patched['values'],refs,strict=True))
                if f['bit_parity_expected']:assert baseline['hex']==patched['hex']
                emit('case_completed',name=f['name'],radial=radial)
        assert len(rows)==12 and calls==dict(original_Q_batches=10,repaired_Q_batches=12,original_conditional_prefixes=20,repaired_conditional_prefixes=24)
        out=dict(complete=True,rows=rows,calls=calls,reused_original_Gaussian_prefixes=2,reference_precision=80,source=meta,gate_sha256=sha(H/'gate.json'),protocol_sha256=sha(H/'protocol.json'),cpu_seconds=time.process_time()-start,new_random_checkpoint_trained_calls=0)
        save(H/'results.json',out);emit('completed',sha256=sha(H/'results.json'));print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
    except BaseException:
        save(H/'failure.json',dict(traceback=traceback.format_exc(),completed_rows=len(rows),calls=calls));emit('failed',rows=len(rows));raise
if __name__=='__main__':main()
