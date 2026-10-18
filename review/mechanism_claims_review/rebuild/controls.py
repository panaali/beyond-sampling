"""Exactly one fixed analytic-control run; no checkpoint or random inputs."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time,traceback,sys

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save_json(p,x):
    with p.open('x') as f: json.dump(x,f,indent=2,allow_nan=False); f.write('\n')

def main():
    assert not (HERE/'controls_results.json').exists() and not (HERE/'controls_progress.jsonl').exists()
    gate=json.loads((HERE/'controls_gate.json').read_text())
    for p,h in gate['pins'].items():assert sha(Path(p))==h,p
    start=time.process_time()
    import numpy as np
    import core as C
    def emit(stage,**data):
        with (HERE/'controls_progress.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage,**data),allow_nan=False)+'\n');f.flush()
    emit('started',gate_sha256=sha(HERE/'controls_gate.json'))
    rows=[]
    try:
        triples={0:[(3,2,4),(4,2,5),(3,1,4),(5,2,6)],
                 1:[(4,1,2),(5,1,2),(4,1,3),(6,1,4)],
                 2:[(3,2,1),(4,2,1),(3,1,0),(5,2,0)]}
        vectors=[]; keys=[]; train=[]; slots=[]
        for case in range(3):
            for p1 in range(9):
                for p2 in range(9):
                    if p1==p2:continue
                    key=(case*9+p1)*9+p2;slot=len(keys)//4
                    for j,(m1,m2,last) in enumerate(triples[case]):
                        x=np.zeros(10);x[p1]=m1;x[p2]=m2;x[9]=last
                        for a,k in enumerate(i for i in range(9) if i not in (p1,p2)):
                            x[k]=-a-1
                        vectors.append(x);keys.append(key);train.append(j<3);slots.append(slot)
        x=np.array(vectors); s=C.states(x); train=np.array(train)
        assert len(x)==864 and np.array_equal(s['key'],keys) and np.array_equal(s['slot'],slots)
        assert len(set(keys))==216 and set(np.flatnonzero(C.SLOT<0))=={(c*9+p)*9+p for c in range(3) for p in range(9)}
        rows.append(dict(name='state_coverage',rows=864,legal_states=216,illegal_codes=27,keys=keys))
        emit('state_coverage',rows=864)

        B=np.array([[[(k+1)*(j+1)+(h%5)-2 for h in range(16)] for j in range(3)] for k in range(216)],dtype=float)/16
        X=C.lookup_design(s)
        y=np.array([[sum(F(float(X[n,j]))*F(float(B[slots[n],j,h])) for j in range(3)) for h in range(16)] for n in range(864)],dtype=float)
        st={k:v[train] for k,v in s.items()}; se={k:v[~train] for k,v in s.items()}
        got,meta=C.fit_lookup(st,y[train]); predictions=C.predict_lookup(se,got)
        perstate=np.max(np.abs(got-B),axis=(1,2)); heldout=np.max(np.abs(predictions-y[~train]))
        assert np.max(perstate)<1e-10 and heldout<1e-10
        assert meta['counts']==[3]*216 and meta['ranks']==[3]*216
        rows.append(dict(name='lookup_known_map',coefficient_max_errors=perstate.tolist(),heldout_max_error=float(heldout),meta=meta))
        emit('lookup_known_map',heldout_rows=216,lstsq_calls=216)

        D=C.separable_design(s); Q=np.array([[(j+1)*(h%4-1) for h in range(16)] for j in range(18)],dtype=float)/8
        sy=D@Q; fitted,smeta=C.fit_separable(st,sy[train]);case2=(~train)&(s['case']==2)
        cerror=float(np.max(np.abs(fitted-Q)));perror=float(np.max(np.abs(D[case2]@fitted-sy[case2])))
        assert cerror<1e-10 and perror<1e-10 and smeta['count']==216 and smeta['rank']==18
        rows.append(dict(name='separable_known_map',coefficient_max_error=cerror,heldout_max_error=perror,heldout_rows=72,meta=smeta))
        emit('separable_known_map',heldout_rows=72,lstsq_calls=1)

        bad=[]
        def rejects(name,call,reason):
            try:call()
            except C.DomainError as e: assert str(e)==reason;bad.append(dict(name=name,reason=reason))
            else:raise AssertionError(name)
        rejects('input_tie',lambda:C.states(np.ones((1,10))),'input_tie_preserve_and_stop_no_resampling')
        rejects('input_nonfinite',lambda:C.states(np.full((1,10),np.inf)),'finite_nonempty_N_by_10_required')
        rejects('lookup_underfilled',lambda:C.fit_lookup(se,y[~train]),'lookup_state_underfilled')
        deficient={k:np.repeat(v[:1],3,axis=0) for k,v in s.items()}
        # Preserve all216 legal keys/counts, make only state0's three features identical.
        broken={k:v.copy() for k,v in st.items()}
        for name in ('m1','m2','x9'):broken[name][:3]=broken[name][0]
        rejects('lookup_deficient',lambda:C.fit_lookup(broken,y[train]),'lookup_rank_deficient:0')
        onlycase0={k:v[s['case']==0] for k,v in s.items()}
        rejects('separable_underfilled',lambda:C.fit_separable(onlycase0,sy[s['case']==0]),'separable_underfilled')
        repeats={k:np.repeat(v[s['case']==2][:1],18,axis=0) for k,v in s.items()}
        rejects('separable_deficient',lambda:C.fit_separable(repeats,np.ones((18,16))),'separable_rank_deficient')
        rejects('bad_prediction',lambda:C.cross_tab(se,np.full(len(se['key']),10,dtype=int)),'integer_predictions_0_to_9_required')
        rows.append(dict(name='domain_guards',cases=bad));emit('domain_guards',cases=len(bad))

        metrics=C.hidden_metrics(np.array([[1,2],[3,2]],float),np.array([[1,3],[1,1]],float))
        assert metrics==dict(n=2,sse=[4.,2.],sst=[2.,0.],mse=[2.,1.],r2=[-1.,None],median_r2=-1.,zero_variance_columns=[1])
        zeros=C.hidden_metrics(np.ones((2,2)),np.ones((2,2)))
        assert zeros['r2']==[None,None] and zeros['median_r2'] is None
        assert np.argmax(np.zeros((2,10)),axis=1).tolist()==[0,0]
        rows.append(dict(name='metric_zero_variance_and_output_ties',metrics=metrics,all_constant=zeros,first_tie_winners=[0,0]))
        emit('metric_zero_variance_and_output_ties')

        choices=[(-2,-3,-1),(-2,-3,F(-5,2)),(-2,-3,-4),
                 (-2,-3,1),(2,-3,-1),(2,-1,-2),
                 (3,2,4),(4,1,2),(3,2,1)]
        tx=[]
        for m1,m2,last in choices:tx.append([float(m1),float(m2),-5.,-6.,-7.,-8.,-9.,-10.,-11.,float(last)])
        ts=C.states(tx); wanted=[1,2,3,4,5,6,7,8,10]
        pred=np.array([int(np.flatnonzero(ts['global_rank'][i]==r)[0]) for i,r in enumerate(wanted)])
        tab=C.cross_tab(ts,pred); expected=np.zeros((3,3,10),dtype=int)
        for i,r in enumerate(wanted):expected[i//3,i%3,r-1]=1
        assert tab==dict(n=9,correct=1,table=expected.tolist())
        rows.append(dict(name='disjoint_cross_tab',inputs=tx,predicted_labels=pred.tolist(),expected_ranks=wanted,result=tab))
        emit('disjoint_cross_tab',rows=9)

        fx=np.array([[1.]*10,[-1.]*10,[(-1.)**j for j in range(10)],
                     [j/8-0.5 for j in range(10)],[0.]*10,[1024.,-1024.]*5])
        Wi=np.array([1.,-1.]);Wh=np.array([[.5,-.25],[.25,.5]])
        expected=[]
        for seq in fx:
            h=[F(0),F(0)]
            for val in seq:
                v=F(float(val));h=[max(F(0),h[0]/2-h[1]/4+v),max(F(0),h[0]/4+h[1]/2-v)]
            expected.append(list(map(float,h)))
        actual=C.forward_final(fx,Wi,Wh)
        assert np.array_equal(actual,np.array(expected))
        rows.append(dict(name='six_dyadic_recurrence_inputs',inputs=fx.tolist(),Wi=Wi.tolist(),Wh=Wh.tolist(),outputs=actual.tolist(),exact_output_fractions=[[str(F(v)) for v in row] for row in expected]))
        emit('six_dyadic_recurrence_inputs',synthetic_forward_calls=1,sequences=6)

        out=dict(complete=True,gate_sha256=sha(HERE/'controls_gate.json'),numpy_version=np.__version__,python=sys.version,
                 rows=rows,lstsq_calls=220,synthetic_forward_batch_calls=1,synthetic_forward_sequences=6,
                 actual_checkpoint_model_calls=0,new_random_inputs=0,cpu_seconds=time.process_time()-start,
                 scope='Synthetic helper controls only; no trained fit, historical validation, accuracy or cost claim.')
        save_json(HERE/'controls_results.json',out);emit('completed',sha256=sha(HERE/'controls_results.json'))
        print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
    except BaseException:
        save_json(HERE/'controls_failure.json',dict(traceback=traceback.format_exc(),completed_rows=rows,cpu_seconds=time.process_time()-start))
        emit('failed',completed=len(rows));raise

if __name__=='__main__':main()
