"""Saved-only independent synthetic-control audit; no author imports or solves."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,subprocess,time,traceback
H=Path(__file__).resolve().parent
A=Path('/workspace/work/recover-small_h2_78/review/mechanism_claims_review/rebuild')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    start=time.process_time(); pins=read(H/'pins.json')
    for p,v in pins.items():assert sha(p)==v,p
    gate=read(A/'controls_gate.json')
    for p,v in gate['pins'].items():assert sha(p)==v,p
    raw=read(A/'controls_results.json'); progress=[json.loads(s) for s in (A/'controls_progress.jsonl').read_text().splitlines()]
    assert raw['complete'] and raw['gate_sha256']==sha(A/'controls_gate.json')
    repo=A.parents[2]
    for n in ['controls_results.json','controls_progress.jsonl','run2.log']:
        assert subprocess.check_output(['git','show','39b10901:review/mechanism_claims_review/rebuild/'+n],cwd=repo)==(A/n).read_bytes()
    for c in ['3fe1d31c','a9e5a3d5','ecb87789','e0281c91','b443c326']:
        subprocess.run(['git','merge-base','--is-ancestor',c,'732867a4'],cwd=repo,check=True)
    assert read(A/'launcher_failure.json')['candidate_calls']==0
    assert '/usr/bin/time: No such file or directory' in (A/'run1.log').read_text()
    assert len(progress)==9 and progress[0]==dict(stage='started',gate_sha256=sha(A/'controls_gate.json'))
    assert progress[-1]==dict(stage='completed',sha256=sha(A/'controls_results.json'))
    names=['state_coverage','lookup_known_map','separable_known_map','domain_guards','metric_zero_variance_and_output_ties','disjoint_cross_tab','six_dyadic_recurrence_inputs']
    rows=raw['rows'];assert [r['name'] for r in rows]==names
    assert [p['stage'] for p in progress[1:-1]]==names
    legal=[(c*9+p)*9+q for c in range(3) for p in range(9) for q in range(9) if p!=q]
    assert rows[0]==dict(name=names[0],rows=864,legal_states=216,illegal_codes=27,keys=[k for k in legal for _ in range(4)])
    r=rows[1];m=r['meta'];assert m['counts']==m['ranks']==[3]*216 and m['rcond']==1e-12
    errors=r['coefficient_max_errors'];assert len(errors)==216 and all(math.isfinite(e) and 0<=e<1e-10 for e in errors)
    assert 0<=r['heldout_max_error']<1e-10
    triples=[[(3,2,4),(4,2,5),(3,1,4)],[(4,1,2),(5,1,2),(4,1,3)],[(3,2,1),(4,2,1),(3,1,0)]]
    for i,sv in enumerate(m['singular_values']):
        X=triples[i//72];assert len(sv)==3 and all(math.isfinite(v) and v>0 for v in sv)
        assert sv==sorted(sv,reverse=True)
        assert abs(sum(v*v for v in sv)-sum(v*v for row in X for v in row))<1e-10
        det=X[0][0]*(X[1][1]*X[2][2]-X[1][2]*X[2][1])-X[0][1]*(X[1][0]*X[2][2]-X[1][2]*X[2][0])+X[0][2]*(X[1][0]*X[2][1]-X[1][1]*X[2][0])
        assert abs(math.prod(sv)-abs(det))<1e-10
    assert len(m['singular_values'])==216
    r=rows[2];m=r['meta'];assert (r['heldout_rows'],m['count'],m['rank'],m['rcond'])==(72,216,18,1e-12)
    assert all(math.isfinite(r[k]) and 0<=r[k]<1e-10 for k in ['coefficient_max_error','heldout_max_error'])
    sv=m['singular_values'];assert len(sv)==18 and all(math.isfinite(v) and v>0 for v in sv)
    # Three case2 feature pairs: (m2,gap)=(2,1),(2,2),(1,2), 72 ordered positions.
    assert abs(sum(v*v for v in sv)-72*(5+8+5))<1e-9
    expected=[('input_tie','input_tie_preserve_and_stop_no_resampling'),('input_nonfinite','finite_nonempty_N_by_10_required'),('lookup_underfilled','lookup_state_underfilled'),('lookup_deficient','lookup_rank_deficient:0'),('separable_underfilled','separable_underfilled'),('separable_deficient','separable_rank_deficient'),('bad_prediction','integer_predictions_0_to_9_required')]
    assert rows[3]['cases']==[dict(name=n,reason=r) for n,r in expected]
    r=rows[4];assert r['metrics']==dict(n=2,sse=[4.,2.],sst=[2.,0.],mse=[2.,1.],r2=[-1.,None],median_r2=-1.,zero_variance_columns=[1])
    assert r['all_constant']==dict(n=2,sse=[0.,0.],sst=[0.,0.],mse=[0.,0.],r2=[None,None],median_r2=None,zero_variance_columns=[0,1])
    assert r['first_tie_winners']==[0,0]
    r=rows[5];tab=[[[0]*10 for _ in range(3)] for _ in range(3)];correct=0;ranks=[]
    assert len(r['inputs'])==len(r['predicted_labels'])==9
    for x,p in zip(r['inputs'],r['predicted_labels'],strict=True):
        assert len(x)==10 and len(set(x))==10
        order=sorted(range(10),key=lambda j:(-x[j],j));head=sorted(range(9),key=lambda j:(-x[j],j))
        case=0 if x[9]>x[head[0]] else 1 if x[9]>x[head[1]] else 2
        rank=order.index(p)+1;ranks.append(rank);tab[min(sum(v>0 for v in x),2)][case][rank-1]+=1;correct+=p==order[1]
    assert ranks==r['expected_ranks']==[1,2,3,4,5,6,7,8,10]
    assert r['result']==dict(n=9,correct=correct,table=tab) and correct==1
    r=rows[6];assert r['Wi']==[1.,-1.] and r['Wh']==[[.5,-.25],[.25,.5]]
    assert len(r['inputs'])==len(r['outputs'])==len(r['exact_output_fractions'])==6
    for seq,out,text in zip(r['inputs'],r['outputs'],r['exact_output_fractions'],strict=True):
        assert len(seq)==10;h=[F(0),F(0)]
        for x in seq:
            old=h;h=[max(F(0),sum(F(w)*v for w,v in zip(row,old,strict=True))+F(wi)*F(x)) for row,wi in zip(r['Wh'],r['Wi'],strict=True)]
        assert [str(v) for v in h]==text and all(F(v)==t for v,t in zip(out,h,strict=True))
    assert raw['actual_call_ledger']==dict(lstsq=219,forward_batches=1,forward_sequences=6)
    assert (raw['lstsq_calls'],raw['synthetic_forward_batch_calls'],raw['synthetic_forward_sequences'],raw['actual_checkpoint_model_calls'],raw['new_random_inputs'])==(219,1,6,0,0)
    assert math.isfinite(raw['cpu_seconds']) and raw['cpu_seconds']>=0
    return dict(complete=True,source_pins=pins,groups=7,legal_states=216,synthetic_exact_scalar_sequences_checked=6,author_imports=0,new_lstsq=0,new_author_forwards=0,author_cpu=raw['cpu_seconds'],review_cpu=time.process_time()-start,limitations=['Saved coefficients/prediction arrays are not serialized; coefficient and heldout residual maxima are source-gated author summaries, with saved bounds/meta/invariants checked here rather than recomputed fits.','Synthetic controls do not establish checkpoint wiring, heldout trained performance, historical provenance or challenge success.'])
if __name__=='__main__':
    assert not (H/'result.json').exists() and not (H/'failure.txt').exists()
    try: result=main()
    except BaseException:
        with (H/'failure.txt').open('x') as f:f.write(traceback.format_exc())
        raise
    with (H/'result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_pins'}))
