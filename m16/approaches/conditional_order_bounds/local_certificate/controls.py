"""Frozen synthetic local-certificate controls; execution requires published gate."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product,permutations
import hashlib,json,time,resource,traceback,sys
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def pack(v):
    if isinstance(v,F):return str(v)
    if isinstance(v,dict):return {k:pack(q) for k,q in v.items()}
    if isinstance(v,(list,tuple)):return [pack(q) for q in v]
    if hasattr(v,'lo') and hasattr(v,'hi'):return dict(lower=str(v.lo),upper=str(v.hi))
    return v

def fixture(name,wi,wh,wo,box,expected=None,D=1,stable=False):
    return dict(name=name,model=dict(wi=wi,wh=wh,wo=wo,D=D,H=len(wi),T=4),box=box,expected=expected,stable=stable)

def fixtures():
    signed=[[1,0],[0,1],[0,0],[-1,-1]];zero=[[0,0],[0,0]]
    return [
      fixture('zero_readout',[1],[[0]],[[0]]*4,[[-2,2]]*4,['correct','wrong','wrong','wrong']),
      fixture('identical_active',[1],[[0]],[[1]]*4,[[2,4]]*4,['correct','wrong','wrong','wrong'],stable=True),
      fixture('negative_final',[1,-1],zero,signed,[[-2,2]]*3+[[-4,-2]],['wrong','correct','wrong','wrong']),
      fixture('positive_final',[1,-1],zero,signed,[[-2,2]]*3+[[2,4]],['correct','wrong','wrong','wrong']),
      fixture('crossing_final',[1,-1],zero,signed,[[-2,2]]*4,['unknown','unknown','wrong','wrong']),
      fixture('signed_recurrence',[1,-1],[[1,-1],[-1,1]],signed,[[-2,2]]*4),
      fixture('dense_denominator',[1,-2,3],[[1,-1,2],[-2,1,1],[3,0,-1]],[[1,0,-1],[0,1,1],[-1,2,0],[2,-1,1]],[[-4,-2],[0,2],[-2,2],[2,4]],D=2),
      fixture('all_stable',[1,2],[[1,1],[0,1]],[[1,0],[0,1],[1,1],[-1,-1]],[[2,4]]*4,['wrong','wrong','correct','wrong'],stable=True),
      fixture('zero_input',[0],[[1]],[[0],[1],[-1],[0]],[[-2,2]]*4,['correct','wrong','wrong','wrong'],stable=True)]

def oracle(model,v):
    H,T,D=(model[k] for k in ('H','T','D'));h=[F(0)]*H;records=[]
    for t in range(T):
        z=[F(model['wi'][i],D)*v[t]/2+sum((F(model['wh'][i][j],D)*h[j] for j in range(H)),F(0)) for i in range(H)]
        h=[max(F(0),q) for q in z];records.append(dict(pre=z,hidden=h.copy()))
    logits=[sum((F(w,D)*a for w,a in zip(row,h,strict=True)),F(0)) for row in model['wo']]
    winner=max(range(T),key=lambda j:logits[j])
    return records,logits,winner

def main():
    out=HERE/'controls_results.json';progress=HERE/'controls_progress.jsonl'
    assert not out.exists() and not progress.exists() and not (HERE/'controls_results.tmp').exists();sys.set_int_max_str_digits(0)
    start=time.process_time();r=dict(complete=False,rows=[],candidate_certification_attempts=0,synthetic_exact_forwards=0,trained_forwards=0,checkpoint_loads=0,random_draws=0)
    def emit(stage,**kw):
        with progress.open('a') as f:f.write(json.dumps(pack(dict(stage=stage,**kw)),allow_nan=False)+'\n')
    def save():
        r['cpu_seconds']=time.process_time()-start;r['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        tmp=HERE/'controls_results.tmp';tmp.write_text(json.dumps(pack(r),indent=2,allow_nan=False)+'\n');tmp.replace(out)
    def cap():assert time.process_time()-start<15,'15CPU primitive soft cap; preserve failure'
    try:
        p=read(HERE/'controls_protocol.json');g=read(HERE/'controls_gate.json')
        assert g['complete'] and g['protocol_sha256']==sha(HERE/'controls_protocol.json')
        for path,h in {**p['pins'],**g['pins']}.items():assert sha(path)==h,path
        r.update(protocol_sha256=sha(HERE/'controls_protocol.json'),gate_sha256=sha(HERE/'controls_gate.json'));emit('started');save()
        from enclosure import Ledger,certify,remainder,target_group
        from probability import setup,crossover,accounting
        import mpmath as mp
        ledger=Ledger();fs=fixtures();assert [f['name'] for f in fs]==p['network_names']
        for f in fs:
            cap();emit('network_attempt',name=f['name']);r['candidate_certification_attempts']+=1
            row=dict(kind='network',fixture=f,certificate=None,exact_points=[],passed=False);r['rows'].append(row);save()
            cert=certify(f['model'],f['box'],ledger,retain_trace=True)
            row['certificate']=cert;r['enclosure_accounting']=ledger.report();save()
            if f['expected'] is not None:assert cert['statuses']==f['expected']
            if f['stable']:assert all(row['post_error']==[0,0] for step in cert['trace'] for row in step['rows'])
            pts=[list(map(F,q)) for q in product(*f['box'])]+[[F(a+b,2) for a,b in f['box']]];assert len(pts)==17
            refs=row['exact_points']
            for v in pts:
                ref=dict(v=v,complete=False);refs.append(ref);r['synthetic_exact_forwards']+=1;save()
                steps,logits,winner=oracle(f['model'],v)
                ref.update(steps=steps,logits=logits,winner=winner);save()
                for step,reference in zip(cert['trace'],steps,strict=True):
                    for row,z,h in zip(step['rows'],reference['pre'],reference['hidden'],strict=True):
                        for prefix,value,den in [('pre',z,step['pre_denominator']),('post',h,step['post_denominator'])]:
                            a=row[prefix+'_A'];lo,hi=row[prefix+'_error'];b0,b1=row[prefix+'_range'];base=sum((c*x for c,x in zip(a,v,strict=True)),F(0))
                            assert lo<=den*value-base<=hi and b0<=den*value<=b1
                for key,(lo,hi) in cert['margins'].items():
                    a,b=map(int,key.split(','));margin=(logits[a]-logits[b])*cert['margin_denominator'];assert lo<=margin<=hi
                for j,status in enumerate(cert['statuses']):
                    if status=='correct':assert winner==j
                    elif status=='wrong':assert winner!=j
                ref['complete']=True;save()
            row['passed']=True;emit('network_complete',name=f['name']);save()
        for L,U in [(-3,1),(-1,3),(-1,1)]:
            for width in [0,2,17]:
                cap();rs=[remainder(L,U,k) for k in range(3)];k=min(range(3),key=lambda j:(j*width+rs[j],j))
                samples=[F(L),F(L,2),F(0),F(U,2),F(U)]
                for a in range(3):
                    values=[2*max(F(0),z)-a*z for z in samples];assert min(values)==0 and max(values)==rs[a]
                r['rows'].append(dict(kind='remainder',L=L,U=U,error_width=width,remainders=rs,chosen=k,passed=True));emit('remainder_complete',L=L,U=U,width=width);save()
        assignments=[([0,0,0,0],[0,1,2,3]),([0,1,2,3],[2]),([3,3,2,1],[0,1]),([3,3,3,0],[0,1,2]),([3,2,2,1],[1,2]),([3,2,2,2],[1,2,3])]
        for ids,expected in assignments:
            group=target_group(ids);assert group==expected;counts=[0]*4;total=0
            for order in permutations(range(4)):
                if any(ids[order[i]]>ids[order[i+1]] for i in range(3)):continue
                counts[order[-2]]+=1;total+=1
            assert total>0 and all(F(counts[j],total)==(F(1,len(group)) if j in group else 0) for j in range(4))
            r['rows'].append(dict(kind='rank',bin_ids=ids,group=group,counts=counts,total=total,passed=True));emit('rank_complete',bin_ids=ids);save()
        cap();mass=setup();r['probability_setup']=mass
        with mp.workdps(120):
            for c in mass['cdf_rows']:
                q=c['exact'];ref=mp.erfc(-mp.mpf(c['v'])/(2*mp.sqrt(2)))/2
                lo=mp.mpf(q.lo.numerator)/q.lo.denominator;hi=mp.mpf(q.hi.numerator)/q.hi.denominator
                assert lo<=ref<=hi and q.hi-q.lo<F(1,10**40)
                r['rows'].append(dict(kind='cdf',v=c['v'],reference120=mp.nstr(ref,110),passed=True));emit('cdf_complete',v=c['v']);save()
        for w,n in [('0',None),('1/2',101),('1/4',401),('1',26)]:
            z=crossover(F(w),1)
            assert z.get('minimum_samples')==n
            r['rows'].append(dict(kind='crossover',width=w,result=z,passed=True));emit('crossover_complete',width=w);save()
        for label in ['zero_denominator','bad_box','boolean_dimension']:
            f=fixtures()[0];m=f['model'];b=f['box']
            if label=='zero_denominator':m['D']=0
            elif label=='bad_box':b[0]=[1,1]
            else:m['H']=True
            r['candidate_certification_attempts']+=1
            row=dict(kind='guard',name=label,passed=False);r['rows'].append(row);emit('guard_attempt',name=label);save()
            try:certify(m,b)
            except AssertionError:pass
            else:raise AssertionError('domain guard failed: '+label)
            row['passed']=True;emit('guard_complete',name=label);save()
        assert len(r['rows'])==44 and r['synthetic_exact_forwards']==153 and r['candidate_certification_attempts']==12
        cap();r.update(complete=True,enclosure_accounting=ledger.report(),probability_accounting=accounting());save();emit('completed',results_sha256=sha(out))
        print(json.dumps(dict(complete=True,rows=len(r['rows']),cpu=r['cpu_seconds'],synthetic_forwards=r['synthetic_exact_forwards'])))
    except BaseException as e:r.update(failure=repr(e),traceback=traceback.format_exc());save();emit('failed',error=repr(e));raise
if __name__=='__main__':main()
