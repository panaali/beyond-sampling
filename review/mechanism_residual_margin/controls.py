"""Fixed ten analytic rows and four guards. Requires published source gate."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,resource,sys,time,traceback
H=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cases():
    # h, hh, truth; identity readout on first three coordinates.
    specs=[('both_correct',[3,1,0],[4,1,0],0,0),
      ('erased',[1,2,0],[3,2,0],0,1),
      ('introduced',[3,1,0],[0,2,0],0,2),
      ('wrong_same',[1,3,0],[0,4,0],0,3),
      ('wrong_different',[1,3,0],[0,1,4],0,4),
      ('first_index_correct',[2,2,0],[2,2,0],0,0),
      ('first_index_wrong',[2,2,0],[2,3,0],1,1),
      ('positive_pair_not_correct',[1,3,0],[4,3,5],0,4),
      ('zero_sum',[0,0,0],[0,0,0],1,3)]
    out=[]
    for name,h,hh,y,t in specs:
        w=[[int(i==j) for j in range(16)] for i in range(3)]
        out.append(dict(name=name,h=h+[0]*13,hh=hh+[0]*13,logits=h,lookup_logits=hh,wo=w,truth=y,transition=t))
    out.append(dict(name='ordered_cancellation',h=[0]*16,hh=[2**53,1,-2**53]+[0]*13,
       logits=[0,0,-1],lookup_logits=[1,0,-1],wo=[[1,1,1]+[0]*13,[0]*16,[-1,-1,-1]+[0]*13],truth=0,transition=0))
    # The last case supplies deliberate stored-logit/term-order discrepancy.
    return out

def main():
    result_path=H/'controls_results.json';progress=H/'controls_progress.jsonl'
    assert not result_path.exists() and not progress.exists(),'no overwrite or replay'
    resource.setrlimit(resource.RLIMIT_CPU,(19,20))
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    start=time.process_time();r=dict(complete=False,rows=[],guards=[],source_calls=0)
    def emit(stage,**kw):
        with progress.open('a') as f:f.write(json.dumps(dict(stage=stage,**kw),allow_nan=False)+'\n');f.flush()
    def save():
        r['cpu_seconds']=time.process_time()-start;r['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        p=H/'controls_results.tmp';p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');p.replace(result_path)
    try:
        p=json.loads((H/'controls_protocol.json').read_text());g=json.loads((H/'controls_gate.json').read_text())
        assert g['passed'] and g['published_commit']!='pending'
        assert g['protocol_sha256']==sha(H/'controls_protocol.json')
        for path,digest in p['pins'].items():assert sha(path)==digest,path
        assert str(Path(sys.executable).resolve())==p['python']
        r.update(protocol_sha256=sha(H/'controls_protocol.json'),gate_sha256=sha(H/'controls_gate.json'))
        emit('started');save()
        import numpy as np
        from primitive import analyze,DomainError
        assert np.__version__==p['numpy_version']
        cs=cases();assert len(cs)==10
        for c in cs:
            emit('case_attempt',name=c['name']);save();r['source_calls']+=1
            ans=analyze([c['h']],[c['hh']],[c['logits']],[c['lookup_logits']],c['wo'],[c['truth']])
            cidx=max((j for j in range(3) if j!=c['truth']),key=lambda j:(c['logits'][j],-j))
            exact_terms=[(F(b)-F(a))*(F(c['wo'][c['truth']][j])-F(c['wo'][cidx][j])) for j,(a,b) in enumerate(zip(c['h'],c['hh'],strict=True))]
            encoded={k:v.tolist() if hasattr(v,'tolist') else v for k,v in ans.items()}
            row=dict(fixture=c,output=encoded,exact_terms=[str(x) for x in exact_terms],exact_sum=str(sum(exact_terms)))
            r['rows'].append(row);emit('case_result',name=c['name']);save()
            assert int(ans['transition'][0])==c['transition'] and int(ans['competitor'][0])==cidx
            assert [F(float(v)) for v in ans['terms'][0]]==exact_terms
            if c['name']=='ordered_cancellation':
                assert sum(exact_terms)==1 and ans['term_sum'][0]==0 and ans['closure_residual'][0]==1
            else:assert ans['closure_residual'][0]==0 and F(float(ans['term_sum'][0]))==sum(exact_terms)
        base=cs[0]
        args=[[base['h']],[base['hh']],[base['logits']],[base['lookup_logits']],base['wo'],[0]]
        for name,k,val in [('hidden_shape',1,[[0]]),('truth_float',5,[0.0]),('truth_range',5,[3]),('nonfinite',2,[[float('inf'),0,0]])]:
            emit('guard_attempt',name=name);r['source_calls']+=1;save()
            altered=args.copy();altered[k]=val
            try:analyze(*altered)
            except DomainError as e:r['guards'].append(dict(name=name,rejected=True,error=str(e)))
            else:raise AssertionError('guard accepted: '+name)
            emit('guard_result',name=name);save()
        assert len(r['rows'])==10 and len(r['guards'])==4 and r['source_calls']==14
        r['complete']=True;save();emit('completed',results_sha256=sha(result_path))
    except BaseException:
        r['failure']=traceback.format_exc();save();emit('failure');raise

if __name__=='__main__':main()
