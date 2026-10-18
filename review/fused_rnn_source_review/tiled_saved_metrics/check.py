"""Independent saved-summary audit only; no author/tensor imports or replay."""
import hashlib,json,math,resource,time,traceback
from pathlib import Path
from fractions import Fraction
P=Path(__file__).resolve().parent

def main():
    out=P/'results.json';assert not out.exists()
    t=time.process_time();r={'complete':False,'pass_all':False,'rows':[],'tensor_reconstruction':False,'author_calls':0}
    def save():
        r['cpu_seconds']=time.process_time()-t
        tmp=P/'results.tmp';tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(out)
    try:
        scope=json.loads((P/'scope.json').read_text())
        for f,h in scope['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
        r['limits']={'cpu':resource.getrlimit(resource.RLIMIT_CPU),'address_space':resource.getrlimit(resource.RLIMIT_AS)};save()
        assert r['limits']=={'cpu':(20,20),'address_space':(2**31,2**31)}
        d=json.loads(Path(scope['input']).read_text());assert d['complete'] is True and d['pass_all'] is True
        expected=[('initial',0,3,32,0),('mixed_active',0,3,32,5),('odd_batch',0,2,13,3),('zero_hidden',0,2,16,1),('sequence',0,2,17,0),('sequence',1,2,17,1),('sequence',2,2,17,2),('production_batch',0,1,1024,7)]
        assert len(d['rows'])==len(expected)==8
        maxabs=Fraction();maxrel=Fraction();count=0;zeros=0
        for a,(name,it,k,b,step) in zip(d['rows'],expected,strict=True):
            assert (a['name'],a['iteration'],a['K'],a['B'])==(name,it,k,b)
            assert a['step']==[float(step)]*k and a['status']=='passed'
            assert all(a[key] is True for key in ['accuracy_equal','steps_equal','inactive_bitwise'])
            assert set(a['comparisons'])=={'params','m','v','grad','logits','loss'}
            rr={'name':name,'iteration':it,'records':0,'zero_reference_records':0}
            for family,records in a['comparisons'].items():
                assert len(records)==(3 if family in ('params','m','v','grad') else 1)
                for q in records:
                    assert set(q)=={'max_abs','relative_l2','reference_l2','finite','pass'}
                    assert q['finite'] is True and q['pass'] is True
                    assert all(isinstance(q[n],(int,float)) and not isinstance(q[n],bool) and math.isfinite(q[n]) and q[n]>=0 for n in ['max_abs','relative_l2','reference_l2'])
                    ab,rel,norm=(Fraction(q[n]) for n in ['max_abs','relative_l2','reference_l2'])
                    assert ab<=Fraction(2,10**6) and rel<=Fraction(1,10**6)
                    if norm==0:
                        assert ab==0 and rel==0;zeros+=1;rr['zero_reference_records']+=1
                    maxabs=max(maxabs,ab);maxrel=max(maxrel,rel);count+=1;rr['records']+=1
            assert rr['records']==14;r['rows'].append(rr);save()
        assert count==112
        r.update(metric_records=count,zero_reference_records=zeros,max_absolute=float(maxabs),max_relative_l2=float(maxrel),complete=True,pass_all=True,limitations=['Saved summary consistency only; underlying tensor errors and equality flags cannot be independently reconstructed without raw tensors.','Relative ratios are retained rounded metrics; original norm-product acceptance is not regenerated.','No GPU numerical, libdevice, graph lifecycle or performance evidence.'])
        save()
    except BaseException:
        r.update(complete=False,pass_all=False,failure=traceback.format_exc());save();raise
if __name__=='__main__':main()
