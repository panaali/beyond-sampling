from pathlib import Path
from fractions import Fraction
import json,time
from signed_rank import integrate
from controls import coefficient

def exact0(T,signs):
    m=T-4;p=2+signs.count('+');neg=4-p
    A=coefficient(m-1,p)/2**neg if m else Fraction(0)
    B=coefficient(m,p-1)/2**neg
    assert m*A+p*B==Fraction(1,16)
    return [float(A),float(B),float(B) if signs[0]=='+' else 0.,float(B) if signs[1]=='+' else 0.]

def main():
    start=time.process_time();rows=[]
    for T in (4,6,10):
        for k in (0.,.25,1.,4.):
            for signs in ('++','+-','-+','--'):
                for outer,inner in ((64,32),(128,32),(64,64),(128,64)):
                    r=integrate(T,k,signs,outer,inner)
                    assert min(r['classes'])>=0
                    if k==0:
                        r['exact_classes']=exact0(T,signs);r['max_exact_error']=max(abs(a-b) for a,b in zip(r['classes'],r['exact_classes']))
                        assert r['max_exact_error']<1e-13
                    rows.append(r)
    out={'rows':rows,'cpu_seconds':time.process_time()-start,'scope':'fixed signed rank controls, no trained result','sampled_inputs':0}
    Path(__file__).with_name('signed_controls_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':out['cpu_seconds'],'max_partition_error':max(abs(r['partition_error']) for r in rows),'max_exact_error':max(r.get('max_exact_error',0) for r in rows),'max_partition128_64':max(abs(r['partition_error']) for r in rows if r['outer']==128 and r['inner']==64)},indent=2))
if __name__=='__main__':main()
