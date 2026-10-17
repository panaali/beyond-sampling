from fractions import Fraction as F
from pathlib import Path
import json,time
from negative_rank import integrate
from two_pair_controls import one_above

def exact(T):
    m=T-4;negative=F(1,2**(m+4)*(m+3))
    Pplus=one_above(m-1,1)/8 if m else F(0)
    Bplus=one_above(m,0)/8
    P=Pplus+negative if m else F(0)
    assert m*P+3*negative+Bplus==F(1,16)
    return [float(P),float(negative),float(Bplus),float(negative),float(negative)]
def main():
    start=time.process_time();rows=[]
    for T in (4,6,10):
        for kappa in (0.,.5,2.):
            for k in (0.,.25,2.):
                for outer,inner in ((64,32),(128,32),(64,64),(128,64)):
                    r=integrate(T,kappa,k,outer,inner)
                    if kappa==k==0:
                        r['exact_classes']=exact(T);r['max_exact_error']=max(abs(a-b) for a,b in zip(r['classes'],r['exact_classes']));assert r['max_exact_error']<1e-13
                    rows.append(r)
    out={'rows':rows,'cpu_seconds':time.process_time()-start,'sampled_inputs':0};Path(__file__).with_name('negative_controls_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':out['cpu_seconds'],'max_partition_error':max(abs(r['partition_error']) for r in rows),'max_exact_error':max(r.get('max_exact_error',0) for r in rows),'max_fine_partition_error':max(abs(r['partition_error']) for r in rows if r['outer']==128 and r['inner']==64)},indent=2))
if __name__=='__main__':main()
