from fractions import Fraction as F
from pathlib import Path
import json,time,math
from two_pair import integrate

def monomial_integral(n):return (1-F(1,2)**(n+1))/F(n+1)
def positive_poly(n,p):
    return sum((F(math.comb(p,j))*(-F(1,2))**(p-j)*(monomial_integral(n+j)-monomial_integral(n+j+1)) for j in range(p+1)),F(0))
def one_above(n,p):
    return (n*positive_poly(n-1,p) if n else 0)+(p*positive_poly(n,p-1) if p else 0)
def exact(T,csign):
    m=T-4;pos=3 if csign=='+' else 2;neg=4-pos
    P=one_above(m-1,pos)/2**neg if m else F(0);B=one_above(m,pos-1)/2**neg
    assert m*P+pos*B==F(1,16)
    return [float(P),float(B),float(B) if csign=='+' else 0.,float(B)]
def main():
    start=time.process_time();rows=[]
    for T in (4,6,10):
        for kappa in (0.,.5,2.):
            for k in (0.,.25,2.):
                for csign in ('+','-'):
                    for outer,inner in ((64,32),(128,32),(64,64),(128,64)):
                        r=integrate(T,kappa,k,csign,outer,inner);assert min(r['classes'])>=0
                        if kappa==k==0:
                            r['exact_classes']=exact(T,csign);r['max_exact_error']=max(abs(a-b) for a,b in zip(r['classes'],r['exact_classes']));assert r['max_exact_error']<1e-13
                        rows.append(r)
    out={'rows':rows,'cpu_seconds':time.process_time()-start,'sampled_inputs':0}
    Path(__file__).with_name('two_pair_controls_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':out['cpu_seconds'],'max_partition_error':max(abs(r['partition_error']) for r in rows),'max_exact_error':max(r.get('max_exact_error',0) for r in rows),'max_fine_partition_error':max(abs(r['partition_error']) for r in rows if r['outer']==128 and r['inner']==64)},indent=2))
if __name__=='__main__':main()
