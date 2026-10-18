"""Independent saved-only T11588 audit: closed-form Rodrigues coefficients.

No author module import, root isolation, NumPy node generation or mass call.
"""
from pathlib import Path
from fractions import Fraction as F
from math import factorial
import hashlib,json,time

SOURCE=Path('/workspace/work/recover-small_h2_56/m16/approaches/gauss_legendre_enclosure')
HERE=Path(__file__).resolve().parent
def frac(x): return F(int(x[0]),int(x[1]))
def interval(x): return tuple(map(frac,x))
def coefficients(n):
    p=[F(0)]*(n+1)
    for k in range(n//2+1):
        p[n-2*k]=F((-1)**k*factorial(2*n-2*k),2**n*factorial(k)*factorial(n-k)*factorial(n-2*k))
    return p
def value(p,x): return sum((c*x**i for i,c in enumerate(p)),F(0))
def multiply(x,y):
    products=[a*b for a in x for b in y];return min(products),max(products)
def ihorner(p,x):
    result=(F(0),F(0))
    for c in p[::-1]:
        result=multiply(result,x);result=result[0]+c,result[1]+c
    return result
def sq(x):
    a,b=x;return (F(0) if a<=0<=b else min(a*a,b*b)),max(a*a,b*b)
def signs(n,x):
    vals=[value(coefficients(k),x) for k in range(n+1)]
    ss=[(v>0)-(v<0) for v in vals];nz=[s for s in ss if s]
    return ss,sum(a!=b for a,b in zip(nz,nz[1:]))
def packed(x): return [str(x.numerator),str(x.denominator)]
def main():
    out=HERE/'review_gl_nodes.json'
    assert not out.exists()
    start=time.process_time();d=json.loads((SOURCE/'results.json').read_text())
    assert d['complete'] and d['all_bounded'] and d['source_leggauss_calls']==3
    controls=json.loads((SOURCE/'controls.json').read_text());assert controls['cases']==13 and controls['failures']==[]
    assert all(r['pass_'] and r['actual']==r['expected'] for r in controls['rows'])
    roots_count=0;moments_count=0;node_max=F(0);weight_max=F(0)
    for row,n in zip(d['orders'],[6,10,16]):
        assert row['n']==n and row['bounded'] and not row['unresolved']
        positive=[];weights=[];previous=F(0)
        for item in row['positive_certificates']:
            lo,hi=interval(item['root']);assert previous<lo<hi<1 and hi-lo==F(1,2**160);previous=hi
            ls,lv=signs(n,lo);rs,rv=signs(n,hi)
            assert lv-rv==1 and ls==item['left_signs'] and rs==item['right_signs']
            assert (lv,rv)==(item['left_variations'],item['right_variations']) and ls[-1] and rs[-1]
            p=coefficients(n);der=[i*p[i] for i in range(1,n+1)]
            dp=ihorner(der,(lo,hi));saved=item['weight'];assert dp==interval(saved['derivative'])
            den=multiply((1-hi*hi,1-lo*lo),sq(dp));assert den==interval(saved['denominator']) and den[0]>0
            raw=F(2)/den[1],F(2)/den[0];assert raw==interval(saved['raw'])
            dyadic=interval(saved['dyadic']);q=2**128
            lower=F((raw[0]*q).__floor__(),q);upper=F((raw[1]*q).__ceil__(),q)
            assert dyadic==(lower,upper) and lower<=raw[0]<=raw[1]<=upper
            positive.append((lo,hi));weights.append(dyadic)
        assert len(positive)==n//2;roots_count+=n
        full_roots=[(-hi,-lo) for lo,hi in positive[::-1]]+positive
        full_weights=weights[::-1]+weights
        for j,item in enumerate(row['source_comparison']):
            assert item['index']==j and interval(item['ideal_node'])==full_roots[j] and interval(item['ideal_weight'])==full_weights[j]
            for kind,bounds,storedhex in [('node',full_roots[j],row['source_nodes_hex'][j]),('weight',full_weights[j],row['source_weights_hex'][j])]:
                h=item[kind];assert h['source_hex']==storedhex
                x=F.from_float(float.fromhex(storedhex));lo,hi=bounds;inside=lo<=x<=hi
                dist=(F(0) if inside else min(abs(x-lo),abs(x-hi)),max(abs(x-lo),abs(x-hi)))
                assert interval(h['absolute_error_interval'])==dist and h['inside_ideal_enclosure']==inside
                if kind=='node':node_max=max(node_max,dist[1])
                else:weight_max=max(weight_max,dist[1])
        assert len(row['source_comparison'])==n
        for k,m in enumerate(row['moments']):
            truth=F(0) if k%2 else F(2,k+1)
            if k%2: bounds=(F(0),F(0))
            else:
                terms=[multiply((lo**k,hi**k),w) for (lo,hi),w in zip(positive,weights)]
                bounds=2*sum(t[0] for t in terms),2*sum(t[1] for t in terms)
            assert interval(m['interval'])==bounds and frac(m['truth'])==truth and bounds[0]<=truth<=bounds[1] and m['contains']
            moments_count+=1
        assert len(row['moments'])==2*n
    result={'pass':True,'positive_root_certificates':roots_count//2,'full_nodes_weights':roots_count,'moments':moments_count,
            'max_node_absolute_error_upper':packed(node_max),'max_weight_absolute_error_upper':packed(weight_max),
            'source_hashes':{name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ['results.json','controls.json','exact.py','run.py']},
            'cpu':time.process_time()-start,'method':'Rodrigues closed coefficients, exact endpoint signs, independently contracted derivative intervals and all source dyadic distances; no author module or numerical rerun',
            'limits':'Original count ledger only source-reviewed; no complete hardware cost or cone/integrand certificate.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','max_node_absolute_error_upper','max_weight_absolute_error_upper')}))
if __name__=='__main__':main()
