"""Exact rational nu3 tail controls, with durable failures and no quadrature."""
from fractions import Fraction as F
from pathlib import Path
from math import isqrt
from collections import Counter
import json,time,resource
from bound import protocol,load,sha
P=Path(__file__).resolve().parent

def root(x):
    a=isqrt(x.numerator);b=isqrt(x.denominator);assert a*a==x.numerator and b*b==x.denominator
    return F(a,b)

def exact_tail(a,b,N):
    if a==b:return F(0)
    ra,rb=root(a),root(b);s=ra*rb;B=(a+b)/(2*s);r=(rb-ra)/(rb+ra);m=N+1
    return r**m/s**3*((m*m+3*B*m+3*B*B-1)/(1-r)+(2*m+3*B)*r/(1-r)**2+r*(1+r)/(1-r)**3)

def main():
    dest=P/'controls.json';assert not dest.exists();start=time.process_time()
    out=dict(complete=False,source_sha256=sha(__file__),bound_sha256=sha(P/'bound.py'),protocol_sha256=sha(P/'protocol.json'),rows=[],invalid=[])
    def save():
        out.update(cpu=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        dest.write_text(json.dumps(out,indent=2)+'\n')
    save()
    try:
        p=protocol();D=load(p).Directed(p['precision'])
        for a0,b0,N in p['controls']:
            row=dict(a=a0,b=b0,degree=N,status='started');out['rows'].append(row);save()
            a,b=F(a0),F(b0);before=Counter(D.counts);aa=D.rational(a);bb=D.rational(b)
            # These square-endpoint controls are exactly representable as Decimal.
            assert F(aa.lo)==F(aa.hi)==a and F(bb.lo)==F(bb.hi)==b
            interval=D.tail(aa.lo,bb.hi,3,N);truth=exact_tail(a,b,N)
            row.update(status='completed',exact=str(truth),interval=[str(interval.lo),str(interval.hi)],
                enclosed=F(interval.lo)<=truth<=F(interval.hi),counts=dict(Counter(D.counts)-before))
            save();assert row['enclosed']
        cases=[('zero_lower',0,1,4),('reversed_range',2,1,4),('negative_degree',1,4,-1),('fractional_degree',1,4,1.5)]
        assert [x[0] for x in cases]==p['invalid_controls']
        for name,a,b,N in cases:
            row=dict(name=name,status='started');out['invalid'].append(row);save()
            try:D.tail(a,b,3,N)
            except ValueError as e:row.update(status='rejected',reason=str(e))
            else:row.update(status='unexpected_acceptance')
            save();assert row['status']=='rejected'
        # Exact closed form of the positive polynomial at nu3.
        got=D.__class__.__module__;module=__import__(got)
        expected={(2,0):F(1),(1,0):F(3),(0,0):F(2),(1,1):F(3),(0,1):F(6),(0,2):F(3)}
        out['nu3_polynomial_equal']=module.polynomial(3)==expected;assert out['nu3_polynomial_equal']
        out['exact_control_cost_scope']='Independent Fraction truth construction is verification work, not included in Directed counters.'
        out['complete']=True
    except BaseException as e:out['failure']=repr(e);raise
    finally:save();print(json.dumps({k:out[k] for k in ('complete','cpu','max_rss_kib')}))
if __name__=='__main__':main()
