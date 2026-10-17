"""One fixed source-pinned primitive-control run; no model or quadrature calls."""
import hashlib
import json
import math
from pathlib import Path
import time
import resource
import mpmath as mp
from primitive import integrate

HERE = Path(__file__).resolve().parent


def reference(P, D, digits):
    with mp.workdps(digits):
        if not all(math.isfinite(x) for x in P + D): return {'domain': 'nonfinite'}
        p = [mp.mpf(x) for x in P]; d = [mp.mpf(x) for x in D]
        a = sum(x*x for x in d); b = sum(x*y for x,y in zip(p,d)); c = sum(x*x for x in p)
        delta = sum((p[i]*d[j]-p[j]*d[i])**2 for i in range(5) for j in range(i+1,5))
        if not a:
            if not c: return {'domain': 'singular'}
            value = c**mp.mpf('-2.5')
        elif not delta:
            if not c or 1+b/c <= 0: return {'domain': 'singular'}
            lam = b/c
            value = c**mp.mpf('-2.5')*((1+lam)**-4-1)/(-4*lam)
        else:
            t0=b; t1=a+b
            s0=t0/mp.sqrt(t0*t0+delta); s1=t1/mp.sqrt(t1*t1+delta)
            value = a**mp.mpf('1.5')/delta**2*((s1-s1**3/3)-(s0-s0**3/3))
        return {'domain': 'finite_positive', 'value': mp.nstr(value, digits)}


def naive(P, D):
    try:
        a=sum(x*x for x in D); b=sum(x*y for x,y in zip(P,D)); c=sum(x*x for x in P)
        delta=a*c-b*b
        if a == 0: return {'value':c**-2.5, 'delta':delta}
        s0=b/math.sqrt(b*b+delta); t1=a+b; s1=t1/math.sqrt(t1*t1+delta)
        return {'value':a**1.5/delta**2*((s1-s1**3/3)-(s0-s0**3/3)), 'delta':delta}
    except (OverflowError, ZeroDivisionError, ValueError) as exc:
        return {'exception': type(exc).__name__+': '+str(exc)}


def safe(obj):
    if isinstance(obj,float) and not math.isfinite(obj): return obj.hex()
    if isinstance(obj,dict): return {k:safe(v) for k,v in obj.items()}
    if isinstance(obj,list): return [safe(x) for x in obj]
    return obj


def main():
    dest=HERE/'results.json'
    if dest.exists(): raise RuntimeError('refuse to overwrite preserved results')
    protocol=json.loads((HERE/'protocol.json').read_text())
    for name, expected in protocol['sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected, name
    start=time.process_time(); rows=[]
    try:
        for item in protocol['controls']:
            P=[float.fromhex(x) for x in item['P']]; D=[float.fromhex(x) for x in item['D']]
            before=time.process_time(); result=integrate(P,D); candidate_cpu=time.process_time()-before
            before=time.process_time(); ref=reference(P,D,160); ref2=reference(P,D,200)
            reference_cpu=time.process_time()-before
            row={'id':item['id'],'P':item['P'],'D':item['D'],'candidate':result,'reference160':ref,
                 'reference200':ref2,'naive':naive(P,D),'candidate_cpu':candidate_cpu,'reference_cpu':reference_cpu}
            with mp.workdps(180):
                if ref['domain']=='finite_positive':
                    v=mp.mpf(ref2['value']); diff=abs(mp.mpf(ref['value'])-v)/v
                    row['reference_relative_difference']=mp.nstr(diff,30)
                    row['reference_stable']=bool(diff<mp.mpf('1e-100'))
                    if result['status']=='ok':
                        err=abs(mp.mpf(result['value'])-v)/v
                        row['relative_error']=mp.nstr(err,30); row['diagnostic_pass']=bool(err<=mp.mpf('1e-10'))
                if item['id']=='constant_unit':
                    row['analytic_reference_pass']=bool(mp.mpf(ref2['value'])==1)
                if item['id']=='orthogonal_unit':
                    row['analytic_reference_pass']=bool(abs(mp.mpf(ref2['value'])-5/(6*mp.sqrt(2)))<mp.mpf('1e-150'))
            rows.append(row)
        output={'complete':True,'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),
                'rows':rows,'research_cpu':time.process_time()-start,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'qualifications':['diagnostic error threshold only','no floating certificate','collinear finite domain unsupported',
                                  'no trained or outer-rule application','named operations exclude runtime and library implementations']}
    except BaseException as exc:
        output={'complete':False,'rows':rows,'exception':repr(exc),'research_cpu':time.process_time()-start}
        dest.write_text(json.dumps(safe(output),indent=2,allow_nan=False)+'\n'); raise
    dest.write_text(json.dumps(safe(output),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':True,'rows':len(rows),'cpu':output['research_cpu'],
                      'status_counts':{s:sum(r['candidate']['status']==s for r in rows) for s in sorted({r['candidate']['status'] for r in rows})}}))


if __name__=='__main__': main()
