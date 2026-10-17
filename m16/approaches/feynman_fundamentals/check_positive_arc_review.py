"""Fresh mixed-sign arc and reversed orientation, both endpoint branches.

Different reference: bounded normalized positive 80-digit integration.
No polygon/conditional-query duplication or cost benchmark.
"""
import importlib.util,json,math,time
from pathlib import Path
import mpmath as mp
SOURCE=Path('/workspace/work/recover-small_h2_78/m16/approaches/real_arc_tail/recurrence.py')
spec=importlib.util.spec_from_file_location('review_real_arc',SOURCE)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
start=time.process_time();mp.mp.dps=80
a,b=.37,-.81;rows=[]
for theta in (.73,2.4):
    aa,bb,tt=map(mp.mpf,(a,b,theta))
    amplitude=abs(aa)+abs(bb)*min(tt,1)
    actual=mod.endpoint_arc(a,b,theta,8)
    ar=a*math.cos(theta)+b*math.sin(theta)
    br=a*math.sin(theta)-b*math.cos(theta)
    reverse=mod.endpoint_arc(ar,br,theta,8)
    for i,n in enumerate(range(0,9,2)):
        ref=tt if not n else tt*amplitude**n*mp.quad(lambda u:((aa*mp.cos(tt*u)+bb*mp.sin(tt*u))/amplitude)**n,[0,mp.mpf('.5'),1])
        relative=abs(mp.mpf(actual[i])-ref)/ref
        reversediff=abs(actual[i]-reverse[i])/float(ref)
        assert relative<mp.mpf('3e-13') and reversediff<3e-13
        rows.append(dict(theta=theta,degree=n,value=actual[i],reference=mp.nstr(ref,80),relative_error=float(relative),reversal_relative_difference=reversediff))
out=dict(rows=rows,cpu_seconds=time.process_time()-start,
         max_relative_error=max(x['relative_error'] for x in rows),max_reversal_relative_difference=max(x['reversal_relative_difference'] for x in rows),
         scope='Fresh mixed-sign arc; normalized independent mpmath reference; special-function floating errors not certified.')
Path(__file__).with_name('positive_arc_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
