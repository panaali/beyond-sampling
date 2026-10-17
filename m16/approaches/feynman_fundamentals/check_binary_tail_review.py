"""Independent fixed edge containment checks of the q^-3 binary alternative.

Frozen alternatives: positive polynomial indexing defect; loss of enclosure
under extreme scaling. Closed q^-3 formula at 200 digits, no owner reference.
"""
import hashlib,json,math,sys,time
from pathlib import Path
import mpmath as mp
import numpy as np
HERE=Path(__file__).resolve().parent
SOURCE=Path('/workspace/work/recover-small_h2_56/m16/approaches/directed_chebyshev')
sys.path.insert(0,str(SOURCE))
from binary import Binary

start=time.process_time();mp.mp.dps=200
cases=[(1.,1.,0),(1.,math.nextafter(1.,math.inf),3),
       (1e-300,4e-300,5),(1e300,4e300,5),(1.,1e8,1000),
       (1.,2.,100000),(.125,2.,0),(.9,1.,2**26-1)]
rows=[]
for a,b,k in cases:
    aa,bb=mp.mpf(a),mp.mpf(b);s=mp.sqrt(aa*bb)
    alpha=(aa+bb)/(2*s);r=(bb-aa)/(aa+bb+2*s);N=k+1
    reference=r**N/(s**3*(1-r))*(N*N+3*alpha*N+3*alpha**2-1+(2*N+3*alpha)*r/(1-r)+r*(1+r)/(1-r)**2)
    ctx=Binary();lo,hi=ctx.tail(a,b,k);lo,hi=float(lo),float(hi)
    assert not math.isnan(lo) and not math.isnan(hi)
    assert mp.mpf(lo)<=reference<=mp.mpf(hi),(a,b,k,lo,hi)
    if a==b:assert lo==hi==0
    rows.append(dict(a=a,b=b,degree=k,lo=str(lo),hi=str(hi),reference=mp.nstr(reference,70)))
out=dict(cases=rows,cpu_seconds=time.process_time()-start,
         source_sha256=hashlib.sha256((SOURCE/'binary.py').read_bytes()).hexdigest(),
         scope='Eight independent closed-form edge checks; IEEE correctly rounded basic operations and sqrt assumed.')
(HERE/'binary_tail_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
