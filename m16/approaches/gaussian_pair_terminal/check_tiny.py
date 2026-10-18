from collections import defaultdict
import json,math,time
from pathlib import Path
import mpmath as mp
from terminal import tiny_integral
HERE=Path(__file__).resolve().parent
mp.mp.dps=80;start=time.process_time();stats=defaultdict(float);rows=[]
fixtures=[(-.5472948572881894,-.5472948572881892,(0.,-6.),(-288.0319287802435,-157.6383933562253),-3.8039629125621475,1)]
# Reflected geometry preserves the positive mass but reverses the rank side.
a,b,l,h,u,side=fixtures[0];fixtures.append((-b,-a,(h[0],-h[1]),(l[0],-l[1]),-u,0))
fixtures.append((1.,math.nextafter(1.,2.),(.3,-.5),(-.2,1.),0.,1))
for a,b,lower,upper,u,second in fixtures:
 value,error=tiny_integral(a,b,lower,upper,u,second,stats)
 aa,bb=mp.mpf(a),mp.mpf(b);ll=[mp.mpf(v) for v in lower];hh=[mp.mpf(v) for v in upper];uu=mp.mpf(u)
 def integrand(t):
  x=aa+(bb-aa)*t;lo=ll[0]*x+ll[1];hi=hh[0]*x+hh[1]
  if second:lo=max(lo,uu)
  else:hi=min(hi,uu)
  if hi<=lo:return mp.mpf(0)
  return (bb-aa)*mp.exp(-x*x/2)/mp.sqrt(2*mp.pi)*(mp.erf(hi/mp.sqrt(2))-mp.erf(lo/mp.sqrt(2)))/2
 exact=mp.quad(integrand,[0,1]);difference=abs(mp.mpf(value)-exact)
 row=dict(interval=[a,b],lower=lower,upper=upper,u=u,second=second,value=value,error_diagnostic=error,reference=str(exact),absolute_difference=float(difference),within_diagnostic=bool(difference<=error));rows.append(row);assert difference<=error
out=dict(rows=rows,stats=dict(stats),cpu_seconds=time.process_time()-start,certificate=False)
(HERE/'tiny_controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
