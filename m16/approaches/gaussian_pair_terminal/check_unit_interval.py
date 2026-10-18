"""High-precision positive integral for the preserved128-ulp failure."""
from collections import defaultdict
import json,math,time
from pathlib import Path
import mpmath as mp
from scipy.integrate import quad
from terminal import mass,SQRT2PI
HERE=Path(__file__).resolve().parent;mp.mp.dps=80;start=time.process_time();stats=defaultdict(float)
a=-2.189179429152813;b=-2.1891794291527567;u=-1.1966540626088599e-14;slope=.2133125364780426;offset=.4669794168381279;width=b-a
value,error=quad(lambda t:math.exp(-(a+width*t)**2/2)/SQRT2PI*mass(-6.,min(slope*(a+width*t)+offset,u),stats),0,1,epsabs=1e-14/width,epsrel=1e-11)
value*=width;error*=width
aa,ww,uu,ss,oo=(mp.mpf(x) for x in (a,width,u,slope,offset))
exact=mp.quad(lambda t:ww*mp.exp(-(aa+ww*t)**2/2)/mp.sqrt(2*mp.pi)*(mp.erf(min(ss*(aa+ww*t)+oo,uu)/mp.sqrt(2))-mp.erf(-6/mp.sqrt(2)))/2,[0,1])
difference=float(abs(mp.mpf(value)-exact));out=dict(interval=[a,b],value=value,quad_error_estimate=error,reference=str(exact),absolute_difference=difference,relative_difference=difference/float(exact),work=dict(stats),cpu_seconds=time.process_time()-start,certificate=False)
assert difference<1e-28
(HERE/'unit_interval_control.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
