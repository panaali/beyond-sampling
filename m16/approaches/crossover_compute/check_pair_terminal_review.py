import json,hashlib,time
from pathlib import Path
import numpy as np
import mpmath as mp
from m16.approaches.gaussian_pair_terminal.terminal import PairTerminal
mp.mp.dps=60
Phi=lambda x:(1+mp.erf(mp.mpf(str(x))/mp.sqrt(2)))/2
M=lambda a,b:Phi(b)-Phi(a)
model=(np.array([1.,-1.]),np.zeros((2,2)),np.array([[1.,0.],[0.,1.],[0.,0.],[0.,0.]]))
start=time.process_time();rows=[]
for hidden in ([0.,0.],[.3,.7]):
 for label in (0,1):
  terminal=PairTerminal(model,hidden,label)
  for u in (-.9,0.,.7):
   got,info=terminal.probabilities(u)
   positive_below=M(0,u)/M(-6,u) if u>0 else mp.mpf(0)
   positive_above=M(max(u,0),6)/M(u,6)
   expected=[positive_below,(positive_below+positive_above)/2]
   if label==1:expected=[1-x for x in expected]
   err=max(abs(mp.mpf(float(a))-b) for a,b in zip(got,expected))
   assert err<mp.mpf('2e-13')
   rows.append(dict(hidden=hidden,label=label,u=u,error=str(err)))
source=Path('m16/approaches/gaussian_pair_terminal/terminal.py')
result=dict(scope='Independent closed-form truncated-Gaussian conditional law; zero-recurrence two-ray fixture only, no trained estimate or cost validation',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),max_error=max(float(x['error']) for x in rows),cpu_seconds=time.process_time()-start,rows=rows)
Path('m16/approaches/crossover_compute/PAIR_TERMINAL_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['max_error'],result['cpu_seconds'])
