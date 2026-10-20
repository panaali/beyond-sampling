#!/usr/bin/env python3
"""Independent exhaustive-gate rational check; differs from author's root-splitting oracle."""
from fractions import Fraction as F
from itertools import product
import hashlib, importlib.util, json, random, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'review/late_bound_soundness/local_verifier.py'
spec=importlib.util.spec_from_file_location('t463',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rng=random.Random(462463);start=time.process_time(); rows=[]
for index in range(32):
    H,T,D=2,3,1
    coeff=lambda:F(rng.randrange(-3,4),4)
    wi=[coeff() for _ in range(H)];wh=[[coeff() for _ in range(H)] for _ in range(H)]
    wo=[[coeff() for _ in range(H)] for _ in range(3)]
    A=[[coeff()] for _ in range(T)];c=[coeff() for _ in range(T)]
    minima=[None,None];regions=0
    for bits in product([False,True],repeat=H*T):
        interval=[F(-1),F(1)];state=[(F(0),F(0))]*H;valid=True
        for t in range(T):
            nxt=[]
            for i in range(H):
                a=sum(wh[i][j]*state[j][0] for j in range(H))+wi[i]*A[t][0]
                b=sum(wh[i][j]*state[j][1] for j in range(H))+wi[i]*c[t]
                active=bits[t*H+i];sgn=1 if active else -1
                aa,bb=sgn*a,sgn*b
                if aa>0:interval[0]=max(interval[0],-bb/aa)
                elif aa<0:interval[1]=min(interval[1],-bb/aa)
                elif bb<0:valid=False
                nxt.append((a,b) if active else (F(0),F(0)))
            state=nxt
            if not valid or interval[0]>interval[1]:valid=False;break
        if not valid:continue
        regions+=1
        for out in [1,2]:
            aa=sum((wo[out][j]-wo[0][j])*state[j][0] for j in range(H))
            bb=sum((wo[out][j]-wo[0][j])*state[j][1] for j in range(H))
            v=min(aa*interval[0]+bb,aa*interval[1]+bb)
            minima[out-1]=v if minima[out-1] is None else min(minima[out-1],v)
    bounds=m.exact_crown([-1],[1],wi,wh,wo,A,c)
    assert regions and all(b<=v for b,v in zip(bounds,minima))
    assert not any(b>0 for b in bounds) or any(v>0 for v in minima)
    rows.append(dict(index=index,valid_gate_patterns=regions,bounds=list(map(str,bounds)),
                     exact_minima=list(map(str,minima)),valid=True))
result=dict(verifier_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),seed=462463,cases=rows,cpu_seconds=time.process_time()-start,
            method='32 independently seeded 1D width2 step3 RNNs;64 exact activation patterns each;3 logits; rational interval intersections')
Path(__file__).with_name('T463_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(cases=len(rows),passed=all(x['valid'] for x in rows),cpu_seconds=result['cpu_seconds'])))
