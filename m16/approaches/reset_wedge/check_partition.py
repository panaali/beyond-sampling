"""Exact direct two-step recurrence and diagonal-scale region controls."""
import json,time
from fractions import Fraction as F
from pathlib import Path
from .two_suffix import regions

def main():
    start=time.process_time();path=Path(__file__).parent;m=json.loads((path/'results.json').read_text())['models'][0];wi,wh,wo=m['weights'];q=lambda x:F.from_float(float(x));wi=list(map(q,wi));wh=[list(map(q,row)) for row in wh];wo=[list(map(q,row)) for row in wo];cells,_=regions(wi,wh,wo);checks=0
    for c in cells:
        z=(c['lo']+c['hi'])/2 if c['hi'] is not None else c['lo']+1
        for v in (F(1,7),F(2),F(2**20)):
            h=[F(0),F(0)]
            for x in (v,z*v):h=[max(F(0),wi[i]*x+sum(wh[i][j]*h[j] for j in (0,1))) for i in (0,1)]
            logits=[sum(row[i]*h[i] for i in (0,1)) for row in wo];assert max(range(len(wo)),key=lambda i:logits[i])==c['label'];checks+=1
    for exponent in (-30,30):
        s=[F(2)**exponent,F(2)**(-exponent)];w2=[wi[i]*s[i] for i in (0,1)];h2=[[wh[i][j]*s[i]/s[j] for j in (0,1)] for i in (0,1)];o2=[[row[i]/s[i] for i in (0,1)] for row in wo];new,_=regions(w2,h2,o2);assert new==cells
    tied,_=regions(wi,wh,[[0.,0.]]*8);assert all(c['label']==0 for c in tied)
    out=dict(complete=True,direct_recurrence_points=checks,exact_scale_cases=2,positive_volume_zero_readout_tie=True,cpu_seconds=time.process_time()-start)
    (path/'partition_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
