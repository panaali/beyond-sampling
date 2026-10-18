"""Seventeen frozen exact upward-contraction controls; no mass calls."""
from fractions import Fraction as F
from pathlib import Path
import json,time
from weighted import ceiling,BITS
HERE=Path(__file__).resolve().parent
CASES=[F(0),F(1),F(1,3),F(1,7),F(2,7),F(1,2**160),F(1,2**161),F(3,2**161),F(2**200,3),F(1,2**500),F(2**160-1,2**160),F(2**160+1,2**160),F(2**400+1,2**300+1),F(7,11),F(23,31),F(2**300-1,2**301),F(999,1000)]
def main():
    out=HERE/'weighted_controls.json';progress=HERE/'weighted_controls_progress.jsonl'
    assert not out.exists() and not progress.exists()
    start=time.process_time();rows=[];fail=[]
    for i,t in enumerate(CASES):
        with progress.open('a') as f:f.write(json.dumps(dict(stage='attempt',index=i))+'\n')
        u=ceiling(t);upper=F(u,1<<BITS);valid=t<=upper and upper-t<F(1,1<<BITS)
        row=dict(index=i,input=[str(t.numerator),str(t.denominator)],units=str(u),valid=valid)
        if not valid:fail.append(i)
        rows.append(row)
        with progress.open('a') as f:f.write(json.dumps(row)+'\n')
    out.write_text(json.dumps(dict(cases=len(rows),failures=fail,rows=rows,cpu_seconds=time.process_time()-start),indent=2)+'\n')
    assert not fail
if __name__=='__main__':main()
