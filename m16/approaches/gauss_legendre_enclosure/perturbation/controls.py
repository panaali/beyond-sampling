"""Frozen dyadic square-root and exact perturbation algebra controls."""
from pathlib import Path
import hashlib,json,time
from bound import *
HERE=Path(__file__).resolve().parent
def main():
    out=HERE/'controls.json';progress=HERE/'controls_progress.jsonl';assert not out.exists() and not progress.exists()
    P=json.loads((HERE/'protocol.json').read_text())
    for name,pin in P['source_pins'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==pin
    rows=[];start=time.process_time()
    def write(x):
        with progress.open('a') as f:f.write(json.dumps(pack(x))+'\n')
    for a in [F(0),F(1),F(1,4),F(2),F(9,4),F(1,2**320),F(1,2**322)]:
        write(dict(stage='sqrt_attempt',a=a));L=Ledger();lo,hi=sqrt_floor(a,L)
        row=dict(kind='sqrt',a=a,interval=(lo,hi),passed=lo*lo<=a<hi*hi,counts=L.count);rows.append(row);write(row)
    cases=[('zero',F(1),F(0),F(1),F(0),F(0)),('node',F(1),F(1,1000),F(1),F(0),F(46,24000)),('weight',F(1),F(0),F(1),F(1,1000),F(1,6000)),('both',F(1),F(1,1000),F(1),F(1,1000),F(1,480)),('source_sum2',F(1),F(0),F(2),F(1),F(5,8)),('radial_quarter',F(1,4),F(1,1000),F(1),F(0),F(5312,24000))]
    for name,a,e,s,d,expected in cases:
        write(dict(stage='bound_attempt',name=name));got=bound(a,F(1),F(1),e,s,d)
        row=dict(kind='bound',name=name,actual=got,expected=expected,passed=got['status']=='bounded' and got['probability_upper']==expected);rows.append(row);write(row)
    for name,a,e,s,d,status in [('nonpositive',F(0),F(0),F(1),F(0),'unknown_nonpositive_radial_lower'),('negative_error',F(1),F(-1),F(1),F(0),'invalid_negative_parameter'),('sqrt_limit',F(1,2**400),F(1),F(1),F(0),'unknown_fixed_sqrt_precision')]:
        write(dict(stage='status_attempt',name=name));got=bound(a,F(1),F(1),e,s,d)
        row=dict(kind='status',name=name,actual=got,expected=status,passed=got['status']==status);rows.append(row);write(row)
    result=dict(cases=len(rows),failures=[i for i,r in enumerate(rows) if not r['passed']],rows=rows,cpu_seconds=time.process_time()-start)
    out.write_text(json.dumps(pack(result),indent=2)+'\n');assert not result['failures']
if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (HERE/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(phase='controls',type=type(e).__name__,message=str(e)))+'\n')
        raise
