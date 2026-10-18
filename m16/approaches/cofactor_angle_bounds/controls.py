"""Frozen21 exact/undefined analytic fixtures and independent high-precision angles."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time
import mpmath as mp
from bound import enclose,pack,serial
HERE=Path(__file__).resolve().parent
FIXTURES=[
 ('zero',1,0,0,0,'bounded',F(0)),
 ('cross',1,-1,1,0,'bounded',F(1)),
 ('positive',1,1,2,F(3,2),'bounded',F(1,4)),
 ('negative',1,-2,-1,F(-3,2),'bounded',F(1,4)),
 ('x0positive',0,1,2,F(3,2),'bounded_constant',F(0)),
 ('x0negative',0,-2,-1,F(-3,2),'bounded_constant',F(0)),
 ('origin_cross',0,-1,1,0,'unknown_origin_possible',None),
 ('origin_endpoint',0,0,1,1,'unknown_origin_possible',None),
 ('origin_point',0,0,0,0,'unknown_origin_possible',None),
 ('negative_x',-1,1,2,1,'unsupported_negative_x',None),
 ('tiny_x_cross',F(1,2**100),-1,1,0,'bounded',F(4)),
 ('huge_x_cross',2**100,-1,1,0,'bounded',F(1,2**100)),
 ('singleton',1,1,1,1,'bounded',F(0)),
 ('zero_left',1,0,1,0,'bounded',F(1)),
 ('zero_right',1,-1,0,0,'bounded',F(1)),
 ('tiny_x_same_sign',F(1,2**500),1,2,F(3,2),'bounded',None),
 ('scaled_positive',2**500,2**500,2**501,3*2**499,'bounded',F(1,4)),
 ('origin_negative_endpoint',0,-1,0,-1,'unknown_origin_possible',None),
 ('point_outside',1,1,2,3,'invalid_point_outside',None),
 ('bad_order',1,2,1,1,'invalid_interval',None),
 ('nonrational',None,1,2,1,'unknown_nonrational',None)]
def m(x):return mp.mpf(x.numerator)/x.denominator
def main():
 out=HERE/'controls.json';progress=HERE/'controls_progress.jsonl';assert not out.exists() and not progress.exists()
 proto=json.loads((HERE/'protocol.json').read_text());assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==proto['controls_sha256']
 assert hashlib.sha256((HERE/'bound.py').read_bytes()).hexdigest()==proto['bound_sha256']
 mp.mp.dps=120;rows=[];fail=[];start=time.process_time()
 for label,x,lo,hi,p,status,expected in FIXTURES:
  with progress.open('a') as f:f.write(json.dumps(dict(stage='attempt',label=label))+'\n')
  values=[None if v is None else F(v) for v in [x,lo,hi,p]];x,lo,hi,p=values;got=enclose(*values)
  ok=got['status']==status and(expected is None or got.get('bound')==expected);refs=[]
  if got['status'].startswith('bounded'):
   for y in sorted({lo,p,hi,(lo+p)/2,(p+hi)/2}):
    delta=abs(mp.atan2(m(y),m(x))-mp.atan2(m(p),m(x)));inside=delta<=m(got['bound'])+mp.mpf('1e-110');ok&=inside
    refs.append(dict(y=serial(y),angle_difference=mp.nstr(delta,110),inside=bool(inside)))
  row=dict(label=label,inputs=[serial(v) for v in values],result=pack(got),references=refs,ok=bool(ok));rows.append(row)
  with progress.open('a') as f:f.write(json.dumps(row)+'\n')
  if not ok:fail.append(label)
 assert len(rows)==21
 out.write_text(json.dumps(dict(cases=len(rows),failures=fail,rows=rows,cpu_seconds=time.process_time()-start,scope='Exact rational derivative proof is separate from120digit diagnostic reference angles; no libm-rounding or mass certificate.'),indent=2)+'\n');print(json.dumps(dict(cases=len(rows),failures=fail)))
if __name__=='__main__':main()
