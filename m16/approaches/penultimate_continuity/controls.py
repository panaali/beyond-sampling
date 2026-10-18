from fractions import Fraction as F
from pathlib import Path
import json,time
from certificate import certify,serial
start=time.process_time()
wi=(F(1),F(-2));wh=((F(1),F(0)),(F(0),F(1)))
wo=((F(0),F(0)),(F(1),F(1)),(F(2),F(-1)))
ordinary=certify((wi,wh,wo),2);assert ordinary['sufficient']
constant=certify((wi,wh,((F(1),F(1)),)*3),2)
assert constant['sufficient'] and all(r['permanent_tie'] for r in constant['margins'])
memoryless=certify(((F(0),F(0)),((F(0),F(0)),)*2,wo),2)
assert memoryless['sufficient'] and memoryless['G']==0
degenerate=certify(((F(1),F(-1)),wh,((F(0),F(0)),(F(2),F(0)),(F(1),F(1)))),2)
assert not degenerate['sufficient'] and any(r['kind']=='margin' for r in degenerate['unknown'])
zero_gate=certify(((F(0),F(1)),wh,wo),2)
assert not zero_gate['sufficient'] and any(r['kind']=='gate' for r in zero_gate['unknown'])
out=dict(ordinary=ordinary,constant=constant,memoryless=memoryless,degenerate=degenerate,
         zero_gate=zero_gate,cpu_seconds=time.process_time()-start)
(Path(__file__).resolve().parent/'controls.json').write_text(json.dumps(serial(out),indent=2)+'\n')
print(json.dumps({k:v['sufficient'] for k,v in out.items() if isinstance(v,dict)}))
