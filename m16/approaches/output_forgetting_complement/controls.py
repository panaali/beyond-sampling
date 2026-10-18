import json,time
from fractions import Fraction as F
from pathlib import Path
from collections import defaultdict
from certificate import find,validate,serial,history,box_radius
HERE=Path(__file__).resolve().parent;start=time.process_time()
wi=(F(1),F(-1));wh=((F(1),F(0)),(F(0),F(1)));wo=((F(1),F(0)),(F(0),F(1)),(F(-1),F(-1)),(F(0),F(0)))
cases={'memoryless':(wi,((F(0),F(0)),(F(0),F(0))),wo),'constant':(wi,wh,tuple((F(0),F(0)) for _ in range(4))),'duplicate':(wi,wh,tuple((F(1),F(1)) for _ in range(4))),'memory':((F(0),F(0)),wh,wo)}
out={'rows':[]}
for name,model in cases.items():
 r=find(model,rescue=False);assert r['found']==(name=='memory')
 if r['found']:r['validation']=validate(model,r)
 out['rows'].append(dict(name=name,result=serial(r)))
out['cpu_seconds']=time.process_time()-start
(HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(cpu=out['cpu_seconds'],found=[(r['name'],r['result']['found']) for r in out['rows']])))
