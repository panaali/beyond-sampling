import json,time,hashlib
from pathlib import Path
import numpy as np
import loaders as L
from reference import exact
start=time.process_time();rows=[];fixtures=[]
for k,s in [(24,4)]+[(20,s) for s in range(5)]:
 base=L.load(k,s)
 for j,m in enumerate([base]+L.noisy(base,120,1,.05)):
  t=time.process_time();p,n=exact(m);old=L.author_reference.exact_acc(m);a=L.t75(m);f=L.fibre(m)
  rows.append(dict(k=k,seed=s,copy=j,reference=p,reference_leaves=n,author_reference=float(old),reference_difference=p-old,t75=a,fibre=f,cpu=time.process_time()-t))
  assert abs(p-old)<1e-11 and abs(a['own_value']-a['value'])<1e-11
base=L.load(24,4)
for name,m in [('zero_input',(np.zeros(4),base[1],base[2])),('zero_output',(base[0],base[1],np.zeros((3,4)))),('duplicate_output',(base[0],base[1],np.tile(base[2][0],(3,1))))]:
 p,n=exact(m);assert abs(p-1/3)<1e-12
 fixtures.append(dict(name=name,reference=p,expected=1/3,t75=L.t75(m),fibre=L.fibre(m),author_reference=float(L.author_reference.exact_acc(m))))
# Independent arithmetic witness verifies supplemental negation/power charge.
for key in L.COUNT.C:L.COUNT.C[key]=0
x=L.COUNT.Num(3);v=-x+x**3
assert v.x==24 and L.COUNT.C['neg']==1 and L.COUNT.C['mul']==2 and L.COUNT.C['add']==1
paths=[L.ROOT/'m16/approaches/ladder_training_axis/est75_e37.py',L.FIB_PATH,L.ROOT/'review/understand_m43_t75/count.py',L.ROOT/'review/understand_m43_t75/recipe.py']
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
weights={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/workspace/work/t114_checkpoints').glob('*.pth')}
out=dict(rows=rows,fixtures=fixtures,source_hashes=hashes,checkpoint_hashes=weights,cpu=time.process_time()-start)
Path(__file__).with_name('controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(cpu=out['cpu'],max_reference_error=max(abs(r['reference_difference']) for r in rows),max_recipe_difference=max(abs(r['t75']['own_value']-r['t75']['value']) for r in rows),fixtures=fixtures),indent=2))
