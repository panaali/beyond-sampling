"""Pinned local exact-zero halfspace guard; original module files remain unchanged."""
from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent
base=HERE/'analytic_controls.py';expected=json.loads((HERE/'analytic_pins.json').read_text())[str(base.resolve())]
assert hashlib.sha256(base.read_bytes()).hexdigest()==expected
source=base.read_text();anchor="name=sys.argv[1];assert name in"
assert source.count(anchor)==1
insertion='''
original_clip=FX.clip
guard_counts={'calls':0,'exact_zero':0}
def guarded_clip(poly,n,ops=None):
 guard_counts['calls']+=1
 # Component equality tests exact zeros; this does not classify tiny normals.
 if ops is not None:ops.add(2*len(n))
 if bool(np.all(np.asarray(n)==0)):
  guard_counts['exact_zero']+=1
  return poly,[]
 return original_clip(poly,n,ops)
FX.clip=guarded_clip
'''
source=source.replace(anchor,insertion+'\n'+anchor)
source=source.replace("path=HERE/('analytic_'+name+'.json')","path=HERE/('guard_'+name+'.json')")
source=source.replace("out.update(complete=True,", "out.update(guard_counts=guard_counts,guard_scope='exact component zero only; nonzero underflow not repaired',guard_sha256='"+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+"',complete=True,")
namespace={'__name__':'__main__','__file__':str(base)}
exec(compile(source,str(base)+' [review-only exact-zero guard]','exec'),namespace)
# Preserve every nonzero case's numerical output exactly, even if timing/counts change.
old=json.loads((HERE/('analytic_'+sys.argv[1]+'.json')).read_text());new=json.loads((HERE/('guard_'+sys.argv[1]+'.json')).read_text())
if sys.argv[1]!='zero':assert [x['estimate'] for x in new['arms']]==[x['estimate'] for x in old['arms']]
assert all(x['passes_frozen_tolerance'] for x in new['arms'])
