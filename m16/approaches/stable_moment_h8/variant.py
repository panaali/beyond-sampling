"""Only the equivalent centered-square variance replaces the original moments."""
from pathlib import Path
import hashlib,importlib.util,inspect,json,types
P=Path(__file__).resolve().parent
PROTOCOL=json.loads((P/'protocol.json').read_text())

def stable_moments(data):
 A,s,r,d=data;t=add(d);off=[dot(A[i],A[j]) for i in range(4) for j in range(i+1,4)];mu=(s+t)/20;center=5*mu
 row_deviations=[r[i]+d[i]-center for i in range(4)]
 entry_deviations=[x-mu for x in d+off]
 var=(add([x*x for x in row_deviations])+add([x*x for x in entry_deviations]))/210
 second=var+mu*mu
 return mu,var,second

def load():
 for p,h in PROTOCOL['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 path=P.parent/'moment_h8_cost/variant.py';spec=importlib.util.spec_from_file_location('reviewed_original_moment_arm',path);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
 M,old,kernel,rescue,radial,moment,source=base.load()
 global add,dot
 add=moment.add;dot=moment.dot;moment.moments=stable_moments
 return M,old,kernel,rescue,radial,moment,source+'\n'+inspect.getsource(stable_moments)

def one(R,M,old,kernel,rescue,radial,rules):
 row=M.one(R,old,kernel,rescue,radial,rules)
 if row['center'].get('moment_used'):row['method']='moment_interval'
 return row
