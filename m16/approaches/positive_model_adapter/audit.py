"""Pinned source dependency and synthetic controls; sentinel values are not masses."""
import time
START=time.process_time()
from pathlib import Path
import ast,hashlib,importlib.util,json,resource,sys
import numpy as np
P=Path(__file__).resolve().parent

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(20,22))
 proto=json.loads((P/'protocol.json').read_text())
 for f,h in proto['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
 source=Path(next(k for k in proto['pins'] if k.endswith('estT_snapshot.py')));m43=Path(next(k for k in proto['pins'] if k.endswith('m43.py')))
 sys.path.insert(0,str(m43.parent));spec=importlib.util.spec_from_file_location('pinned_positive_adapter_engine',source);E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
 tree=ast.parse(source.read_text());chain={'orthant','mass','_cell_error','_finish','estT'};sites=[]
 for func in tree.body:
  if isinstance(func,ast.FunctionDef):
   for n in ast.walk(func):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in chain:sites.append(dict(caller=func.name,callee=n.func.id,line=n.lineno,expression=ast.unparse(n)))
 assert {(x['caller'],x['callee']) for x in sites}=={('mass','orthant'),('_cell_error','mass'),('_finish','_cell_error'),('_finish','_finish'),('estT','_finish')}
 models={'active_identical_nonzero':(np.array([1.,-1.]),np.zeros((2,2)),np.ones((4,2))),'active_identical_zero':(np.array([1.,-1.]),np.zeros((2,2)),np.zeros((4,2))),'alloff':(np.zeros(2),np.zeros((2,2)),np.zeros((4,2)))}
 rows=[]
 for name,model in models.items():
  digests=[];stats=[]
  for value in proto['sentinel_returns']:
   stream=[];E._HALVES.clear();E.STATS.update({k:0 for k in E.STATS});ops=E.Ops()
   def recorder(rays,o):
    assert rays.shape==(4,4);stream.append(rays.copy());E.STATS['simplices']+=1;return value
   E.orthant=recorder
   estimate=E.estT(model,ops,mode='exact');raw=np.asarray(stream,dtype=np.float64).tobytes();digest=hashlib.sha256(raw).hexdigest();digests.append(digest);stats.append(dict(E.STATS))
   rows.append(dict(model=name,sentinel=value,synthetic_accumulator=estimate,cone_count=len(stream),ray_stream_sha256=digest,stored_float64_bytes=len(raw),stats=dict(E.STATS),inherited_ops_without_mass=ops.n))
   assert estimate==1-value*len(stream)
   if name.startswith('active_identical'):assert estimate==1 and len(stream)==0
  assert len(set(digests))==1 and all(s==stats[0] for s in stats)
 # This exact analytic family has logits constant across all four positions.
 # For x3 !=0, exactly one ReLU(x3),ReLU(-x3) is positive.
 # Hence prediction0, and exchangeability gives P(second-largest index0)=1/4.
 out=dict(complete=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),call_sites=sites,rows=rows,sentinel_geometry_invariant=True,active_identical_counterexample=dict(source_accuracy=1.,exact_accuracy=.25,error=.75,claim='analytic first-index symmetry; no input sampling or numerical mass'),cpu_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Dependency and finite synthetic geometry audit only. Source finite returns/no exception assumed; timers/budget exceptions are distinct. No generic adapter adoption or trained rerun.')
 (P/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
