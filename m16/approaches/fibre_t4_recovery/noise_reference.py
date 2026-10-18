"""Original CRC32-name Gaussian truth stream, checkpointed in 1M-input blocks."""
import ast,hashlib,json,resource,sys,time,zlib
from pathlib import Path
import numpy as np
from observer import HERE,load
FX,F3,Ops=load()
from zoo import forward as original_forward
ARC=Path('/workspace/work/recover-small_h4_4/m16/arc_eval.py')
PINS=json.loads((HERE/'reference_pins.json').read_text())
for p,sha in PINS.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha
# Extract the actual shared evaluator function, without importing its T10-only
# checkpoint or unrelated random advice. The sole adaptation is global T=4.
tree=ast.parse(ARC.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='forward'];assert len(nodes)==1
ns={'np':np,'T':4};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ARC)+' [T4 adapter]','exec'),ns);forward=ns['forward']
identity=json.loads((HERE/'noise_identity.json').read_text());model=tuple(np.array(x) for x in identity['model'])
assert hashlib.sha256(b''.join(w.astype('<f8').tobytes() for w in model)).hexdigest()==identity['model_sha256']
N=1_000_000;BATCH=500_000;PLANNED=100_000_000

def controls():
 x=np.array([[0,0,0,0],[-2,-1,0,1],[2,1,0,-1],[1,1,1,1],[-1,-1,-1,-1],[.1,-.2,.3,-.4],[8,-4,2,-1]],dtype=float)
 rows=[]
 for name in ('zero','constant0','plus3_minus0','plus0_minus3'):
  saved=json.loads((HERE/('guard_'+name+'.json')).read_text());m=tuple(np.array(w) for w in saved['model'])
  a=forward(m,x);b=original_forward(m,x);assert np.array_equal(a,b)
  rows.append({'name':name,'n':len(x),'predictions':a.argmax(1).tolist()})
 assert rows[0]['predictions']==rows[1]['predictions']==[0]*len(x)
 assert rows[2]['predictions']==np.where(x[:,-1]>0,3,0).tolist()
 assert rows[3]['predictions']==np.where(x[:,-1]<0,3,0).tolist()
 return rows,2*len(x)*len(rows)

def run(index):
 assert 0<=index<PLANNED//N
 folder=HERE/'noise_reference';folder.mkdir(exist_ok=True);path=folder/f'{index:03d}.json';assert not path.exists(),'Never overwrite an attempt'
 seed=zlib.crc32(identity['name'].encode());rng=np.random.default_rng(seed)
 if index:
  previous=json.loads((folder/f'{index-1:03d}.json').read_text());assert previous['complete'] and previous['n']==N and previous['index']==index-1 and previous['model_sha256']==identity['model_sha256'];rng.bit_generator.state=previous['rng_after']
 start=time.process_time();out={'index':index,'n':0,'target_n':N,'planned_total_n':PLANNED,'batch':BATCH,'seed_crc32':seed,'model_sha256':identity['model_sha256'],'source_pins':PINS,'rng_before':rng.bit_generator.state,'complete':False,'hits':0,'batches':[],'dtype':'float64','semantics':'shared arc_eval.forward with T=4; first-index numpy.argmax; original numpy.argsort second-largest truth'}
 path.write_text(json.dumps(out,indent=2)+'\n')
 try:
  if index==0:out['analytic_parity'],out['extra_parity_evaluations']=controls()
  for batch in range(N//BATCH):
   x=rng.standard_normal((BATCH,4))
   if index==0 and batch==0:
    assert np.array_equal(forward(model,x[:128]),original_forward(model,x[:128]));out['extra_parity_evaluations']+=256
   logits=forward(model,x);truth=np.argsort(x,axis=1)[:,-2]
   hit=int(np.count_nonzero(np.argmax(logits,axis=1)==truth));out['hits']+=hit;out['n']+=BATCH;out['batches'].append({'n':BATCH,'hits':hit})
   out.update(rng_after=rng.bit_generator.state,cpu_seconds=time.process_time()-start);path.write_text(json.dumps(out,indent=2)+'\n')
  out['complete']=True
 except Exception as exc:out['failure']=repr(exc);raise
 finally:
  out.update(rng_after=rng.bit_generator.state,cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('index','complete','n','hits','cpu_seconds','max_rss_kib')}),flush=True)
if __name__=='__main__':
 index=int(sys.argv[1]);assert index==0,'Only frozen throughput pilot authorized by current entry point';run(index)
