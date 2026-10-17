"""Exact original mc_acc body; return integer hits instead of final mean."""
import argparse,ast,hashlib,json,resource,time,types
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load():
 protocol=json.loads((HERE/'protocol.json').read_text())
 for row in protocol['files'].values():assert sha(row['path'])==row['sha256']
 path=Path(protocol['files']['forward']['path']);tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mc_acc'];assert len(tree.body)==1
 original=types.ModuleType('original_fixed_forward');original.np=np;exec(compile(tree,str(path),'exec'),original.__dict__)
 returns=[n for n in ast.walk(tree) if isinstance(n,ast.Return)];assert len(returns)==1 and ast.unparse(returns[0].value)=='ok / n'
 returns[0].value=ast.Name(id='ok',ctx=ast.Load());integer=types.ModuleType('integer_hits_same_forward');integer.np=np;exec(compile(ast.fix_missing_locations(tree),str(path)+' [integer hit return only]','exec'),integer.__dict__)
 return protocol,original.mc_acc,integer.mc_acc

def model_for(protocol,seed,copy):
 data=json.loads(Path(protocol['files']['models']['path']).read_text());row=next(r for r in data['rows'] if (r['seed'],r['copy'])==(seed,copy));m=tuple(np.asarray(w,dtype=np.float64) for w in row['model']);h=hashlib.sha256(b''.join(w.astype('<f8').tobytes() for w in m)).hexdigest();assert h==row['model_sha256'];return m,h

class Fixed:
 def __init__(self,x):self.x=x;self.index=0
 def standard_normal(self,shape):
  n,t=shape;out=self.x[self.index:self.index+n];assert out.shape==(n,t);self.index+=n;return out.copy()

def controls(protocol,old,new):
 m,h=model_for(protocol,0,0);out=[]
 for n in (5,17,31):
  x=(np.arange(n*4,dtype=np.float64).reshape(n,4)%19-9)/4
  # Two nontrivial chunk partitions exercise hit reduction and final batches.
  for chunk in (3,11):
   a=Fixed(x);b=Fixed(x);value=old(m,n,a,chunk=chunk);hits=new(m,n,b,chunk=chunk)
   hidden=np.zeros((n,4))
   for t in range(4):hidden=np.maximum(hidden@m[1].T+x[:,t,None]*m[0][None],0)
   direct=int(np.sum(np.argmax(hidden@m[2].T,axis=1)==np.argsort(x,axis=1)[:,-2]))
   assert hits==direct and value==hits/n and a.index==b.index==n
   out.append(dict(n=n,chunk=chunk,hits=hits,value=value))
 return dict(records=out,extra_forward_evaluations=3*2*(5+17+31),model_sha256=h,scope='Deterministic parity diagnostics, not part of random reference stream; same stored-float network and original argsort tie semantics.')

def run(seed,copy,first,last):
 protocol,old,new=load();assert 0<=first<=last<100
 ident=next(r for r in protocol['models'] if (r['seed'],r['copy'])==(seed,copy));model,h=model_for(protocol,seed,copy)
 assert ident['model_sha256']==h
 assert sha(HERE/'controls.json')=='7c9a13ef81207e5e9e3fdcf948208b2a57af99ad5d6fe39e6cfa2dcda52efaf1'
 checked=json.loads((HERE/'controls.json').read_text());assert len(checked['records'])==6 and checked['extra_forward_evaluations']==318
 folder=HERE/f's{seed}_c{copy}';folder.mkdir(exist_ok=True)
 pin=sha(HERE/'protocol.json');source=sha(HERE/'reference.py')
 for block in range(first,last+1):
  dst=folder/f'block{block:03d}.json';assert not dst.exists();rng=np.random.default_rng(ident['rng_seed'])
  if block:
   prior=json.loads((folder/f'block{block-1:03d}.json').read_text());assert prior['status']=='complete' and prior['block']==block-1 and prior['model_sha256']==h and prior['protocol_sha256']==pin and prior['source_sha256']==source
   rng.bit_generator.state=prior['state_after']
  out=dict(status='running',seed=seed,copy=copy,block=block,n=1000000,model_sha256=h,protocol_sha256=pin,source_sha256=source,state_before=rng.bit_generator.state,rng_seed=ident['rng_seed'])
  save=lambda:dst.write_text(json.dumps(out,indent=2)+'\n');save();start=time.process_time()
  try:
   hits=new(model,1000000,rng,chunk=1000000);assert isinstance(hits,int) and 0<=hits<=1000000
   out.update(status='complete',hits=hits,state_after=rng.bit_generator.state,cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save()
  except Exception as e:out.update(status='failed',failure=repr(e),cpu_seconds=time.process_time()-start);save();raise
 print(json.dumps(dict(seed=seed,copy=copy,first=first,last=last,last_hits=out['hits'],last_cpu=out['cpu_seconds'])))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--controls',action='store_true');args=ap.parse_args()
 if args.controls:
  dst=HERE/'controls.json';assert not dst.exists();p,a,b=load();out=controls(p,a,b);dst.write_text(json.dumps(out,indent=2)+'\n');print(out)
 else:run(0,0,0,0)
if __name__=='__main__':main()
