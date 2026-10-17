"""Pinned original forward, integer hit return only; controls and fixed first block."""
import argparse,ast,hashlib,json,resource,time,types
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_scope(name):
 scope=json.loads((HERE/name).read_text())
 for p,h in scope['pins'].items():assert sha(p)==h,p
 return scope

def load():
 protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['H']==5 and protocol['T']==4
 for row in protocol['files'].values():assert sha(row['path'])==row['sha256']
 path=Path(protocol['files']['forward']['path']);tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mc_acc'];assert len(tree.body)==1
 old=types.ModuleType('original_h5_forward');old.np=np;exec(compile(tree,str(path),'exec'),old.__dict__)
 returns=[n for n in ast.walk(tree) if isinstance(n,ast.Return)];assert len(returns)==1 and ast.unparse(returns[0].value)=='ok / n'
 returns[0].value=ast.Name(id='ok',ctx=ast.Load());new=types.ModuleType('integer_h5_forward');new.np=np;exec(compile(ast.fix_missing_locations(tree),str(path)+' [integer hit return only]','exec'),new.__dict__)
 return protocol,old.mc_acc,new.mc_acc

def model_for(protocol,index):
 ident=protocol['models'][index];assert ident['index']==index and (ident['seed'],ident['copy'])==divmod(index,5)
 data=json.loads(Path(protocol['files']['models']['path']).read_text());r=data['models'][index];assert (r['seed'],r['copy'])==divmod(index,5)
 m=tuple(np.asarray(x,dtype=np.float64) for x in r['model']);assert [x.shape for x in m]==[(5,),(5,5),(4,5)] and all(np.isfinite(x).all() for x in m)
 h=hashlib.sha256(b''.join(x.astype('<f8').tobytes() for x in m)).hexdigest();assert h==r['model_sha256']==ident['model_sha256']
 return m,h

class Fixed:
 def __init__(self,x):self.x=x;self.index=0
 def standard_normal(self,shape):
  n,t=shape;assert t==4;v=self.x[self.index:self.index+n].copy();self.index+=n;assert v.shape==shape;return v

def controls():
 resource.setrlimit(resource.RLIMIT_CPU,(19,20));dst=HERE/'controls.json';assert not dst.exists();start=time.process_time()
 out={'complete':False,'rows':[],'random_draws':0,'completed_extra_forward_evaluations':0,'source_sha256':sha(__file__),'protocol_sha256':sha(HERE/'protocol.json')}
 def save():dst.write_text(json.dumps(out,indent=2)+'\n')
 save()
 try:
  check_scope('source_scope.json');protocol,old,new=load();model,h=model_for(protocol,0);out['model_sha256']=h
  for n in (5,17,31):
   x=(np.arange(n*4,dtype=float).reshape(n,4)%19-9)/4;x[0]=0;x[1]=1;x[2]=[1,1,0,-1]
   for chunk in (3,11):
    row={'n':n,'chunk':chunk,'status':'started','input_sha256':hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()};out['rows'].append(row);save()
    a,b=Fixed(x),Fixed(x);v=old(model,n,a,chunk=chunk);out['completed_extra_forward_evaluations']+=n
    hits=new(model,n,b,chunk=chunk);out['completed_extra_forward_evaluations']+=n
    hidden=np.zeros((n,5))
    for t in range(4):hidden=np.maximum(hidden@model[1].T+x[:,t,None]*model[0][None],0)
    direct=int(np.sum(np.argmax(hidden@model[2].T,axis=1)==np.argsort(x,axis=1)[:,-2]));out['completed_extra_forward_evaluations']+=n
    row.update(value=v,value_hex=float(v).hex(),hits=hits,direct_hits=direct,consumed=[a.index,b.index]);save()
    assert type(hits) is int and hits==direct and float(v).hex()==float(hits/n).hex() and a.index==b.index==n
    row['status']='complete';save()
  assert len(out['rows'])==6 and out['completed_extra_forward_evaluations']==318;out['complete']=True
 except BaseException as e:out['failure']=repr(e);raise
 finally:out.update(cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save()
 print(json.dumps({'complete':out['complete'],'controls':len(out['rows']),'cpu':out['cpu_seconds']}))

def pilot():
 resource.setrlimit(resource.RLIMIT_CPU,(105,110));dst=HERE/'pilot.json';assert not dst.exists();start=time.process_time()
 out={'complete':False,'index':0,'block':0,'source_sha256':sha(__file__),'protocol_sha256':sha(HERE/'protocol.json')}
 def save():dst.write_text(json.dumps(out,indent=2)+'\n')
 save()
 try:
  check_scope('source_scope.json');check_scope('pilot_scope.json');control=json.loads((HERE/'controls.json').read_text());assert control['complete'] and len(control['rows'])==6 and control['completed_extra_forward_evaluations']==318
  protocol,old,new=load();assert protocol['first_pilot']==[0,0,0];model,h=model_for(protocol,0);assert h==control['model_sha256'];ident=protocol['models'][0]
  rng=np.random.Generator(np.random.PCG64(ident['rng_seed']));out.update(seed=0,copy=0,model_sha256=h,n=protocol['block_size'],rng_seed=ident['rng_seed'],rng_name=ident['rng_name'],state_before=rng.bit_generator.state,numpy_version=np.__version__,phase='forward_started');save()
  t=time.process_time();hits=new(model,out['n'],rng,chunk=out['n']);forward_cpu=time.process_time()-t
  assert type(hits) is int and 0<=hits<=out['n']
  projection=forward_cpu*protocol['blocks_per_model']*len(protocol['models'])
  out.update(complete=True,phase='complete',hits=hits,state_after=rng.bit_generator.state,forward_cpu_seconds=forward_cpu,projected_full_forward_cpu_seconds=projection,projection_gate_pass=projection<=protocol['total_projection_gate_forward_cpu_seconds'],scope='One predeclared block; no estimator evaluation or accuracy-based stopping. Projection excludes imports/IO and is scheduling only, not a runtime ceiling. Pilot must be reused once if expansion independently clears.')
 except BaseException as e:out['failure']=repr(e);raise
 finally:out.update(cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save()
 print(json.dumps({k:out[k] for k in ('complete','hits','forward_cpu_seconds','projected_full_forward_cpu_seconds','projection_gate_pass')}))

def main():
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--controls',action='store_true');g.add_argument('--pilot',action='store_true');a=ap.parse_args()
 controls() if a.controls else pilot()
if __name__=='__main__':main()
