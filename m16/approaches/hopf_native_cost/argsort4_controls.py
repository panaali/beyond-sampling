"""Frozen scalar reconstruction of source's short AVX512 key/index network."""
import itertools,json,time,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def network(a):
 if all(a[i-1]<=a[i] for i in range(1,4)):return np.arange(4,dtype=np.int64)
 key=np.concatenate((np.array(a,dtype=float),np.full(4,np.inf)));ids=np.concatenate((np.arange(4,dtype=np.int64),np.full(4,np.iinfo(np.int64).max)))
 for action,block,mask in [('reverse',2,0xAA),('reverse',4,0xCC),('swap',2,0xAA),('reverse',8,0xF0),('swap',4,0xCC),('swap',2,0xAA)]:
  perm=np.arange(8).reshape(-1,block)
  perm=(perm[:,::-1] if action=='reverse' else np.roll(perm,block//2,axis=1)).reshape(-1)
  other=key[perm];otheridx=ids[perm]
  low=np.minimum(other,key);high=np.maximum(other,key);choose=np.array([bool(mask&(1<<i)) for i in range(8)])
  new=np.where(choose,high,low);ids=np.where(new==key,ids,otheridx);key=new
 return ids[:4]

def main():
 dst=HERE/'argsort4_controls.json';assert not dst.exists();start=time.process_time();out={'status':'running','rows':[]};save=lambda:dst.write_text(json.dumps(out,indent=2)+'\n');save()
 try:
  cases=list(itertools.permutations(range(4)))+list(itertools.product((-1.,0.,1.),repeat=4))+[(-0.,0.,1.,-1.),(0.,-0.,0.,-0.),(-0.,-1.,0.,1.)]
  for i,x in enumerate(cases):
   a=np.array(x,dtype=np.float64);got=network(a);expected=np.argsort(a);out['rows'].append({'case':i,'keys':a.tolist(),'numpy':expected.tolist(),'source_network':got.tolist()});assert np.array_equal(got,expected),(i,a,got,expected)
  out.update(status='complete',cpu_seconds=time.process_time()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Finite length4 scalar network parity, not binary dispatch or full runtime cost.');save();print(json.dumps({'status':out['status'],'cases':len(out['rows']),'cpu_seconds':out['cpu_seconds']}))
 except Exception as e:out.update(status='failed',failure=repr(e),cpu_seconds=time.process_time()-start);save();raise
if __name__=='__main__':main()
