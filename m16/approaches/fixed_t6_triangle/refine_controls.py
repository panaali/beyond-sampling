"""Frozen permutation, selected positive reference and paired timing cycle."""
import itertools,json,time
import numpy as np
from check import HERE,Audit,Engine,outer_rule,prepare,evaluate

def main():
 start=time.process_time();out=dict(permutations=[],timings=[]);old=Engine(Audit());new=Engine(Audit())
 eye=np.eye(3,6);anchor=.2*np.ones(6)/6+.8*np.eye(6)[0];cases=[]
 for width in (1.,.1,.01,.001,.0001,.00001,.0000001,0.):
  v=np.ones((3,6))+width*eye;v/=v.sum(axis=1)[:,None];cases.append((['symmetric',width],v,np.zeros((1,6)),np.ones(1)))
 for width in (.1,.01,.001,.0001,.00001,.0000001,0.):
  v=np.stack([anchor,anchor+width*(eye[1]-eye[0]),anchor+width*(eye[2]-eye[0])]);cases.append((['asymmetric',width],v,np.zeros((1,6)),np.ones(1)))
 rays=np.array(json.loads((HERE.parent/'directed_chebyshev/t6_fixture.json').read_text())['rays']);ray=rays[2]
 y,rest,w=outer_rule(8,Audit());cases.append((['stored',2,8],ray[:,3:].T,y@ray[:,:3].T,rest))
 for label,base,prefix,rest in cases:
  values=[];row=dict(label=label,permutations=[])
  for perm in itertools.permutations(range(3)):
   b=base[list(perm)];vv=prefix[:,None,:]+rest[:,None,None]*b[None,:,:]
   v0,s0=old.batch(vv);v1,s1=evaluate(prefix,rest,prepare(b),new.positive);values.append(v1)
   row['permutations'].append(dict(order=perm,max_relative_difference=float(np.max(np.abs(v1/v0-1))),old_fallbacks=s0['positive_quadrature_fallbacks'],new_fallbacks=s1['positive_quadrature_fallbacks']))
   if label==['stored',2,8] and perm==(0,1,2):
    index=int(np.argmax(np.abs(v1/v0-1)));ref,meta=new.positive(vv[index],6,1e-11,128)
    out['selected_positive_control']=dict(node=index,vertices=vv[index].tolist(),old_value=float(v0[index]),new_value=float(v1[index]),positive_value=ref,positive_stats=meta,old_relative_difference=abs(v0[index]/ref-1),new_relative_difference=abs(v1[index]/ref-1))
  row['maximum_permutation_relative_difference']=float(np.max(np.abs(np.array(values)/values[0]-1)));out['permutations'].append(row)
 ray=np.eye(6);y,rest,w=outer_rule(16,Audit());prefix=y@ray[:,:3].T;vertices=prefix[:,None,:]+rest[:,None,None]*ray[:,3:].T[None,:,:]
 # Both paths include endpoint creation; Engine import/setup separately below.
 for rep in range(20):
  row=dict(repetition=rep)
  for name in (('old','new') if rep%2==0 else ('new','old')):
   c=time.process_time();t=time.perf_counter()
   engine=Engine(Audit())
   if name=='old':v,stats=engine.batch(prefix[:,None,:]+rest[:,None,None]*ray[:,3:].T[None,:,:])
   else:v,stats=evaluate(prefix,rest,prepare(ray[:,3:].T),engine.positive)
   row[name+'_wall']=time.perf_counter()-t;row[name+'_cpu']=time.process_time()-c
   row[name+'_sum']=float(v.sum())
  out['timings'].append(row)
 out['timing_totals']={k:sum(r[k] for r in out['timings']) for k in ('old_wall','new_wall','old_cpu','new_cpu')}
 out['cpu_seconds']=time.process_time()-start
 out['maximum_permutation_difference']=max(r['maximum_permutation_relative_difference'] for r in out['permutations'])
 out['changed_fallbacks']=[dict(label=r['label'],**p) for r in out['permutations'] for p in r['permutations'] if p['old_fallbacks']!=p['new_fallbacks']]
 (HERE/'refined_controls.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k not in ('permutations','timings','selected_positive_control')},indent=2))
 print('positive',out['selected_positive_control'])
if __name__=='__main__':main()
