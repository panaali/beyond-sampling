"""Frozen20 alternating4096-node calls per variant; setup included."""
import json,time
import numpy as np
from check import HERE,Audit,Engine,outer_rule,prepare,evaluate,old

def main():
 start=time.process_time();ray=np.eye(6);y,rest,w=outer_rule(16,Audit());prefix=y@ray[:,:3].T;out=dict(rows=[])
 names=('baseline','raw','normalized')
 for rep in range(20):
  row=dict(repetition=rep)
  # Cyclic ordering prevents always granting a given implementation first/last.
  for name in names[rep%3:]+names[:rep%3]:
   c=time.process_time();t=time.perf_counter();engine=Engine(Audit());frame=prepare(ray[:,3:].T)
   if name=='baseline':v,s=old.evaluate(prefix,rest,frame,engine.positive)
   else:v,s=evaluate(prefix,rest,frame,engine.positive,normalize_edges=name=='normalized')
   row[name]=dict(cpu=time.process_time()-c,wall=time.perf_counter()-t,sum=float(v.sum()),fallbacks=s['positive_quadrature_fallbacks'])
  out['rows'].append(row)
 out['totals']={name:{field:sum(r[name][field] for r in out['rows']) for field in ('cpu','wall')} for name in names}
 out['cpu_seconds']=time.process_time()-start
 (HERE/'benchmark_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
