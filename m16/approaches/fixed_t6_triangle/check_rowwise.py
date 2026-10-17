"""Frozen512-row scalar/batch discriminator plus unchanged initial suite."""
import json,time
import numpy as np
import check
from check import HERE,Audit,Engine,outer_rule
import rowwise

def main():
 start=time.process_time();rays=np.array(json.loads((HERE.parent/'directed_chebyshev/t6_fixture.json').read_text())['rays']);ray=rays[2]
 y,rest,w=outer_rule(8,Audit());prefix=y@ray[:,:3].T;frame=rowwise.prepare(ray[:,3:].T);engine=Engine(Audit())
 batched,stats=rowwise.evaluate(prefix,rest,frame,engine.positive)
 scalar=[];fallbacks=0
 for i in range(len(rest)):
  value,s=rowwise.evaluate(prefix[i:i+1],rest[i:i+1],frame,engine.positive);scalar.append(value[0]);fallbacks+=s['positive_quadrature_fallbacks']
 scalar=np.array(scalar)
 result=dict(source_sha=rowwise.SOURCE_SHA,max_batch_scalar_relative=float(np.max(np.abs(batched/scalar-1))),bit_identical=bool(np.array_equal(batched,scalar)),batch_fallbacks=stats['positive_quadrature_fallbacks'],scalar_fallbacks=fallbacks,node447_batch=float(batched[447]),node447_scalar=float(scalar[447]),cpu_seconds=time.process_time()-start)
 # Explicitly redirect the old suite's output without changing its frozen source.
 original=(HERE/'control_results.json').read_bytes()
 check.evaluate=rowwise.evaluate;check.prepare=rowwise.prepare
 try:
  check.main();(HERE/'rowwise_control_results.json').write_bytes((HERE/'control_results.json').read_bytes())
 finally:(HERE/'control_results.json').write_bytes(original)
 result['suite_cpu_seconds']=json.loads((HERE/'rowwise_control_results.json').read_text())['cpu_seconds']
 result['total_cpu_seconds']=time.process_time()-start
 (HERE/'rowwise_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
