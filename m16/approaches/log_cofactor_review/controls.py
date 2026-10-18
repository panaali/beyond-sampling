import time
START=time.process_time()
import itertools,json,math,resource
import numpy as np
import audit as A

def main():
 A.verify_pins();rows=[];arrays={}
 cases=[('orthant',np.eye(4),1/16)]
 for signs in itertools.product([-1,1],repeat=4):
  s=np.array(signs);C=(.5*np.eye(4)+.5*np.ones((4,4)))*np.outer(s,s);k=sum(x<0 for x in signs)
  cases.append(('equicorrelation_'+''.join('m' if x<0 else 'p' for x in signs),C,1/(5*math.comb(4,k))))
 for label,C,p in cases:
  R=np.linalg.inv(np.linalg.cholesky(C)).T;row,a=A.check_case(R,label,p);rows.append(row)
  for key,v in a.items():arrays[label+'_'+key]=v
 fixture=json.loads(next(A.Path(f) for f in A.PROTOCOL['pins'] if f.endswith('negative_mass_fixture.json')).read_text())
 row,a=A.check_case(np.array(fixture['rays']),'fixture672',fixture['positive'],fixture['bound']);rows.append(row)
 for key,v in a.items():arrays['fixture672_'+key]=v
 out=dict(rows=rows,gate_pass=all(r['parity'] and np.isfinite(r['value']) and abs(r['difference'])<=1e-10+r['reference_bound'] for r in rows),cpu=time.process_time()-START,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 np.savez_compressed(A.HERE/'control_nodes.npz',**arrays);(A.HERE/'controls.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(gate_pass=out['gate_pass'],maxerror=max(abs(r['difference']) for r in rows),fixture=rows[-1],cpu=out['cpu'])))
if __name__=='__main__':main()
