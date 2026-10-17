"""Independent MP comparison of T71's fixed tangent optimization."""
from pathlib import Path
import sys
import time
import hashlib
import json
import numpy as np
import mpmath as mp

HERE=Path(__file__).resolve().parent
SOURCE=Path('/workspace/work/recover-small_h2_56/m16/approaches/small_h2_56/odd_tetra')

def main():
    started=time.process_time();mp.mp.dps=70
    sys.path.insert(0,str(SOURCE))
    from fixed_tangent import fixed_tangent_tetra
    p=np.array([.8,.05,.05,.05,.05])
    rays=np.vstack((p,np.eye(5)[:4]))[None]
    nodes=np.array([.99,.999])
    values,stats=fixed_tangent_tetra(rays,nodes)
    permuted,_=fixed_tangent_tetra(rays[:,[0,4,2,1,3]],nodes)
    roots,weights=mp.gauss_quadrature(12,'legendre')
    roots=[(x+1)/2 for x in roots];weights=[w/2 for w in weights]
    cases=[]
    for index,node in enumerate(nodes):
        endpoints=node*rays[0,0]+(1-node)*rays[0,1:]
        e=[[mp.mpf(float(x)) for x in row] for row in endpoints]
        reference=mp.mpf(0)
        for u,wu in zip(roots,weights):
            for v,wv in zip(roots,weights):
                for z,wz in zip(roots,weights):
                    y=(u,(1-u)*v,(1-u)*(1-v)*z,(1-u)*(1-v)*(1-z))
                    q=sum(sum(y[j]*e[j][a] for j in range(4))**2 for a in range(5))
                    reference+=wu*wv*wz*(1-u)**2*(1-v)/q**mp.mpf('2.5')
        cases.append(dict(outer_node=float(node),value=float(values[0,index]),reference=str(reference),
                          relative_error=float(abs(mp.mpf(float(values[0,index]))/reference-1)),
                          permutation_relative_difference=float(abs(permuted[0,index]/values[0,index]-1))))
    result=dict(cpu_seconds=time.process_time()-started,high_precision_nodes=2*12**3,
                precision_decimal_digits=70,model_passes=0,cases=cases,
                stats={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in stats.items()},
                source_hashes={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ('fixed_tangent.py','primitive.py')})
    (HERE/'fixed_tangent_review_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
