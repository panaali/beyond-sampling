"""Frozen finite-representation inventory; no quadrature or author calls."""
import hashlib,json,time,resource
from pathlib import Path
import numpy as np
from ..cone_mass4_review.check_rotation import rotate_to_orthant
HERE=Path(__file__).parent
CAPTURE=HERE.parent/'class_region_t4/quadratic_exact_mass_calls.npz'
HASH='07a0720e229bd2e0aeda511b60038efdc37f76c17aadfdeb17cf0caf3f264581'
def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,22));start=time.process_time();assert hashlib.sha256(CAPTURE.read_bytes()).hexdigest()==HASH
    rays=np.load(CAPTURE)['rays'];transformed=[];rows=[];uncovered=[];counts=dict(native=0,rotated=0,uncovered=0)
    for i,R in enumerate(rays):
        if np.all((R.min(0)>=0)|(R.max(0)<=0)):
            Y=R;meta=dict(attempts=0,rotation_arithmetic=0);mode='native'
        else:
            Y,meta=rotate_to_orthant(R);mode='rotated' if Y is not None else 'uncovered'
        counts[mode]+=1
        if Y is None:uncovered.append(i);transformed.append(np.full((4,4),np.nan))
        else:transformed.append(abs(Y))
        rows.append(dict(index=i,mode=mode,**meta))
    out=dict(capture_sha256=HASH,counts=counts,uncovered=uncovered,rows=rows,cpu=time.process_time()-start,rotation_arithmetic=sum(r['rotation_arithmetic'] for r in rows),full_cost_certified=False)
    np.savez_compressed(HERE/'represented_rays.npz',rays=np.array(transformed));(HERE/'inventory.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('rows','uncovered')},indent=2))
if __name__=='__main__':main()
