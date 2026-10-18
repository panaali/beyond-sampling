"""Same-panel attribution of the seven exact radial discrepancies."""
from pathlib import Path
import json,math,time
import numpy as np
from radial_probe import cases,scalar_radial,vector_radial


def scalar_stages(v):
    total=v.sum(axis=0);norm=math.sqrt(float(total@total));center=total/norm
    projection=float(np.min(v@center));square=projection**2
    signs=np.sign(v.sum(axis=0));signproj=float(np.min(v@signs));signnorm=float(signs@signs)
    signbound=signproj*signproj/signnorm
    hi=float(np.max(np.sum(v*v,axis=1)))
    return dict(total=total,norm=norm,center=center,projection=projection,square=square,
                sign_projection=signproj,sign_norm=signnorm,sign_bound=signbound,hi=hi)


def vector_stages(v):
    total=v.sum(axis=1);norm=np.sqrt(np.matmul(total[:,None,:],total[:,:,None])[:,0,0]);center=total/norm[:,None]
    projection=np.min(np.matmul(v,center[:,:,None])[:,:,0],axis=1);square=projection**2
    signs=np.sign(v.sum(axis=1));signproj=np.min(np.matmul(v,signs[:,:,None])[:,:,0],axis=1)
    signnorm=np.matmul(signs[:,None,:],signs[:,:,None])[:,0,0]
    signbound=signproj*signproj/signnorm;hi=np.max(np.sum(v*v,axis=2),axis=1)
    return dict(total=total,norm=norm,center=center,projection=projection,square=square,
                sign_projection=signproj,sign_norm=signnorm,sign_bound=signbound,hi=hi)


def hybrid_radial(v):
    a=vector_stages(v)
    square=np.fromiter((float(x)**2 for x in a['projection']),dtype=float,count=len(v))
    lo=np.where(a['sign_projection']>0,np.maximum(square,a['sign_bound']),square)
    hi=a['hi'];lo=np.minimum(lo,hi)
    return np.stack((lo,hi,(lo+hi)/2,(hi-lo)/(hi+lo)),axis=1)


if __name__=='__main__':
    start=time.process_time();v=cases();ss=[scalar_stages(x) for x in v];vv=vector_stages(v);rows={}
    for name,value in vv.items():
        reference=np.array([r[name] for r in ss]);different=reference!=value
        rows[name]=dict(different_elements=int(different.sum()),maximum_absolute_difference=float(np.max(abs(reference-value))))
    old=np.array([scalar_radial(x) for x in v]);hybrid=hybrid_radial(v);vector=vector_radial(v)
    pure_bad=np.flatnonzero(np.any(old!=vector,axis=1));hybrid_bad=np.flatnonzero(np.any(old!=hybrid,axis=1))
    only_square=all(row['different_elements']==0 for name,row in rows.items() if name!='square')
    out=dict(cases=len(v),stages=rows,pure_vector_mismatch_indices=pure_bad.tolist(),
             hybrid_mismatch_indices=hybrid_bad.tolist(),only_square_differs=only_square,
             isolated_rescue_gate_pass=only_square and len(hybrid_bad)==0,
             cpu_seconds=time.process_time()-start)
    (Path(__file__).resolve().parent/'attribution.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))
