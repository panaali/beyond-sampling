"""Fixed H2T3 Gaussian closure conditioned on exact-law cone support (numerical geometry)."""
import time
START=time.process_time()
import hashlib,json,math,resource
from pathlib import Path
import numpy as np
import torch
import gate_mixture as G
import full_covariance_bounded as B
HERE=Path(__file__).resolve().parent

def stats():
    return dict(quad_calls=0,quad_evaluations=0,quad_error_estimate_sum=0.,singular_limits=0,
        outside_unit_corrections=[],singular_orthant_error_bounds=[],analytic_tail_bound_sum=0.)

def wedge(mu,cov,lo,hi,eps):
    s=stats()
    if hi<=lo:return 0.,s
    D=np.array([[-math.sin(lo),math.cos(lo)],[math.sin(hi),-math.cos(hi)]])
    p=B.orthant(D@mu,D@cov@D.T,[False,False],eps,s)
    return p,s

def clip(lo,hi,n,strict=False):
    a,b=n
    if a==0 and b==0:return (1.,0.) if strict else (lo,hi)
    if a>=0 and b>=0:return lo,hi
    if a<=0 and b<=0:return 1.,0.
    if a<0:lo=max(lo,math.atan2(-a,b))
    else:hi=min(hi,math.atan2(a,-b))
    return lo,hi

def err(s):
    return s['quad_error_estimate_sum']+s['analytic_tail_bound_sum']+sum(s['singular_orthant_error_bounds'])

def probabilities(mu,cov,Wo,bounds,eps):
    lo,hi=bounds;den,ds=wedge(mu,cov,lo,hi,eps)
    assert den>1e-10,('small support denominator',den)
    probs=[];rows=[]
    for j in range(len(Wo)):
        l,h=lo,hi
        for k in range(len(Wo)):
            if j!=k:l,h=clip(l,h,Wo[j]-Wo[k],k<j)
        num,ns=wedge(mu,cov,l,h,eps);p=num/den
        de,ne=err(ds),err(ns)
        bound=(ne+p*de)/(den-de) if den>de else float('inf')
        probs.append(p);rows.append(dict(numerator=num,bounds=[l,h],stats=ns,ratio_numerical_error_estimate=bound))
    return dict(q=probs,denominator=den,denominator_stats=ds,rows=rows,sum_residual=sum(probs)-1.)

def main():
    saved=json.loads((HERE/'gate_mixture.json').read_text())
    state=torch.load(G.T.CP,weights_only=True,map_location='cpu')
    M=(state['rnn.weight_ih_l0'].numpy()[:,0].astype(float),state['rnn.weight_hh_l0'].numpy().astype(float),state['linear.weight'].numpy().astype(float))
    Wo=M[2];cond=float(np.linalg.cond(Wo));assert cond<1e8
    inv=np.linalg.pinv(Wo);assert np.max(abs(inv@Wo-np.eye(2)))<1e-12
    begin=time.process_time();pieces,geom=G.groups(M);geom['cpu_seconds']=time.process_time()-begin
    assert len(pieces)==len(saved['components'])
    controls=[]
    for lo,hi in [(0.,math.pi/2),(.1,.7),(.6,1.2),(.1,.1001)]:
        p,s=wedge(np.zeros(2),np.eye(2),lo,hi,1e-12);truth=(hi-lo)/(2*math.pi)
        assert abs(p-truth)<1e-10
        controls.append(dict(bounds=[lo,hi],value=p,truth=truth,stats=s))
    control_W=np.array([[1.,0.],[0.,1.],[0.,0.]])
    c=probabilities(np.zeros(2),np.eye(2),control_W,[0,math.pi/2],1e-12)
    assert np.max(abs(np.array(c['q'])-[.5,.5,0.]))<1e-10
    assert clip(0,math.pi/2,[0,0],True)==(1.,0.)
    totals={'orthant':0.,'component_cone':0.};rows=[];fixed=0.
    for old,g in zip(saved['components'],pieces):
        assert (old['gate_cell'],old['order'],old['target'])==(g['gate_cell'],list(g['order']),g['target'])
        assert abs(old['mass']-g['mass'])<1e-12 and old['fixed_winner']==g['fixed_winner']
        mass=old['mass'];j=old['target']
        if old['fixed_winner'] is not None:
            val=mass*(old['fixed_winner']==j);fixed+=val
            for key in totals:totals[key]+=val
            continue
        K=old['cumulants'];mu=inv@np.array(K['1']);cov=inv@np.array(K['2'])@inv.T
        scale=max(float(np.max(abs(cov))),1e-300)
        asym=float(np.max(abs(cov-cov.T))/scale);assert asym<1e-12
        # Symmetrization is explicitly recorded; no eigenvalue clipping.
        cov=(cov+cov.T)/2;ev=np.linalg.eigvalsh(cov);assert ev[0]>0,('nonpositive residual covariance',ev)
        residual=max(float(np.max(abs(Wo@mu-np.array(K['1'])))),float(np.max(abs(Wo@cov@Wo.T-np.array(K['2'])))))
        assert residual<1e-10
        rays=np.concatenate([R@L.T@inv.T for _,R,L,_ in g['triangles']]);rs=max(1.,float(np.max(abs(rays))))
        negative=float(np.min(rays));assert negative>=-1e-12*rs
        # Tiny negative roundoff on known ReLU support is recorded, then set0.
        rays=np.maximum(rays,0);active=np.linalg.norm(rays,axis=1)>1e-14*rs
        assert np.any(active)
        angles=np.arctan2(rays[active,1],rays[active,0]);bounds=[float(min(angles)),float(max(angles))]
        assert bounds[1]>bounds[0]
        row=dict(gate_cell=g['gate_cell'],order=g['order'],target=j,mass=mass,hidden_mean=mu.tolist(),hidden_covariance=cov.tolist(),
            covariance_eigenvalues=ev.tolist(),covariance_relative_asymmetry=asym,reconstruction_residual=residual,
            hidden_support_rays=rays.tolist(),minimum_unclipped_hidden_ray=negative,component_bounds=bounds,methods={})
        for method,interval in [('orthant',[0.,math.pi/2]),('component_cone',bounds)]:
            coarse=probabilities(mu,cov,Wo,interval,1e-10);fine=probabilities(mu,cov,Wo,interval,1e-12)
            scales=[probabilities(mu*s,cov*s*s,Wo,interval,1e-12) for s in [1e-8,1e8]]
            rule=float(np.max(abs(np.array(coarse['q'])-fine['q'])))
            rescale=max(float(np.max(abs(np.array(v['q'])-fine['q']))) for v in scales)
            assert abs(fine['sum_residual'])<1e-9 and rescale<1e-9
            row['methods'][method]=dict(coarse=coarse,fine=fine,scales=scales,rule_sensitivity=rule,scale_sensitivity=rescale)
            totals[method]+=mass*fine['q'][j]
        rows.append(row)
        assert time.process_time()-START<20
    out=dict(complete=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256((HERE/'gate_mixture.json').read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(Path(G.T.CP).read_bytes()).hexdigest(),
        controls=controls,partition_control=c,condition_number_Wo=cond,geometry=geom,constant_correct_mass=fixed,rows=rows,
        truth=saved['truth'],accuracy=totals,bias={k:v-saved['truth'] for k,v in totals.items()},
        cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        qualifications='Numerical topology, moment quadrature, special functions, covariance symmetrization and tiny-ray clipping remain uncertified. Error estimates are conditional numerical diagnostics, not global enclosures. Cold cost includes prior geometry/moments; endpoint scalar proxy incomplete.')
    (HERE/'support_conditioned.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['rows','controls','partition_control']},indent=2))
if __name__=='__main__':main()
