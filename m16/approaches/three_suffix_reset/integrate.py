"""Positive three-suffix rules; complete older-label rank contribution retained."""
import time
START=time.process_time()
import hashlib,json,math,resource,sys
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from scipy.special import ndtr,owens_t,roots_legendre,erfc
import topology as G
P=Path(__file__).resolve().parent
DATA=json.loads((P/'topology.json').read_text());K=float(F(DATA['kappa']));Q=math.atan(K)/(2*math.pi);RESET=.5-Q;N=3
COUNT=dict(owens_t=0,normal_cdf=0,older_target_nodes=0,negative_roundoff_clamps=0)
def rule(n,a=0.,b=1.):
    x,w=roots_legendre(n);return (a+b)/2+(b-a)/2*x,w*(b-a)/2

def pair(t):
    u=ndtr(t);sf=ndtr(-t);ot=owens_t(t,K);g0=.5*u+ot-Q;g1=.5*sf-ot
    COUNT['owens_t']+=t.size;COUNT['normal_cdf']+=2*t.size;COUNT['negative_roundoff_clamps']+=int(np.count_nonzero(g1<0))
    assert np.min(g0)>=-1e-14 and np.min(g1)>=-1e-14
    return u,sf,np.maximum(g0,0),np.maximum(g1,0)

def older(suffix,n=32):
    # Flatten batches, cap memory independently of angular rule size.
    shape=suffix.shape[:-1];v=suffix.reshape(-1,3);out=np.empty(len(v));gx,gw=rule(n)
    for start in range(0,len(v),512):
        z=np.sort(v[start:start+512],axis=-1);S=z[:,1];L=z[:,2];assert np.min(S)>=0 and np.max(L)<=8+1e-12
        total=np.zeros(len(z))
        for lo,hi,band in ((S,L,True),(L,np.full_like(L,8.),False)):
            t=lo[:,None]+(hi-lo)[:,None]*gx;u,sf,g0,g1=pair(t)
            f=u**(N-1)*g0 if band else (N-1)*sf*u**(N-2)*g0+u**(N-1)*g1
            total+=(hi-lo)*((np.exp(-t*t/2)/math.sqrt(2*math.pi)*f)@gw);COUNT['older_target_nodes']+=t.size
        out[start:start+len(z)]=total
    return out.reshape(shape)

def rank_suffix(suffix,label,order):
    if label==0:return older(suffix)
    j=label-5;t=suffix[...,j];assert np.min(t)>=0
    above=sum(i!=j and order.index(i)>order.index(j) for i in range(3))
    if above>=2:return np.zeros_like(t)
    u,sf,g0,g1=pair(t)
    return u**N*g0 if above==1 else N*sf*u**(N-1)*g0+u**N*g1

def cells():
    out=[]
    for row in DATA['cells']:
        polys=[[tuple(F(v) for v in p) for p in row['vertices']]];sign=row['sign']
        for i,j in ((0,1),(0,2),(1,2)):
            plane=[F(0)]*3;plane[i]=sign[i];plane[j]=-sign[j];new=[]
            for poly in polys:
                for normal in (plane,[-v for v in plane]):
                    c=G.clip(poly,normal)
                    if c:new.append(c)
            polys=new
        for poly in polys:
            center=[sum(p[i] for p in poly)*sign[i]/len(poly) for i in range(3)];order=sorted(range(3),key=lambda i:center[i])
            out.append(dict(vertices=poly,label=row['label'],sign=sign,order=order))
    assert len(out)<=256
    return out

def triangle_mass(v):
    norms=np.linalg.norm(v,axis=1);det=abs(np.linalg.det(v))
    denominator=np.prod(norms)+np.dot(v[0],v[1])*norms[2]+np.dot(v[0],v[2])*norms[1]+np.dot(v[1],v[2])*norms[0]
    return 2*math.atan2(det,denominator)/(4*math.pi)

def integrate(rankcells,n,correct=True):
    tick=time.process_time();gx,gw=rule(n);rad,rw=rule(32,0.,8.);rw=rw*rad**2*np.exp(-rad*rad/2)/(2*math.pi)**1.5
    u,v=np.meshgrid(gx,gx,indexing='ij');ww=np.outer(gw,gw)*(1-u);bylabel={};mass=0.;solid=0.;triangles=0
    for cell in rankcells:
        vv=np.array(cell['vertices'],float);sign=np.array(cell['sign'])
        for i in range(1,len(vv)-1):
            tri=vv[[0,i,i+1]];det=abs(np.linalg.det(np.stack([tri[1,:2]-tri[0,:2],tri[2,:2]-tri[0,:2]])))
            y=tri[0]+u[...,None]*(tri[1]-tri[0])+(1-u[...,None])*v[...,None]*(tri[2]-tri[0]);norm=np.linalg.norm(y,axis=-1)
            weight=ww*det/norm**3;mass+=float(np.sum(weight)*np.sum(rw))*RESET;solid+=triangle_mass(tri)*RESET;triangles+=1
            if correct:
                suffix=y[...,None,:]*sign*rad[None,None,:,None]/norm[...,None,None]
                f=rank_suffix(suffix,cell['label'],cell['order']);value=float(np.sum(weight*np.einsum('ijk,k->ij',f,rw)));bylabel[cell['label']]=bylabel.get(cell['label'],0.)+value
            if time.process_time()-START>90:raise TimeoutError('frozen CPU cap')
    return dict(order=n,contribution=sum(bylabel.values()),bylabel=bylabel,event_mass_rule=mass,event_mass_solid=solid,event_mass_analytic=RESET*(.125+Q),triangles=triangles,cpu_seconds=time.process_time()-tick)

def main():
    # Different generic target implementation integrates the reset pair adaptively.
    sys.path.insert(0,'/workspace/work/recover-small_h2_34')
    from m16.approaches.reset_wedge.conditional import conditional_rank
    controls=[]
    for values in ((.1,.2,.3),(1.,-.01,.5),(.5,1.,-.01),(.01,.02,.03),(3.,5.,7.),(7.,8.,.1)):
        x=np.array([values]);a=float(older(x,32)[0]);b=float(older(x,64)[0]);ref,stats=conditional_rank(K,N,values,0)
        controls.append(dict(suffix=values,rule32=a,rule64=b,reference=ref,difference=a-ref,ref_stats=stats));assert abs(a-b)<1e-10 and abs(a-ref)<1e-10
    rankcells=cells();geometry=integrate(rankcells,32,False)
    assert abs(geometry['event_mass_solid']-geometry['event_mass_analytic'])<1e-13
    assert abs(geometry['event_mass_rule']-geometry['event_mass_analytic'])<1e-7
    out=dict(complete=False,controls=controls,rank_cells=len(rankcells),geometry=geometry,runs=[],topology_sha256=hashlib.sha256((P/'topology.json').read_bytes()).hexdigest())
    for n in (8,16,32):
        run=integrate(rankcells,n);out['runs'].append(run);out['counts']=COUNT.copy();out['cpu_seconds_including_imports']=time.process_time()-START
        (P/'integration.json').write_text(json.dumps(out,indent=2)+'\n');print(n,run,flush=True)
    out.update(complete=True,radial_tail_allowance=RESET*float(erfc(8/math.sqrt(2))+math.sqrt(2/math.pi)*8*math.exp(-32)),older_target_tail_allowance=RESET*float(ndtr(-8.)),cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (P/'integration.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback
        (P/'integration_failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),counts=COUNT,cpu_seconds=time.process_time()-START),indent=2)+'\n');raise
