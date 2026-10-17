"""Frozen primitive controls and eight bounded trained face slices."""
import hashlib,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from .slices import Slice,clip,area,vertical_mass,walker,ROOT,WALKER

HERE=Path(__file__).resolve().parent


def reference_values(model,kind,configuration,q):
    T=len(model[2]); count=len(walker.geometry.configs(T,kind))
    U=np.full((len(q),T),.5);U[:,0]=(configuration+.5)/count;U[:,2]=q
    y,a,b,tau=walker.face_points(T,kind,U)
    value,cost,cells=walker.walk_mass(model,y,a,b,tau,kind)
    return value,dict(faces=len(q),walker_fpe_proxy=float(cost.sum()),walker_cells=int(cells.sum()))


def controls():
    start=time.process_time(); work=Counter()
    square=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    triangle=clip(square,np.array([-1.,-1.,1.]),work)
    assert abs(area(triangle)-.5)<1e-15
    strip=clip(clip(square,np.array([1.,0.,-.5]),work),np.array([-1.,0.,.5+1e-12]),work)
    assert area(strip)>0 and abs(area(strip)-1e-12)<1e-16
    wi=np.ones(2);wh=np.diag([1.,2.]);zero=np.zeros((4,2))
    duplicate=np.array([[1.,1.],[1.,1.],[0.,0.],[-1.,-1.]])
    q=.05+.9*(np.arange(17)+.37)/17
    records=[]
    for name,wo in [('zero',zero),('duplicate',duplicate)]:
        model=(wi,wh,wo)
        for kind in ('23','12'):
            s=Slice(model,kind,0)
            actual=s.values(q);ref,cost=reference_values(model,kind,0,q)
            total=vertical_mass([s.domain],s.toz(q),Counter())
            defect=float(np.max(abs(actual-total/2)))
            assert defect<1e-12 and np.max(abs(actual-ref))<1e-8
            assert s.state_defect<1e-10
            records.append(dict(model=name,kind=kind,tie_defect=defect,
                                walker_defect=float(np.max(abs(actual-ref))),
                                partition_area_defect=abs(sum(area(p) for p,B in s.cells)-area(s.domain)),
                                work=dict(s.work),reference_work=cost))
    return dict(triangle_area=area(triangle),thin_strip_area=area(strip),controls=records,
                clipping_work=dict(work),cpu_seconds=time.process_time()-start)


def main():
    start=time.process_time();out=dict(controls=controls(),cases=[])
    paths=[HERE/'slices.py',WALKER,ROOT/'m16/approaches/sci_tao/core.py']
    out['sources']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    def save():
        out['cpu_seconds']=time.process_time()-start
        (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n')
    save()
    import torch
    for H in (2,4):
        checkpoint=ROOT/f'work/checkpoints/H{H}_T4_seed0.pth'
        st=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model=(st['rnn.weight_ih_l0'].numpy().astype(float).ravel(),
               st['rnn.weight_hh_l0'].numpy().astype(float),st['linear.weight'].numpy().astype(float))
        for kind in ('23','12'):
            count=len(walker.geometry.configs(4,kind))
            for config in (0,count-1):
                if time.process_time()-start>105: raise TimeoutError('T82 CPU guard')
                cold=time.process_time();s=Slice(model,kind,config)
                construction_cpu=time.process_time()-cold;construction_work=dict(s.work)
                q=.05+.9*(np.arange(17)+.37)/17
                values=s.values(q);ref,refwork=reference_values(model,kind,config,q)
                row=dict(H=H,T=4,kind=kind,configuration=config,
                         checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                         construction_cpu_seconds=construction_cpu,construction_work=construction_work,
                         pointwise_defect=float(np.max(abs(values-ref))),reference_point_work=refwork,
                         centroid_state_defect=s.state_defect,
                         split_area_defect=max(s.area_defects,default=0),
                         total_area_defect=abs(sum(area(p) for p,B in s.cells)-area(s.domain)),
                         knots=s.knots.tolist(),geometry_knots=s.geometry_knots.tolist())
                out['cases'].append(row);save()
                assert row['pointwise_defect']<1e-8, row
                assert row['centroid_state_defect']<1e-9 and row['total_area_defect']<1e-9
                schemes=[('network_GL4',s.knots,4),('equal_count_GL4',np.linspace(.05,.95,len(s.knots)),4),
                         ('geometry_GL4',np.unique(np.r_[.05,s.geometry_knots,.95]),4),
                         ('reference_GL16',s.knots,16),('reference_GL32',s.knots,32)]
                estimates={}; costs={}
                for name,knots,order in schemes:
                    before=Counter(s.work);t=time.process_time()
                    estimates[name]=s.integrate(knots,order)
                    costs[name]=dict(cpu_seconds=time.process_time()-t,work=dict(s.work-before))
                row.update(estimates=estimates,integration_costs=costs,
                           reference_disagreement=abs(estimates['reference_GL16']-estimates['reference_GL32']),
                           errors={k:v-estimates['reference_GL32'] for k,v in estimates.items()})
                save();print(json.dumps({k:row[k] for k in ('H','kind','configuration','pointwise_defect','errors','construction_cpu_seconds')}),flush=True)
    out['complete']=True;save()


if __name__=='__main__':main()
