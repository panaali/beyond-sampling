import json,time
from collections import Counter
import numpy as np
from .slices import clip,area,signed_area,vertical_mass,Slice,ROOT
from .check_slices import HERE,reference_values


def main():
    start=time.process_time()
    import torch
    p=np.array([[.6815355480744857,-.5137321224031886],
                [.6815355480744858,-.5137321224031886],[2.370139237852187,-.5773595518832472]])
    work=Counter();fixed=clip(p,np.array([0.,0.,1.]),work)
    assert signed_area(p)<0 and signed_area(fixed)>0 and area(p)==area(fixed)
    mass=vertical_mass([fixed],np.linspace(p[:,0].min(),p[:,0].max(),19)[1:-1],work)
    assert mass.min()>-1e-14
    square=np.array([[0.,0.],[0.,1.],[1.,1.],[1.,0.]])
    triangle=clip(square,np.array([-1.,-1.,1.]),work)
    assert abs(signed_area(triangle)-.5)<1e-15
    st=torch.load(ROOT/'work/checkpoints/H2_T4_seed0.pth',map_location='cpu',weights_only=True)
    model=(st['rnn.weight_ih_l0'].numpy().astype(float).ravel(),st['rnn.weight_hh_l0'].numpy().astype(float),st['linear.weight'].numpy().astype(float))
    nodes,_=np.polynomial.legendre.leggauss(5);nodes=(nodes+1)/2
    s=Slice(model,'23',10,qlo=1e-6,qhi=1-1e-6,fixed_quantiles=[.5,nodes[1],.5,nodes[0]])
    value,norm=s.integrate_gaussian(s.knots,16)
    q=1e-6+(1-2e-6)*(np.arange(17)+.37)/17
    # Independent walker at the actual fixed outer coordinates.
    from .slices import walker
    U=np.tile([.5,nodes[1],.5,nodes[0]],(len(q),1));U[:,0]=10.5/12;U[:,2]=q
    y,a,b,tau=walker.face_points(4,'23',U);ref,cost,cells=walker.walk_mass(model,y,a,b,tau,'23')
    defect=float(np.max(abs(s.values(q)-ref)));assert defect<1e-8
    out=dict(recorded_signed_area=signed_area(p),retained_area=area(fixed),sliver_mass_min=float(mass.min()),
             primitive_work=dict(work),slice_integral=value,weight_normalization_defect=norm-(1-2e-6),
             slice_pointwise_defect=defect,slice_work=dict(s.work),walker_fpe_proxy=float(cost.sum()),
             cpu_seconds=time.process_time()-start)
    st=torch.load(ROOT/'work/checkpoints/H4_T4_seed0.pth',map_location='cpu',weights_only=True)
    model=(st['rnn.weight_ih_l0'].numpy().astype(float).ravel(),st['rnn.weight_hh_l0'].numpy().astype(float),st['linear.weight'].numpy().astype(float))
    nodes,_=np.polynomial.legendre.leggauss(3);nodes=(nodes+1)/2
    s=Slice(model,'23',7,qlo=1e-6,qhi=1-1e-6,fixed_quantiles=[.5,nodes[0],.5,nodes[1]])
    value,norm=s.integrate_gaussian(s.knots,16)
    U=np.tile([.5,nodes[0],.5,nodes[1]],(len(q),1));U[:,0]=7.5/12;U[:,2]=q
    y,a,b,tau=walker.face_points(4,'23',U);ref,cost,cells=walker.walk_mass(model,y,a,b,tau,'23')
    defect=float(np.max(abs(s.values(q)-ref)));assert defect<1e-8
    out['nonconvex_fixture']=dict(slice_integral=value,pointwise_defect=defect,work=dict(s.work),
                                  walker_fpe_proxy=float(cost.sum()))
    out['cpu_seconds']=time.process_time()-start
    (HERE/'orientation_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':main()
