"""Independent one-sided line-density and area-continuity jump checks."""
from copy import copy
from fractions import Fraction as F
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from density_shape import ShapeOperator
from mixed_volume import PHI
from two_input import Law

HERE=Path(__file__).resolve().parent

def boundary_density(cell,L):
    a=next(row for row in cell['B'] if any(row));norm=math.sqrt(sum(float(v)**2 for v in a))
    constraints=[((F(1),F(0)),F(0)),((F(-1),F(0)),-F(L)),
        ((F(0),F(1)),-F(L)),((F(0),F(-1)),-F(L))]
    if cell['lo'] is not None:constraints.append(((-cell['lo'],F(1)),F(0)))
    if cell['hi'] is not None:constraints.append(((cell['hi'],F(-1)),F(0)))
    lo,hi=-math.inf,math.inf
    for c,bound in constraints:
        coefficient=-c[0]*a[1]+c[1]*a[0]
        if coefficient>0:lo=max(lo,float(bound)*norm/float(coefficient))
        elif coefficient<0:hi=min(hi,float(bound)*norm/float(coefficient))
        elif bound>0 or (bound==0 and sum(c[i]*a[i] for i in (0,1))<0):return 0.
    return float(PHI(0))/norm*max(0.,ndtr(hi)-ndtr(lo))

def main():
    started=time.process_time();deadline=started+20
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth')
    data=torch.load(path,map_location='cpu',weights_only=True)
    trained=tuple(data[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    trained=(trained[0].reshape(-1),trained[1],trained[2])
    toy=(np.ones(2),np.diag([1.,2.]),np.array([[0.,0.],[2.,-1.],[-1.,2.],[.5,.5]]))
    paths=[HERE/s for s in ('check_density_jump.py','density_shape.py','two_input.py','support_volume.py','mixed_volume.py')]
    sources=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result={'classification':'line-source density jump and represented area continuity diagnostic',
        'random_model_inputs':0,'sources':sources(),'rows':[]}
    for name,model in (('toy',toy),('trained_H2T8',trained)):
        clock=time.process_time();op=ShapeOperator(model,16,shape='uniform');law=Law(model)
        w,W,_=model;n=np.array([-w[1],w[0]]);unit=n/np.linalg.norm(n)
        limits=[sum(boundary_density(c,6) for c in law.cells if c['kind']=='axis'+str(i)) for i in (0,1)]
        limits.append((.5-ndtr(-6))*float(PHI(0)))
        coefficients=[float(np.sign(n@(W@d)))*g/det for d,g,det in zip(op.directions,limits,op.line_det)]
        total=sum(coefficients)
        state=op.initial()
        for _ in range(2):state,_=op.step(state)
        area=copy(state);area.origin=0.;area.axes=np.zeros_like(state.axes);area.ray_factor=0.
        def line_density(h):
            values=[]
            for i,inverse in enumerate(op.line_inverse):
                r,x=inverse@h
                if r<=0 or abs(x)>6:values.append(0.);continue
                if i<2:
                    point=np.zeros((1,2));point[0,i]=r;g=float(op.density('axis'+str(i),point)[0])
                else:g=(.5-ndtr(-6))*float(PHI(r)) if r<=6 else 0.
                values.append(g*float(PHI(x))/op.line_det[i])
            return values
        checks=[]
        for s in (.5,1.):
            predicted=total*float(PHI(s))
            for epsilon in (1e-3,1e-5,1e-7):
                if time.process_time()>deadline:raise TimeoutError('density jump deadline')
                plus=s*w+epsilon*unit;minus=s*w-epsilon*unit
                left=line_density(minus);right=line_density(plus)
                area_left=op.area_density(area,minus);area_right=op.area_density(area,plus)
                observed=sum(right)-sum(left)
                checks.append({'s':s,'epsilon':epsilon,'line_plus':right,'line_minus':left,'line_difference':observed,
                    'predicted_limit':predicted,'line_limit_error':observed-predicted,
                    'represented_area_plus':area_right,'represented_area_minus':area_left,
                    'represented_area_difference':area_right-area_left})
                if epsilon==1e-7:
                    assert abs(observed-predicted)<1e-5*max(1,abs(predicted))
                    assert abs(area_right-area_left)<1e-5
        row={'model':name,'boundary_amplitude_densities':limits,'signed_jump_coefficients':coefficients,
            'net_coefficient':total,'checks':checks,'work':dict(op.work),'exact_gate_work':dict(law.work),
            'cpu_seconds':time.process_time()-clock}
        result['rows'].append(row);print(json.dumps({'model':name,'coefficient':total,'jump_at_s1':total*float(PHI(1))}),flush=True)
    result['cpu_seconds_including_torch_import']=time.process_time()-started
    result['sources_after']=sources();assert result['sources']==result['sources_after']
    (HERE/'density_jump_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'total_cpu':result['cpu_seconds_including_torch_import']}))

if __name__=='__main__':main()
