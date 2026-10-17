"""Positive arc quadrature with an explicit integration-only remainder."""
import math,time
import numpy as np
from recurrence import add,baseline,module_for

C=math.factorial(8)**4/(17*math.factorial(16)**3)

class ArcRule:
    def __init__(self,tolerance=1e-13):
        t=time.process_time()
        self.x,self.w=np.polynomial.legendre.leggauss(8)
        self.tolerance=tolerance;self.generation_cpu=time.process_time()-t
        if not np.all(self.w>0):raise ArithmeticError('nonpositive rule weight')

    def evaluate(self,a,b,theta,degree,stats=None):
        if degree<0 or degree%2 or degree>6 or not 0<=theta<=math.pi:raise ValueError('supported even degree0..6, minor arc required')
        add(stats,arc_chains=1,arc_values=degree//2+1)
        if not degree:return [theta],[0.]
        panels=max(1,math.ceil(theta*degree*(C/self.tolerance)**(1/16)))
        width=theta/panels
        nodes=((np.arange(panels)[:,None]+.5)+self.x[None,:]/2)*width
        weights=np.broadcast_to(self.w[None,:]*(width/2),nodes.shape)
        values=a*np.cos(nodes)+b*np.sin(nodes);squares=values*values
        power=np.ones_like(values);answer=[theta];errors=[0.];r=math.hypot(a,b)
        for n in range(2,degree+1,2):
            power*=squares;answer.append(float(np.sum(weights*power)))
            errors.append(C*theta*(width*n)**16*r**n)
        N=8*panels;steps=degree//2
        add(stats,arc_panels=panels,arc_nodes=N,arc_trig=2*N,arc_hypot_calls=1,
            arc_scalar_arithmetic=3*panels+4*N+5*N+steps*(3*N-1),
            arc_rule_error_power_calls=1+2*steps)
        return answer,errors

    def moment(self,vertices,degree,stats=None):
        if degree<1 or degree%2!=1:raise ValueError('positive odd degree required')
        vs=np.array([baseline.unit(v) for v in vertices])
        if len(vs)<3 or np.min(vs[:,0]) < -1e-14:raise ValueError('invalid positive-hemisphere polygon')
        center=vs.sum(axis=0);steps=(degree+1)//2
        boundary=np.zeros(steps);error=np.zeros(steps)
        for u,v in zip(vs,np.roll(vs,-1,axis=0)):
            cross=np.cross(u,v);length=np.linalg.norm(cross)
            if not length:raise ValueError('coincident or antipodal consecutive rays')
            inward=cross/length;side=inward@center
            if not side:raise ValueError('degenerate polygon')
            inward*=np.sign(side);outward0=-inward[0]
            theta=np.arctan2(length,u@v);tangent=np.cross(cross/length,u)
            vals,errs=self.evaluate(float(u[0]),float(tangent[0]),float(theta),degree-1,stats)
            boundary+=outward0*np.array(vals);error+=abs(outward0)*np.array(errs)
        value=err=0.
        for i,p in enumerate(range(1,degree+1,2)):
            value=((p-1)*value-boundary[i])/(p+1)
            err=((p-1)*err+error[i])/(p+1)
        add(stats,polygon_calls=1,polygon_edges=len(vs),polygon_recurrence_steps=steps,
            tail_arc_integration_error_sum=err*(degree+1)/(2*math.pi))
        return value

    def module(self):
        mod=module_for(lambda *args:None)
        mod.polygon_moment=self.moment
        return mod
