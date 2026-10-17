"""Positive variable-order source quadrature with unchanged cell projection."""
import numpy as np
from scipy.special import ndtr
from mixed_volume import PHI
from support_volume import SupportOperator


class QuadratureOperator(SupportOperator):
    def __init__(self,model,n,order=2,L=6.):
        super().__init__(model,n,L)
        q,w=np.polynomial.legendre.leggauss(order)
        self.q=(1+q)/2;self.qw=w/2;self.order=order
        self.work['source_rule_constructions']+=1

    def source_reweight(self,kind,points,weights):
        return weights

    def area_rule(self,state,i,j):
        q,qw=self.q,self.qw
        if state.support_kind in ('upper','lower') and not state.full_inside[i,j]:
            poly=self.area_polygon(state,i,j);points=[];raw=[]
            for k in range(1,len(poly)-1):
                a,b,c=poly[0],poly[k],poly[k+1]
                determinant=abs(np.linalg.det(np.column_stack((b-a,c-a))))
                if not determinant:continue
                for s,ws in zip(q,qw):
                    for t,wt in zip(q,qw):
                        points.append((1-s)*a+s*((1-t)*b+t*c));raw.append(determinant*s*ws*wt)
                self.work['support_source_triangles']+=1
            raw=np.asarray(raw)
            if not len(raw) or np.any(raw<=0):raise ArithmeticError('invalid positive polygon quadrature')
            self.work['support_polygon_source_nodes']+=len(raw)
            return np.array(points),raw/raw.sum()
        a,b=state.edges[0][i:i+2];c,d=state.edges[1][j:j+2]
        x,y=np.meshgrid(a+(b-a)*q,c+(d-c)*q,indexing='ij')
        return np.column_stack((x.ravel(),y.ravel())),(qw[:,None]*qw).ravel()

    def source_nodes(self,state):
        nodes=[];weights=[];q=self.q;qw=self.qw
        for i in (0,1):
            for j in np.flatnonzero(state.axes[i]):
                a,b=state.edges[i][j:j+2];r=a+(b-a)*q
                h=np.zeros((len(q),2));h[:,i]=r
                nodes.append(h);weights.append(self.source_reweight('axis'+str(i),h,state.axes[i,j]*qw))
        if state.ray_factor:
            a,b=state.ray_edges[:-1],state.ray_edges[1:]
            r=a[:,None]+(b-a)[:,None]*q
            p=PHI(r)*qw;p/=p.sum(axis=1)[:,None]
            p*=(state.ray_factor*(ndtr(b)-ndtr(a)))[:,None]
            nodes.append(r.ravel()[:,None]*self.wi);weights.append(p.ravel())
        for i,j in zip(*np.nonzero(state.area)):
            points,raw=self.area_rule(state,i,j)
            nodes.append(points);weights.append(self.source_reweight('area',points,state.area[i,j]*raw))
        if not nodes:return np.empty((0,2)),np.empty(0)
        H=np.concatenate(nodes);P=np.concatenate(weights)
        if state.support_kind in ('upper','lower'):
            normal=self.normal if state.support_kind=='upper' else -self.normal
            deficit=float(np.max(np.maximum(-(H@normal),0)))
            self.work['max_source_support_deficit']=max(self.work['max_source_support_deficit'],deficit)
            if deficit>1e-12*max(1.,float(state.box.max())):raise ArithmeticError('source outside cone')
        assert np.all(P>=0)
        assert abs(P.sum()-(sum(state.masses().values())-state.origin))<1e-12
        self.work['constructed_source_nodes']+=len(P)
        self.work['zero_weight_source_nodes']+=int(np.sum(P==0))
        return H[P>0],P[P>0]
