"""Positive area quadrature aligned to the next input-ray side event."""
import numpy as np
from quadrature_volume import QuadratureOperator
from affine_volume import AffineOperator
from loglinear_volume import LoglinearOperator
from interface_volume import InterfaceOperator


class EventRule:
    def area_rule(self,state,i,j):
        poly=self.area_polygon(state,i,j);normal=self.wh.T@self.normal
        if self.event_cut:
            pieces=[self.clip_support(poly,normal),self.clip_support(poly,-normal)]
            self.work['event_polygon_cuts']+=2
        else:pieces=[poly]
        points=[];weights=[]
        for piece in pieces:
            for k in range(1,len(piece)-1):
                a,b,c=piece[0],piece[k],piece[k+1]
                determinant=abs(np.linalg.det(np.column_stack((b-a,c-a))))
                if not determinant:continue
                for s,ws in zip(self.q,self.qw):
                    for t,wt in zip(self.q,self.qw):
                        points.append((1-s)*a+s*((1-t)*b+t*c));weights.append(determinant*s*ws*wt)
                self.work['event_source_triangles']+=1
        weights=np.array(weights)
        if not len(weights) or np.any(weights<=0):raise ArithmeticError('empty event quadrature')
        self.work['event_area_source_nodes']+=len(weights)
        return np.array(points),weights/weights.sum()


class EventUniform(EventRule,QuadratureOperator):pass
class EventAffine(EventRule,AffineOperator):pass
class EventLoglinear(EventRule,LoglinearOperator):pass


class EventInterfaceOperator(InterfaceOperator):
    def __init__(self,model,n,reconstruction='uniform',event_cut=True,order=4,L=6.):
        super().__init__(model,n,reconstruction,order,L)
        for helper in self.helpers:helper.event_cut=event_cut

    def helper_type(self,reconstruction):
        return {'uniform':EventUniform,'affine':EventAffine,'loglinear':EventLoglinear}[reconstruction]
