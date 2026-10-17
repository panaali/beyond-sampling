"""Two area densities separated by the input-ray interface; fixed state size."""
from copy import copy
import numpy as np
from support_volume import SupportOperator
from quadrature_volume import QuadratureOperator
from affine_volume import AffineOperator


class InterfaceOperator(SupportOperator):
    def __init__(self,model,n,reconstruction='uniform',order=4,L=6.):
        super().__init__(model,n,L)
        cls={'uniform':QuadratureOperator,'affine':AffineOperator}[reconstruction]
        self.helpers=[cls(model,n,order,L) for _ in range(3)]
        self.reconstruction=reconstruction
        self.work['interface_helper_operators_constructed']+=3

    def empty(self,box):
        state=super().empty(box);state.side_states=[]
        for helper,kind in zip(self.helpers[:2],('upper','lower')):
            helper.pending_kind=kind;state.side_states.append(helper.empty(box))
        return state

    def source_nodes(self,state):
        singular=copy(state);singular.area=np.zeros_like(state.area)
        H,P=self.helpers[2].source_nodes(singular);nodes=[H];weights=[P]
        for helper,view in zip(self.helpers[:2],state.side_states):
            h,p=helper.source_nodes(view);nodes.append(h);weights.append(p)
        H=np.concatenate(nodes);P=np.concatenate(weights)
        assert abs(P.sum()-(sum(state.masses().values())-state.origin))<1e-12
        return H,P

    def add_area(self,state,indices,mass,interior,preactivation):
        super().add_area(state,indices,mass,interior,preactivation)
        sign=preactivation@self.normal
        plus=np.broadcast_to((sign>=0)[:,None],interior.shape)
        zero=interior&np.broadcast_to((sign==0)[:,None],interior.shape)
        self.work['floating_zero_side_mass']+=float(mass[zero].sum())
        self.work['interface_side_classifications']+=len(sign)
        for side,mask in enumerate((interior&plus,interior&~plus)):
            view=state.side_states[side]
            np.add.at(view.area,(indices[0][mask],indices[1][mask]),mass[mask])

    def step(self,state):
        new,output=super().step(state)
        total=new.side_states[0].area+new.side_states[1].area
        defect=float(np.max(abs(total-new.area)))
        self.work['max_area_side_sum_defect']=max(self.work['max_area_side_sum_defect'],defect)
        assert defect<1e-12
        for view in new.side_states:
            if np.any((view.area>0)&(view.cell_volumes<=0)):
                raise ArithmeticError('positive area in empty side polygon')
        return new,output

    def report(self):
        return {'transport':dict(self.work),'upper_area_helper':dict(self.helpers[0].work),
            'lower_area_helper':dict(self.helpers[1].work),'singular_helper':dict(self.helpers[2].work)}
