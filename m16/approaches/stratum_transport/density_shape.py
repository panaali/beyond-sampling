"""Two-input density oracle for one-stage projection diagnosis, not closure."""
import math
import numpy as np
from scipy.special import ndtr
from mixed_volume import PHI
from two_input import Law
from quadrature_volume import QuadratureOperator


class ShapeOperator(QuadratureOperator):
    def __init__(self,model,n,shape='both',order=4,L=6.):
        super().__init__(model,n,order,L);self.shape=shape;self.law=Law(model)
        self.density_cells=[]
        for cell in self.law.cells:
            if cell['kind'] not in ('area','axis0','axis1'):continue
            B=np.array(cell['B'],float)
            entry={'kind':cell['kind'],'lo':cell['lo'],'hi':cell['hi']}
            if cell['kind']=='area':entry.update(inverse=np.linalg.inv(B),det=abs(np.linalg.det(B)))
            else:
                coordinate=int(cell['kind'][-1]);a=B[coordinate];norm=np.linalg.norm(a)
                entry.update(coordinate=coordinate,a=a,norm=norm,perp=np.array([-a[1],a[0]])/norm)
            self.density_cells.append(entry)
        self.work['two_input_density_cells_constructed']+=len(self.density_cells)

    def density(self,kind,points):
        values=np.zeros(len(points));self.work['shape_density_queries']+=len(points)
        for cell in self.density_cells:
            if cell['kind']!=kind:continue
            if kind=='area':
                x=points@cell['inverse'].T
                mask=(x[:,0]>0)&(x[:,0]<=self.L)&(np.abs(x[:,1])<=self.L)
                if cell['lo'] is not None:mask&=x[:,1]>=float(cell['lo'])*x[:,0]
                if cell['hi'] is not None:mask&=x[:,1]<=float(cell['hi'])*x[:,0]
                values[mask]+=np.exp(-.5*np.sum(x[mask]**2,axis=1))/(2*math.pi*cell['det'])
                self.work['shape_bivariate_density_values']+=int(mask.sum())
            else:
                z=points[:,cell['coordinate']];a=cell['a'];norm=cell['norm'];perp=cell['perp']
                mean=z[:,None]*a/(norm*norm)
                lo=np.full(len(z),-np.inf);hi=np.full(len(z),np.inf)
                constraints=[(np.array([1.,0.]),0.),(np.array([-1.,0.]),-self.L),
                    (np.array([0.,1.]),-self.L),(np.array([0.,-1.]),-self.L)]
                if cell['lo'] is not None:constraints.append((np.array([-float(cell['lo']),1.]),0.))
                if cell['hi'] is not None:constraints.append((np.array([float(cell['hi']),-1.]),0.))
                for normal,bound in constraints:
                    coefficient=float(normal@perp);rhs=bound-mean@normal
                    if coefficient>0:lo=np.maximum(lo,rhs/coefficient)
                    elif coefficient<0:hi=np.minimum(hi,rhs/coefficient)
                    else:hi=np.where(rhs>0,lo,hi)
                mask=(z>0)&(hi>lo)
                values[mask]+=PHI(z[mask]/norm)/norm*(ndtr(hi[mask])-ndtr(lo[mask]))
                self.work['shape_univariate_density_values']+=int(mask.sum())
                self.work['shape_normal_cdf_values']+=2*int(mask.sum())
        return values

    def source_reweight(self,kind,points,weights):
        if len(self.support_history)!=2 or self.shape=='uniform':return weights
        if self.shape=='area' and kind!='area':return weights
        if self.shape=='axis' and kind=='area':return weights
        density=self.density(kind,points);raw=weights*density;mass=float(weights.sum())
        self.work['shape_reweighted_cells']+=1
        if raw.sum()==0:
            self.work['shape_unresolved_cell_mass']+=mass;self.work['shape_unresolved_cells']+=1
            return weights
        answer=mass*raw/raw.sum()
        self.work['max_shape_cell_mass_defect']=max(self.work['max_shape_cell_mass_defect'],abs(answer.sum()-mass))
        return answer
