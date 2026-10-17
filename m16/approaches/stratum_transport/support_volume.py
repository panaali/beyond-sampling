"""T77: exact directional-cone recursion and positive clipped-cell projection."""
from dataclasses import replace
from fractions import Fraction as F
import math
import numpy as np
from scipy.special import ndtr
from mixed_volume import Operator


class SupportOperator(Operator):
    def __init__(self,model,n,L=6.):
        super().__init__(model,n,L)
        self.wq=tuple(F(float(v)) for v in self.wi)
        W=[[F(float(v)) for v in row] for row in self.wh]
        normal=(-self.wq[1],self.wq[0])
        self.dq=tuple(sum(normal[i]*W[i][j] for i in (0,1)) for j in (0,1))
        self.work['support_rational_products']+=4;self.work['support_rational_sums']+=2
        self.normal=np.array([-self.wi[1],self.wi[0]])/max(self.wi)
        self.pending_kind='origin';self.support_history=[]

    def advance(self,kind):
        if kind=='origin':return 'ray'
        extremes={'ray':[self.wq],'upper':[(F(0),F(1)),self.wq],
            'lower':[(F(1),F(0)),self.wq],'full':[(F(1),F(0)),(F(0),F(1))]}[kind]
        values=[sum(self.dq[i]*r[i] for i in (0,1)) for r in extremes]
        self.work['support_rational_products']+=2*len(values);self.work['support_rational_sums']+=len(values)
        self.work['support_sign_checks']+=len(values)
        positive=any(v>0 for v in values);negative=any(v<0 for v in values)
        return 'full' if positive and negative else ('upper' if positive else ('lower' if negative else 'ray'))

    @staticmethod
    def polygon_area(p):
        if len(p)<3:return 0.
        return abs(float(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))))/2

    def clip_support(self,poly,normal):
        values=poly@normal;answer=[]
        for a,b,va,vb in zip(poly,np.roll(poly,-1,axis=0),values,np.roll(values,-1)):
            if va>=0:answer.append(a)
            if (va>0 and vb<0) or (va<0 and vb>0):answer.append(a+(b-a)*va/(va-vb))
        clean=[]
        for point in answer:
            if not clean or not np.array_equal(point,clean[-1]):clean.append(point)
        if len(clean)>1 and np.array_equal(clean[0],clean[-1]):clean.pop()
        self.work['support_cell_clips']+=1
        return np.array(clean).reshape(-1,2)

    def empty(self,box):
        state=super().empty(box);state.support_kind=self.pending_kind
        state.polygons={};state.cell_volumes=np.zeros((self.n,self.n));state.full_inside=np.ones((self.n,self.n),dtype=bool)
        if state.support_kind not in ('upper','lower'):return state
        normal=self.normal if state.support_kind=='upper' else -self.normal
        for i in range(self.n):
            for j in range(self.n):
                poly=super().area_polygon(state,i,j);values=poly@normal
                if np.all(values>=0):
                    state.cell_volumes[i,j]=super().area_cell_volume(state,i,j);continue
                state.full_inside[i,j]=False
                clipped=self.clip_support(poly,normal)
                state.polygons[i,j]=clipped;state.cell_volumes[i,j]=self.polygon_area(clipped)
        return state

    def initial(self):
        self.pending_kind='origin';return super().initial()

    def area_polygon(self,state,i,j):
        if (i,j) in state.polygons:return state.polygons[i,j]
        return super().area_polygon(state,i,j)

    def area_cell_volume(self,state,i,j):
        if state.support_kind in ('upper','lower'):return state.cell_volumes[i,j]
        return super().area_cell_volume(state,i,j)

    def source_nodes(self,state):
        if state.support_kind not in ('upper','lower'):return super().source_nodes(state)
        # Existing mass is assigned to the admissible polygon at destination.
        # Positive polygon quadrature never samples a forbidden piece of a cell.
        base=replace(state,area=np.where(state.full_inside,state.area,0.))
        h,p=super().source_nodes(base);nodes=[h];weights=[p]
        q=(1+np.array([-1.,1.])/math.sqrt(3))/2
        for i,j in zip(*np.nonzero(state.area*(~state.full_inside))):
            poly=state.polygons[i,j];cell_mass=state.area[i,j]
            if state.cell_volumes[i,j]<=0:raise ArithmeticError('positive mass in empty support cell')
            points=[];raw=[]
            for k in range(1,len(poly)-1):
                a,b,c=poly[0],poly[k],poly[k+1]
                determinant=abs(np.linalg.det(np.column_stack((b-a,c-a))))
                if not determinant:continue
                for s in q:
                    for t in q:
                        points.append((1-s)*a+s*((1-t)*b+t*c));raw.append(determinant*s/4)
                self.work['support_source_triangles']+=1
            raw=np.array(raw);points=np.array(points)
            defect=abs(raw.sum()-state.cell_volumes[i,j])
            self.work['max_polygon_quadrature_volume_defect']=max(self.work['max_polygon_quadrature_volume_defect'],defect)
            if not len(raw) or np.any(raw<=0):raise ArithmeticError('invalid positive polygon quadrature')
            nodes.append(points);weights.append(cell_mass*raw/raw.sum())
            self.work['support_polygon_source_nodes']+=len(raw)
        H=np.concatenate(nodes);P=np.concatenate(weights)
        normal=self.normal if state.support_kind=='upper' else -self.normal
        deficit=float(np.max(np.maximum(-(H@normal),0))) if len(H) else 0.
        self.work['max_source_support_deficit']=max(self.work['max_source_support_deficit'],deficit)
        if deficit>1e-12*max(1.,float(state.box.max())):raise ArithmeticError('source node outside proven support')
        return H,P

    def step(self,state):
        self.pending_kind=self.advance(state.support_kind)
        new,output=super().step(state);kind=new.support_kind
        if kind in ('upper','lower'):
            forbidden_axis=0 if kind=='upper' else 1
            leakage=float(new.axes[forbidden_axis].sum())
            self.work['max_forbidden_axis_mass']=max(self.work['max_forbidden_axis_mass'],leakage)
            if leakage!=0:raise ArithmeticError(f'forbidden axis mass {leakage}')
            if np.any((new.area>0)&(new.cell_volumes<=0)):raise ArithmeticError('positive area transfer into empty support cell')
        elif kind in ('ray','origin') and (np.any(new.area) or np.any(new.axes)):
            raise ArithmeticError('unexpected transverse mass on ray-only support')
        self.support_history.append(kind)
        return new,output

    def area_density(self,state,y):
        if state.support_kind not in ('upper','lower'):return super().area_density(state,y)
        y=np.asarray(y)
        if np.any(y<=0):return 0.
        # Axis/ray contribution remains unchanged. The area contribution is an
        # exact scalar Gaussian line integral of the polygonal cell density.
        value=super().area_density(replace(state,area=np.zeros_like(state.area)),y)
        a=self.inverse@y;b=self.inverse@self.wi
        for i,j in zip(*np.nonzero(state.area)):
            poly=self.area_polygon(state,i,j);lo,hi=-self.L,self.L
            for v,w in zip(poly,np.roll(poly,-1,axis=0)):
                edge=w-v;normal=np.array([-edge[1],edge[0]])
                offset=float(normal@v);coefficient=-float(normal@b);rhs=offset-float(normal@a)
                if coefficient>0:lo=max(lo,rhs/coefficient)
                elif coefficient<0:hi=min(hi,rhs/coefficient)
                elif rhs>0:hi=lo;break
            if hi>lo:value+=state.area[i,j]/self.area_cell_volume(state,i,j)*(ndtr(hi)-ndtr(lo))/self.det
            self.work['support_line_convolution_cells']+=1
        return value
