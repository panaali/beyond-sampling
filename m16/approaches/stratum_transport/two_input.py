"""Exact sector representation of a two-input H2 Gaussian pushforward.

Sector boundaries and ranks use exact dyadic rational arithmetic. Evaluated
angles/densities use floating point and are not certified interval arithmetic.
"""
from collections import defaultdict
from fractions import Fraction as F
from functools import cmp_to_key
import math
import numpy as np
from scipy.special import ndtr


class Law:
    def __init__(self,model):
        self.wi,self.wh,self.wo=(np.asarray(x,dtype=float) for x in model)
        assert self.wi.shape==(2,) and self.wh.shape==(2,2) and self.wo.shape[1]==2
        self.work=defaultdict(int);self.cells=[]
        wi=[F(float(v)) for v in self.wi];wh=[[F(float(v)) for v in row] for row in self.wh]
        self.woq=[[F(float(v)) for v in row] for row in self.wo]
        for sign in (-1,1):
            v=[max(F(0),self.mul(F(sign),w)) for w in wi];self.work['rational_comparisons']+=2
            a=[self.add(self.mul(row[0],v[0]),self.mul(row[1],v[1])) for row in wh]
            cuts=self.sorted_unique([self.div(-a[i],wi[i]) for i in (0,1) if wi[i]])
            bounds=[None,*cuts,None]
            for lo,hi in zip(bounds,bounds[1:]):
                q=self.midpoint(lo,hi)
                gate=[self.add(a[i],self.mul(wi[i],q))>0 for i in (0,1)];self.work['rational_comparisons']+=2
                B=[[a[i] if gate[i] else F(0),wi[i] if gate[i] else F(0)] for i in (0,1)]
                det=self.add(self.mul(B[0][0],B[1][1]),-self.mul(B[0][1],B[1][0]))
                nonzero=[any(row) for row in B]
                rank=2 if det else int(any(nonzero))
                kind='area' if rank==2 else ('origin' if not rank else ('axis0' if not nonzero[1] else ('axis1' if not nonzero[0] else 'input_ray')))
                self.cells.append({'sign':sign,'lo':lo,'hi':hi,'B':B,'rank':rank,'kind':kind,'mass':self.angle_mass(lo,hi)})

    def add(self,a,b):self.work['rational_arithmetic']+=1;return a+b
    def mul(self,a,b):self.work['rational_arithmetic']+=1;return a*b
    def div(self,a,b):self.work['rational_arithmetic']+=1;return a/b
    def compare(self,a,b):self.work['rational_comparisons']+=1;return (a>b)-(a<b)
    def sorted_unique(self,values):
        return sorted(set(values),key=cmp_to_key(self.compare))
    def midpoint(self,lo,hi):
        if lo is None:return F(0) if hi is None else self.add(hi,F(-1))
        if hi is None:return self.add(lo,F(1))
        return self.div(self.add(lo,hi),F(2))
    def angle_mass(self,lo,hi):
        self.work['angle_evaluations']+=int(lo is not None)+int(hi is not None)
        a=-math.pi/2 if lo is None else math.atan(float(lo))
        b=math.pi/2 if hi is None else math.atan(float(hi))
        return (b-a)/(2*math.pi)

    def masses(self):
        out={k:0. for k in ('origin','axis0','axis1','input_ray','area')}
        for c in self.cells:out[c['kind']]+=c['mass']
        return out

    def line_density(self,cell,z):
        """Density in a nonzero output coordinate, not arclength."""
        self.work['line_density_calls']+=1
        if z<=0:return 0.
        assert cell['rank']==1
        row=next(row for row in cell['B'] if any(row));a=np.array(row,dtype=float)
        norm=np.linalg.norm(a);mean=a*z/(norm*norm)
        constraints=[(F(1),F(0))]
        if cell['lo'] is not None:constraints.append((-cell['lo'],F(1)))
        if cell['hi'] is not None:constraints.append((cell['hi'],F(-1)))
        lower,upper=-math.inf,math.inf
        for c in constraints:
            dot=self.add(self.mul(c[0],-row[1]),self.mul(c[1],row[0]))
            shift=float(c[0])*mean[0]+float(c[1])*mean[1]
            if not dot:
                if shift<0:return 0.
            else:
                root=-shift*norm/float(dot)
                if dot>0:lower=max(lower,root)
                else:upper=min(upper,root)
        self.work['normal_cdf_calls']+=2;self.work['normal_density_calls']+=1
        return math.exp(-.5*(z/norm)**2)/(math.sqrt(2*math.pi)*norm)*max(0.,ndtr(upper)-ndtr(lower))

    def area_density(self,cell,h):
        self.work['area_density_calls']+=1
        assert cell['rank']==2
        B=np.array(cell['B'],dtype=float);x=np.linalg.solve(B,h)
        if x[0]<=0:return 0.
        ratio=x[1]/x[0]
        if cell['lo'] is not None and ratio<float(cell['lo']):return 0.
        if cell['hi'] is not None and ratio>float(cell['hi']):return 0.
        self.work['bivariate_density_calls']+=1
        return math.exp(-.5*float(x@x))/(2*math.pi*abs(np.linalg.det(B)))

    def output_masses(self):
        out=np.zeros(len(self.wo))
        for cell in self.cells:
            B=cell['B'];lines=[[self.add(self.mul(row[0],B[0][k]),self.mul(row[1],B[1][k])) for k in (0,1)] for row in self.woq]
            cuts=[]
            for i in range(len(lines)):
                for j in range(i):
                    slope=self.add(lines[i][1],-lines[j][1]);intercept=self.add(lines[i][0],-lines[j][0])
                    if slope:
                        q=self.div(-intercept,slope)
                        self.work['rational_comparisons']+=2
                        if (cell['lo'] is None or q>cell['lo']) and (cell['hi'] is None or q<cell['hi']):cuts.append(q)
            bounds=[cell['lo'],*self.sorted_unique(cuts),cell['hi']]
            for lo,hi in zip(bounds,bounds[1:]):
                q=self.midpoint(lo,hi);values=[self.add(line[0],self.mul(line[1],q)) for line in lines]
                best=0
                for i in range(1,len(values)):
                    self.work['rational_comparisons']+=1
                    if values[i]>values[best]:best=i
                out[best]+=self.angle_mass(lo,hi)
        return out

    def summary(self):
        return {'stratum_masses':self.masses(),'output_masses':self.output_masses().tolist(),
            'cells':[{'kind':c['kind'],'rank':c['rank'],'mass':c['mass'],
                'sign':c['sign'],'lo':str(c['lo']),'hi':str(c['hi']),
                'B':[[str(x) for x in row] for row in c['B']]} for c in self.cells],
            'work':dict(self.work),'rational_bit_and_hash_cost_excluded':True}
