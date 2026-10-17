"""Positive fixed-dimensional H2 mixed-measure finite-volume prototype.

Truncated inputs are not renormalized. Projection/source quadrature is
approximate; singular supports and per-row retained probability are explicit.
"""
from collections import defaultdict
from fractions import Fraction
from dataclasses import dataclass
import math
import numpy as np
from scipy.special import ndtr

PHI=lambda x:np.exp(-.5*np.asarray(x)**2)/math.sqrt(2*math.pi)


@dataclass
class State:
    origin:float
    axes:np.ndarray
    area:np.ndarray
    ray_factor:float
    box:np.ndarray
    edges:tuple
    ray_edges:np.ndarray
    def masses(self):
        return {'origin':self.origin,'axis0':float(self.axes[0].sum()),'axis1':float(self.axes[1].sum()),
            'input_ray':float(self.ray_factor*(ndtr(self.ray_edges[-1])-.5)),'area':float(self.area.sum())}


class Operator:
    def __init__(self,model,n,L=6.):
        self.wi,self.wh,self.wo=tuple(np.asarray(x,dtype=float) for x in model)
        self.n=n;self.L=L;self.retained=float(ndtr(L)-ndtr(-L));self.work=defaultdict(float)
        if np.any(self.wi<=0):raise ValueError('initial closure prototype requires both Wi positive')
        self.inverse=np.linalg.inv(self.wh);self.det=abs(np.linalg.det(self.wh))
        self.directions=(np.array([1.,0.]),np.array([0.,1.]),self.wi)
        self.ray_matrices=[np.column_stack((self.wh@d,self.wi)) for d in self.directions]
        if not self.det or any(np.linalg.det(M)==0 for M in self.ray_matrices):raise ValueError('degenerate transfer outside prototype scope')
        self.line_inverse=[np.linalg.inv(M) for M in self.ray_matrices]
        self.line_det=[abs(np.linalg.det(M)) for M in self.ray_matrices]
        scores=[sum(Fraction(float(a))*Fraction(float(b)) for a,b in zip(row,self.wi)) for row in self.wo]
        self.ray_winner=max(range(len(scores)),key=lambda j:scores[j])
        self.axis_winner=np.argmax(self.wo,axis=0)
        differences=np.array([self.wo[i]-self.wo[j] for i in range(len(self.wo)) for j in range(i)])
        slopes=differences@self.wi;keep=slopes!=0
        self.cross_d=differences[keep];self.cross_s=slopes[keep]

    def empty(self,box):
        edges=tuple(float(b)*np.linspace(0,1,self.n+1)**2 for b in box)
        return State(0.,np.zeros((2,self.n)),np.zeros((self.n,self.n)),0.,np.array(box),edges,self.L*np.linspace(0,1,self.n+1)**2)
    def initial(self):
        state=self.empty(np.zeros(2));state.origin=1.;return state
    @staticmethod
    def nodes(edges):
        mid=(edges[:-1]+edges[1:])/2;half=np.diff(edges)/2
        return mid[:,None]+half[:,None]*np.array([-1.,1.])/math.sqrt(3)

    def source_nodes(self,state):
        nodes=[];weights=[]
        for i in (0,1):
            r=self.nodes(state.edges[i]).ravel();p=np.repeat(state.axes[i]/2,2);keep=p>0
            h=np.zeros((keep.sum(),2));h[:,i]=r[keep];nodes.append(h);weights.append(p[keep])
        if state.ray_factor:
            r=self.nodes(state.ray_edges);p=PHI(r);p/=p.sum(axis=1)[:,None]
            p*= (state.ray_factor*np.diff(ndtr(state.ray_edges)))[:,None]
            nodes.append(r.ravel()[:,None]*self.wi);weights.append(p.ravel())
        if np.any(state.area):
            a,b=self.nodes(state.edges[0]).ravel(),self.nodes(state.edges[1]).ravel()
            x,y=np.meshgrid(a,b,indexing='ij');p=np.repeat(np.repeat(state.area,2,axis=0),2,axis=1).ravel()/4
            keep=p>0;nodes.append(np.column_stack((x.ravel()[keep],y.ravel()[keep])));weights.append(p[keep])
        if not nodes:return np.empty((0,2)),np.empty(0)
        return np.concatenate(nodes),np.concatenate(weights)

    def step(self,state):
        box=np.maximum(self.wh,0)@state.box+self.L*self.wi
        new=self.empty(box);new.origin=state.origin*(.5-ndtr(-self.L));new.ray_factor=state.origin
        output=np.zeros(len(self.wo));output[0]+=new.origin;output[self.ray_winner]+=state.origin*(ndtr(self.L)-.5)
        h,p=self.source_nodes(state);self.work['source_nodes']+=len(p)
        for start in range(0,len(p),1024):
            H=h[start:start+1024];P=p[start:start+1024];c=H@self.wh.T
            boundaries=[np.full((len(H),1),-self.L),np.full((len(H),1),self.L)]
            boundaries.extend((edges[None,:]-c[:,i,None])/self.wi[i] for i,edges in enumerate(new.edges))
            if len(self.cross_s):boundaries.append(-(c@self.cross_d.T)/self.cross_s[None,:])
            cuts=np.sort(np.clip(np.concatenate(boundaries,axis=1),-self.L,self.L),axis=1)
            probability=np.diff(ndtr(cuts),axis=1);mid=(cuts[:,:-1]+cuts[:,1:])/2
            self.work['normal_cdf_values']+=cuts.size;self.work['input_intervals']+=mid.size
            self.work['max_row_mass_defect']=max(self.work['max_row_mass_defect'],float(np.max(abs(probability.sum(axis=1)-self.retained))))
            mass=P[:,None]*probability;dest=np.maximum(c[:,None,:]+mid[:,:,None]*self.wi,0)
            active=dest>0;positive=mass>0
            origin=positive&~active[:,:,0]&~active[:,:,1]
            new.origin+=float(mass[origin].sum());output[0]+=float(mass[origin].sum())
            idx=[np.searchsorted(new.edges[i],dest[:,:,i],side='right')-1 for i in (0,1)]
            for i in (0,1):
                mask=positive&active[:,:,i]&~active[:,:,1-i]
                if np.any(idx[i][mask]>=self.n):raise ArithmeticError('axis transfer outside finite bound')
                np.add.at(new.axes[i],idx[i][mask],mass[mask]);output[self.axis_winner[i]]+=float(mass[mask].sum())
            interior=positive&active[:,:,0]&active[:,:,1]
            if any(np.any(idx[i][interior]>=self.n) for i in (0,1)):raise ArithmeticError('area transfer outside finite bound')
            np.add.at(new.area,(idx[0][interior],idx[1][interior]),mass[interior])
            winners=np.argmax(dest[interior]@self.wo.T,axis=1)
            np.add.at(output,winners,mass[interior]);self.work['interior_readout_evaluations']+=len(winners)
        oldmass=sum(state.masses().values());newmass=sum(new.masses().values())
        self.work['max_total_mass_defect']=max(self.work['max_total_mass_defect'],abs(newmass-oldmass*self.retained))
        assert abs(newmass-oldmass*self.retained)<1e-12
        assert abs(output.sum()-newmass)<1e-12
        return new,output

    def area_density(self,state,y):
        """Exact positive convolution of the represented area/line densities."""
        y=np.asarray(y);self.work['area_density_queries']+=1
        if np.any(y<=0):return 0.
        value=0.;a=self.inverse@y;b=self.inverse@self.wi
        if np.any(state.area):
            lows=np.full(state.area.shape,-self.L);highs=np.full(state.area.shape,self.L)
            for i in (0,1):
                lo=state.edges[i][:-1];hi=state.edges[i][1:]
                shape=(-1,1) if i==0 else (1,-1)
                if b[i]:
                    first=(a[i]-hi)/b[i];second=(a[i]-lo)/b[i]
                    lows=np.maximum(lows,np.minimum(first,second).reshape(shape));highs=np.minimum(highs,np.maximum(first,second).reshape(shape))
                else:
                    # Half-open source cells, with the final endpoint included.
                    valid=(a[i]>=lo)&((a[i]<hi)|((np.arange(self.n)==self.n-1)&(a[i]==hi)))
                    highs=np.where(valid.reshape(shape),highs,lows)
            valid=highs>lows;prob=np.zeros_like(lows);prob[valid]=ndtr(highs[valid])-ndtr(lows[valid])
            density=state.area/(np.diff(state.edges[0])[:,None]*np.diff(state.edges[1])[None,:])
            value=float(np.sum(density*prob))/self.det;self.work['line_convolution_cells']+=state.area.size
        for k in range(3):
            r,x=self.line_inverse[k]@y
            if r<=0 or abs(x)>self.L:continue
            if k==2:g=state.ray_factor*float(PHI(r)) if r<self.L else 0.
            else:
                edge=state.edges[k];i=np.searchsorted(edge,r,side='right')-1
                g=state.axes[k,i]/(edge[i+1]-edge[i]) if 0<=i<self.n else 0.
            value+=g*float(PHI(x))/self.line_det[k];self.work['line_source_density_terms']+=1
        return value

    def histogram_output(self,state):
        """Readout integral of the represented measure; area cells are uniform."""
        out=np.zeros(len(self.wo));out[0]=state.origin
        out[self.ray_winner]+=state.ray_factor*(ndtr(self.L)-.5)
        for i in (0,1):out[self.axis_winner[i]]+=state.axes[i].sum()
        def clip(poly,normal):
            if not len(poly):return poly
            vals=poly@normal;answer=[]
            for a,b,va,vb in zip(poly,np.roll(poly,-1,axis=0),vals,np.roll(vals,-1)):
                if va>=0:answer.append(a)
                if (va>0 and vb<0) or (va<0 and vb>0):answer.append(a+(b-a)*va/(va-vb))
            self.work['readout_polygon_clips']+=1
            return np.array(answer).reshape(-1,2)
        for i,j in zip(*np.nonzero(state.area)):
            x0,x1=state.edges[0][i:i+2];y0,y1=state.edges[1][j:j+2]
            rectangle=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
            for label in range(len(self.wo)):
                poly=rectangle
                for other in range(len(self.wo)):
                    if label==other:continue
                    normal=self.wo[label]-self.wo[other]
                    if not np.any(normal):
                        if other<label:poly=np.empty((0,2));break
                    else:poly=clip(poly,normal)
                    if not len(poly):break
                if len(poly)>2:
                    cross=poly[:,0]*np.roll(poly[:,1],-1)-poly[:,1]*np.roll(poly[:,0],-1)
                    out[label]+=state.area[i,j]*abs(cross.sum())/(2*(x1-x0)*(y1-y0))
        return out
