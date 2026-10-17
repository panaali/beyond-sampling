"""Positive local affine density reconstruction using only cell masses."""
import numpy as np
from quadrature_volume import QuadratureOperator


class AffineOperator(QuadratureOperator):
    def __init__(self,model,n,order=4,L=6.):
        super().__init__(model,n,order,L)

    def centroid(self,poly):
        origin=poly[0];total=0.;moment=np.zeros(2)
        for i in range(1,len(poly)-1):
            a,b=poly[i]-origin,poly[i+1]-origin
            area=abs(a[0]*b[1]-a[1]*b[0])/2
            total+=area;moment+=area*(origin+poly[i]+poly[i+1])/3
            self.work['centroid_triangles']+=1
        if total<=0:raise ArithmeticError('centroid of empty polygon')
        return moment/total

    def fit(self,displacements,differences,dimension):
        self.work['gradient_fits']+=1
        if not len(displacements):return np.zeros(dimension)
        X=np.asarray(displacements).reshape(-1,dimension);y=np.array(differences)
        scale=np.linalg.norm(X,axis=1);keep=scale>0;X=X[keep];y=y[keep];scale=scale[keep]
        self.work['gradient_neighbor_rows']+=len(y)
        if len(y)<dimension:return np.zeros(dimension)
        gradient,_,rank,_=np.linalg.lstsq(X/scale[:,None],y/scale,rcond=None)
        if rank<dimension:
            self.work['rank_deficient_fits']+=1;return np.zeros(dimension)
        return gradient

    def limit(self,density,gradient,vertices):
        variations=vertices@gradient;low=float(np.min(variations));factor=1.
        self.work['limiter_vertex_checks']+=len(vertices)
        if low< -density:
            factor=np.nextafter(density/(-low),0.)
            self.work['limited_cells']+=1
        return gradient*factor

    def reconstruct_gradient(self,density,displacements,neighbors,vertices,dimension):
        gradient=self.fit(displacements,[value-density for value in neighbors],dimension)
        return self.limit(density,gradient,vertices)

    def density_factors(self,density,offsets,gradient):
        return 1+offsets@gradient/density

    def prepare(self,state):
        self.current_state=state
        self.area_centers=np.zeros((self.n,self.n,2));self.area_density_grid=np.zeros((self.n,self.n))
        self.area_gradients=np.zeros((self.n,self.n,2));self.volumes=np.zeros((self.n,self.n))
        self.widths=np.array([np.diff(e) for e in state.edges])
        self.axis_centers=np.array([(e[:-1]+e[1:])/2 for e in state.edges])
        self.axis_density_grid=np.divide(state.axes,self.widths,out=np.zeros_like(state.axes),where=self.widths>0)
        self.axis_gradients=np.zeros_like(state.axes)
        for axis in (0,1):
            for i in np.flatnonzero(state.axes[axis]):
                density=self.axis_density_grid[axis,i];x=self.axis_centers[axis,i];width=self.widths[axis,i]
                neighbors=[j for j in (i-1,i+1) if 0<=j<self.n]
                displacement=[[(self.axis_centers[axis,j]-x)/width] for j in neighbors]
                values=[self.axis_density_grid[axis,j] for j in neighbors]
                self.axis_gradients[axis,i]=self.reconstruct_gradient(density,displacement,values,np.array([[-.5],[.5]]),1)[0]
        if not np.any(state.area):return
        for i in range(self.n):
            for j in range(self.n):
                volume=self.area_cell_volume(state,i,j);self.volumes[i,j]=volume
                if volume>0:
                    self.area_centers[i,j]=self.centroid(self.area_polygon(state,i,j))
                    self.area_density_grid[i,j]=state.area[i,j]/volume
        for i,j in zip(*np.nonzero(state.area)):
            center=self.area_centers[i,j];density=self.area_density_grid[i,j]
            width=np.array([self.widths[0,i],self.widths[1,j]])
            neighbors=[(a,b) for a in range(max(0,i-1),min(self.n,i+2))
                for b in range(max(0,j-1),min(self.n,j+2)) if (a,b)!=(i,j) and self.volumes[a,b]>0]
            displacement=[(self.area_centers[a,b]-center)/width for a,b in neighbors]
            values=[self.area_density_grid[a,b] for a,b in neighbors]
            vertices=(self.area_polygon(state,i,j)-center)/width
            self.area_gradients[i,j]=self.reconstruct_gradient(density,displacement,values,vertices,2)

    def source_nodes(self,state):
        self.prepare(state)
        return super().source_nodes(state)

    def source_reweight(self,kind,points,weights):
        state=self.current_state;mean=points.mean(axis=0)
        if kind=='area':
            i,j=[np.searchsorted(state.edges[a],mean[a],side='right')-1 for a in (0,1)]
            width=np.array([self.widths[0,i],self.widths[1,j]])
            density=self.area_density_grid[i,j]
            factors=self.density_factors(density,(points-self.area_centers[i,j])/width,self.area_gradients[i,j])
        else:
            axis=int(kind[-1]);i=np.searchsorted(state.edges[axis],mean[axis],side='right')-1
            density=self.axis_density_grid[axis,i]
            factors=self.density_factors(density,((points[:,axis]-self.axis_centers[axis,i])/self.widths[axis,i])[:,None],np.array([self.axis_gradients[axis,i]]))
        if np.any(factors<0):raise ArithmeticError('negative limited affine density')
        raw=weights*factors;mass=float(weights.sum());total=float(raw.sum())
        self.work['affine_source_density_values']+=len(weights)
        key='max_loglinear_normalization_change' if getattr(self,'loglinear',False) else 'max_affine_normalization_defect'
        self.work[key]=max(self.work[key],abs(total-mass))
        if total<=0:raise ArithmeticError('empty reconstructed cell')
        return mass*raw/total
