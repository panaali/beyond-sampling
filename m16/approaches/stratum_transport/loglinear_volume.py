"""Positive local log-linear reconstruction; no oracle or history lifting."""
import numpy as np
from affine_volume import AffineOperator


class LoglinearOperator(AffineOperator):
    loglinear=True

    def reconstruct_gradient(self,density,displacements,neighbors,vertices,dimension):
        rows=[];differences=[]
        for offset,value in zip(displacements,neighbors):
            if value>0:
                rows.append(offset);differences.append(np.log(value)-np.log(density))
                self.work['reconstruction_logarithms']+=2
            else:self.work['zero_density_neighbors_omitted']+=1
        return self.fit(rows,differences,dimension)

    def density_factors(self,density,offsets,gradient):
        exponent=offsets@gradient;exponent-=exponent.max()
        self.work['reconstruction_exponentials']+=len(exponent)
        return np.exp(exponent)
