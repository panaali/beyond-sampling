"""Frozen mass/complement/refinement tests before any trained model run."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import eval_legendre
from space import Space, unit

start = time.process_time()
rows = []
normals = [np.array(v, dtype=float) for v in ((1,0,0),(0,1,0),(-np.sqrt(3),1,0),
            (.3,.4,.5),(1,1e-3,0),(.01,1,.3))]
for N, M in ((4,0),(4,1),(8,1),(16,2)):
    space = Space(N,M)
    fine = Space(N,M,2*space.order)
    for normal in normals:
        v = unit(normal)
        matrix = space.matrix(v)
        refined = fine.matrix(v)
        expected_mass = (np.pi-np.arccos(v[0]))/(2*np.pi)
        mass_error = space.mass(matrix@space.one())-expected_mass
        complement = np.max(abs(matrix+space.matrix(-v)-space.eye))
        hermitian = np.max(abs(matrix-matrix.conj().T))
        eig = np.linalg.eigvalsh(matrix)
        refinement = np.max(abs(matrix-refined))
        row = dict(N=N,M=M,normal=v.tolist(),mass_error=mass_error,
                   complement_error=float(complement),hermitian_error=float(hermitian),
                   minimum_eigenvalue=float(eig[0]),maximum_eigenvalue=float(eig[-1]),
                   refinement_error=float(refinement))
        rows.append(row)
        assert abs(mass_error)<1e-8, row
        assert complement<1e-12 and hermitian<1e-12, row
        assert eig[0]>-1e-8 and eig[-1]<1+1e-8, row
    # All twelve root half-lunes; every mass includes factor one half.
    root = space.cut(space.one(), [-np.sqrt(3),1,0])
    root_mass_error = 12*space.mass(root)-1
    assert abs(root_mass_error)<1e-10
    rows.append(dict(N=N,M=M,root_partition_error=root_mass_error,
        protected_wall_readout_error=space.mass(space.cut(root,[1,0,0]))-space.mass(root),
        work=dict(space.work)))
# Independent direct-z adaptive integral for a nontrivial radial off-diagonal entry.
sp = Space(4,1)
v = unit([.3,.4,.5])
cut = sp.matrix(v)
n,p = 2,4
boundary = np.hypot(v[1],v[2])
def integrand(z):
    radial = np.sqrt((2*n+1)*(2*p+1))*eval_legendre(n,2*z-1)*eval_legendre(p,2*z-1)
    if z >= boundary:
        arc = 1.
    else:
        arc = np.arccos(-v[0]*z/(boundary*np.sqrt(1-z*z)))/np.pi
    return radial*arc
reference, error = quad(integrand,0,1,points=[boundary],epsabs=1e-12,epsrel=1e-12)
independent = abs(cut[n*3+1,p*3+1]-reference)
assert independent<1e-10
output = dict(rows=rows,independent_direct_z_error=float(independent),
              scipy_error_estimate=error,cpu_seconds=time.process_time()-start)
(Path(__file__).resolve().parent/'checks_results.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(dict(max_refinement=max(r.get('refinement_error',0) for r in rows),
    max_mass_error=max(abs(r.get('mass_error',0)) for r in rows),
    independent_direct_z_error=float(independent),cpu_seconds=output['cpu_seconds'])))
