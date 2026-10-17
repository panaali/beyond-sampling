"""Deterministic, positive-integrand independent checks of spherical moments."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import json
from pathlib import Path
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
import mpmath as mp
from moments import polygon_moments, tail_mass, unit


def simplex_moments(vertices, degree, order):
    """Positive integrals over a flat triangle, a separate numerical route."""
    rays = np.asarray(vertices).T
    nodes, weights = leggauss(order)
    nodes, weights = (nodes + 1)/2, weights/2
    s, t = np.meshgrid(nodes, nodes, indexing="ij")
    y = np.array([s, (1-s)*t, (1-s)*(1-t)])
    points = np.einsum("ab,bij->aij", rays, y)
    radius = np.sqrt(np.sum(points**2, axis=0))
    w = weights[:, None]*weights[None, :]*(1-s)
    jac = abs(np.linalg.det(rays))/radius**3
    return np.array([np.sum(w*jac*(points[0]/radius)**p)
                     for p in range(degree+1)])


def mp_moments(vertices, degree):
    """High-precision version of recurrence, not an independent algorithm."""
    mp.mp.dps = 100
    def dot(a, b):
        return sum(x*y for x,y in zip(a,b))
    def cross(a, b):
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    def normalize(a):
        length = mp.sqrt(dot(a,a))
        return [x/length for x in a]
    vs = [normalize([mp.mpf(float(x)) for x in v]) for v in vertices]
    center = [sum(v[i] for v in vs) for i in range(3)]
    area = mp.mpf(0)
    a = vs[0]
    for b,c in zip(vs[1:-1],vs[2:]):
        area += 2*mp.atan2(abs(dot(a,cross(b,c))), 1+dot(a,b)+dot(b,c)+dot(c,a))
    boundary = [mp.mpf(0) for _ in range(degree)]
    for u,v in zip(vs,vs[1:]+vs[:1]):
        normal = normalize(cross(u,v))
        outward0 = -normal[0]*mp.sign(dot(normal,center))
        theta = mp.atan2(mp.sqrt(dot(cross(u,v),cross(u,v))),dot(u,v))
        tangent = cross(normal,u)
        aa,bb = u[0],tangent[0]
        c = (aa-mp.j*bb)/2
        for p in range(degree):
            integral = mp.mpc(0)
            for m in range(p+1):
                frequency=2*m-p
                part = theta if not frequency else mp.expm1(mp.j*frequency*theta)/(mp.j*frequency)
                integral += mp.binomial(p,m)*c**m*mp.conj(c)**(p-m)*part
            boundary[p] += outward0*mp.re(integral)
    result=[area]
    for p in range(1,degree+1):
        previous=(p-1)*result[p-2] if p>1 else 0
        result.append((previous-boundary[p-1])/(p+1))
    return np.array([float(x) for x in result])


def main():
    start=time.process_time()
    cases=[]
    for sy in (-1,1):
        for sz in (-1,1):
            cases.append((f"octant_{sy}_{sz}",np.diag([1.,sy,sz])))
    cases.append(("oblique",np.array([unit(v) for v in ((1,.2,.1),(1,-.3,.8),(.1,.8,.6))])))
    for eps in (1e-3,1e-7):
        cases.append((f"equatorial_{eps}",np.array([unit(v) for v in (
            (eps,1,0),(eps,np.cos(.4),np.sin(.4)),(2*eps,np.cos(.2),np.sin(.2)))])))
    for eps in (1e-3,1e-8):
        cases.append((f"thin_polar_{eps}",np.array([unit(v) for v in ((1,0,0),(1,eps,0),(1,0,eps))])))
    rows=[]
    for name,vertices in cases:
        analytic=polygon_moments(vertices,9)
        reverse=polygon_moments(vertices[::-1],9)
        positive64=simplex_moments(vertices,9,64)
        positive128=simplex_moments(vertices,9,128)
        high=mp_moments(vertices,9)
        rejected=[]
        for k in range(1,11):
            try:
                tail_mass(vertices,k)
            except ArithmeticError:
                rejected.append(k)
        row=dict(case=name,analytic=analytic.tolist(),positive128=positive128.tolist(),
                 high_precision=high.tolist(),
                 max_abs_error_positive=float(np.max(abs(analytic-positive128))),
                 positive_refinement=float(np.max(abs(positive64-positive128))),
                 max_abs_error_high_precision=float(np.max(abs(analytic-high))),
                 orientation_difference=float(np.max(abs(analytic-reverse))),
                 guarded_tail_rejected_k=rejected,
                 negative_degrees=np.flatnonzero(analytic<0).tolist())
        rows.append(row)
    hemisphere=[]
    for k in range(1,11):
        value=sum(tail_mass(v,k) for _,v in cases[:4])
        hemisphere.append(dict(k=k,mass=value,error=abs(value-1)))
    output=dict(cases=rows,hemisphere=hemisphere,cpu_seconds=time.process_time()-start,
                model_forwards=0,random_inputs=0,
                simplex_nodes=len(cases)*(64**2+128**2),
                high_precision_independent=False,
                classification="analytic primitive; positive quadrature used only for validation")
    target=Path(__file__).with_name("moment_results.json")
    target.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({"max_abs_error_positive":max(r["max_abs_error_positive"] for r in rows),
                      "max_abs_error_high_precision":max(r["max_abs_error_high_precision"] for r in rows),
                      "negative_cases":[r["case"] for r in rows if r["negative_degrees"]],
                      "cpu_seconds":output["cpu_seconds"]}))


if __name__=="__main__":
    main()
