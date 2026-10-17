"""Vectorized degree1..3 spherical triangle reduction for signed-L1 endpoints."""
import numpy as np
from triangle import _positive_original


def signed_l1_triangles(vertices, exponent=None, absolute_tolerance=1e-11,
                        min_tangent_ratio=1e-5, roundoff_screen=1e-10):
    """Return batch of affine triangle integrals and aggregate diagnostics.

    vertices shape(...,3,d), d>=3, each endpoint has signed L1 norm1 and each
    triangle shares an orthant. exponent4..6; no sampled inputs or model calls.
    roundoff_screen is a heuristic, NOT a certified floating-point bound.
    """
    vv=np.asarray(vertices,dtype=float)
    if vv.ndim<2 or vv.shape[-2]!=3 or vv.shape[-1]<3:
        raise ValueError('need endpoint arrays (...,3,d)')
    shape=vv.shape[:-2]; d=vv.shape[-1]
    vv=vv.reshape(-1,3,d)
    if not np.isfinite(vv).all():
        raise ValueError('finite endpoints required')
    if np.any((vv.min(axis=1)<0)&(vv.max(axis=1)>0)):
        raise ValueError('each triangle must share an orthant')
    residual=float(np.max(np.abs(np.sum(np.abs(vv),axis=-1)-1))) if len(vv) else 0.
    if residual>1e-10:
        raise ValueError('each endpoint must have signed L1 norm1')
    p=(d if exponent is None else exponent)-3
    if p not in (1,2,3) or absolute_tolerance<=0:
        raise ValueError('exponent must be4,5,6 and tolerance positive')
    n=len(vv); A,B,C=vv[:,0],vv[:,1],vv[:,2]
    u=B-A; v=C-A
    lu=np.linalg.norm(u,axis=1)
    e1=u/np.where(lu>0,lu,1)[:,None]
    v1=np.einsum('ij,ij->i',v,e1)
    vt=v-v1[:,None]*e1
    lv=np.linalg.norm(vt,axis=1)
    e2=vt/np.where(lv>0,lv,1)[:,None]
    a1=np.einsum('ij,ij->i',A,e1)
    a2=np.einsum('ij,ij->i',A,e2)
    perpendicular=A-a1[:,None]*e1-a2[:,None]*e2
    delta=np.linalg.norm(perpendicular,axis=1)
    endpoint_norm=np.linalg.norm(vv,axis=2)
    resolution=np.minimum(lu,lv)/np.max(endpoint_norm,axis=1)
    good=(resolution>=min_tangent_ratio)&(delta>0)
    values=np.empty(n)
    ids=np.flatnonzero(good)
    cancellations=np.zeros(len(ids))
    series_bound=np.zeros(len(ids))
    geometric_rejections=0
    if len(ids):
        # Consistent positive orientation: determinant is delta*lu*lv.
        rays=np.stack([np.stack([delta,a1,a2],axis=1),
                       np.stack([delta,a1+lu,a2],axis=1),
                       np.stack([delta,a1+v1,a2+lv],axis=1)],axis=1)[ids]
        lengths=np.linalg.norm(rays,axis=2)
        rays=rays/lengths[:,:,None]
        nxt=np.roll(rays,-1,axis=1)
        cross=np.cross(rays,nxt)
        edge_sine=np.linalg.norm(cross,axis=2)
        edge_cosine=np.einsum('nij,nij->ni',rays,nxt)
        theta=np.arctan2(edge_sine,edge_cosine)
        nu0=-cross[:,:,0]/edge_sine
        determinant=delta[ids]*lu[ids]*lv[ids]
        solid_angle=2*np.arctan2(determinant/np.prod(lengths,axis=1),1+np.sum(edge_cosine,axis=1))
        m1=-np.sum(nu0*theta,axis=1)/2
        sum_abs_m1=np.sum(np.abs(nu0*theta),axis=1)/2
        if p==1:
            moment=m1; abs_terms=sum_abs_m1
        elif p==2:
            integral1=(rays[:,:,0]+nxt[:,:,0])*np.tan(theta/2)
            boundary=np.sum(nu0*integral1,axis=1)
            moment=(solid_angle-boundary)/3
            abs_terms=(solid_angle+np.sum(np.abs(nu0*integral1),axis=1))/3
        else:
            half=theta/2
            aa=(rays[:,:,0]+nxt[:,:,0])/(2*np.cos(half))
            bb=(nxt[:,:,0]-rays[:,:,0])/(2*np.sin(half))
            difference=theta-np.sin(theta)
            small=theta<.1
            tt=theta[small]
            difference[small]=tt**3*(1/6-tt**2/120+tt**4/5040-tt**6/362880+tt**8/39916800)
            integral2=aa**2*(theta+np.sin(theta))/2+bb**2*difference/2
            boundary=np.sum(nu0*integral2,axis=1)
            moment=(2*m1-boundary)/4
            abs_terms=(2*sum_abs_m1+np.sum(np.abs(nu0*integral2),axis=1))/4
            sine_remainder=np.zeros_like(theta)
            sine_remainder[small]=tt**13/6227020800.
            series_bound=np.sum(np.abs(nu0)*bb**2*sine_remainder,axis=1)/(8*determinant*delta[ids]**p)
        cancellations=32*np.finfo(float).eps*abs_terms/np.maximum(np.abs(moment),np.finfo(float).tiny)
        estimate=moment/(determinant*delta[ids]**p)
        qhi=np.max(endpoint_norm[ids]**2,axis=1)
        # Shared signed-L1 plane provides q>=1/d throughout the triangle.
        lower=.5*qhi**(-(p+3)/2)
        upper=np.full(len(ids),.5*d**((p+3)/2))
        acceptable=(np.isfinite(estimate)&(estimate>=lower*(1-1e-10))&
                    (estimate<=upper*(1+1e-10))&(cancellations<=roundoff_screen))
        geometric_rejections=int(np.count_nonzero(~acceptable))
        values[ids[acceptable]]=estimate[acceptable]
        good[ids[~acceptable]]=False
    fallback_stats=[]
    for i in np.flatnonzero(~good):
        values[i],meta=_positive_original(vv[i],p+3,absolute_tolerance,128)
        fallback_stats.append(meta)
    fallback_counters=('quadrature_nodes','quadrature_error_bound','legendre_rule_generations',
                       'legendre_rule_generation_nodes','bound_trials','bound_lgamma_calls',
                       'bound_exp_calls','bound_log_calls')
    stats={key:sum(s.get(key,0) for s in fallback_stats) for key in fallback_counters}
    stats['maximum_quadrature_error_bound']=max((s['quadrature_error_bound'] for s in fallback_stats),default=0.)
    used_series=series_bound[good[ids]]
    stats['analytic_series_truncation_bound_sum']=float(np.sum(used_series))
    stats['maximum_analytic_series_truncation_bound']=float(np.max(used_series)) if len(used_series) else 0.
    stats.update(triangles=n,ambient_dimension=d,exponent=p+3,
                 analytic_triangles=int(np.count_nonzero(good)),
                 positive_quadrature_fallbacks=len(fallback_stats),
                 tangent_basis_constructions=n,qr_factorizations=0,svd_condition_estimates=0,
                 linear_solves=0,model_forward_passes=0,known_plane_residual=residual,
                 boundary_triangles_attempted=len(ids),
                 great_circle_edges=3*len(ids),great_circle_arctan2_calls=3*len(ids),
                 solid_angle_arctan2_calls=len(ids),
                 elementary_tan_calls=3*len(ids) if p==2 else 0,
                 elementary_sin_calls=9*len(ids) if p==3 else 0,
                 elementary_cos_calls=3*len(ids) if p==3 else 0,
                 numerical_screen_rejections=geometric_rejections,
                 maximum_heuristic_relative_roundoff=float(np.max(cancellations)) if len(ids) else 0.,
                 method='analytic_weighted_spherical_triangles_with_guarded_positive_quadrature',
                 roundoff_certified=False)
    return values.reshape(shape),stats
