"""T106 derivative of T104 6b4f511: explicit degree-three edge identities.

No shared source edits. Positive fallback is injected, allowing the exact same
counted T52 implementation in old/new comparisons. No rounding certificate.
"""
import numpy as np


def prepare(endpoints):
    rays=np.asarray(endpoints,dtype=float)
    if rays.shape!=(3,6) or not np.isfinite(rays).all():
        raise ValueError('finite3x6 base rays required')
    if np.any((rays.min(axis=0)<0)&(rays.max(axis=0)>0)):
        raise ValueError('base rays must share orthant')
    if np.max(np.abs(np.sum(np.abs(rays),axis=1)-1))>1e-10:
        raise ValueError('base rays must have signed L1 norm1')
    u=rays[1]-rays[0]; v=rays[2]-rays[0]
    lu=float(np.linalg.norm(u)); e1=u/(lu if lu>0 else 1)
    v1=float(v@e1); vt=v-v1*e1
    lv=float(np.linalg.norm(vt)); e2=vt/(lv if lv>0 else 1)
    return dict(rays=rays.copy(),lu=lu,lv=lv,v1=v1,e1=e1,e2=e2,
                tangent_basis_constructions=1)


def evaluate(prefix,rest,frame,positive,absolute_tolerance=1e-11,
             min_tangent_ratio=1e-5,roundoff_screen=1e-10,normalize_edges=False):
    prefix=np.asarray(prefix,dtype=float); rest=np.asarray(rest,dtype=float)
    if prefix.ndim!=2 or prefix.shape[1]!=6 or rest.shape!=(len(prefix),):
        raise ValueError('prefix shape(n,6), rest shape(n,) required')
    if not np.isfinite(prefix).all() or not np.isfinite(rest).all() or np.any(rest<0):
        raise ValueError('finite prefix and nonnegative remainder required')
    vv=prefix[:,None,:]+rest[:,None,None]*frame['rays'][None,:,:]
    if np.any((vv.min(axis=1)<0)&(vv.max(axis=1)>0)):
        raise ValueError('each triangle must share an orthant')
    residual=float(np.max(np.abs(np.sum(np.abs(vv),axis=-1)-1))) if len(vv) else 0.
    if residual>1e-10:raise ValueError('endpoints must have signed L1 norm1')
    shape=(len(vv),); d=6
    p=3
    if p not in (1,2,3) or absolute_tolerance<=0:
        raise ValueError('exponent must be4,5,6 and tolerance positive')
    n=len(vv); A,B,C=vv[:,0],vv[:,1],vv[:,2]
    lu=rest*frame['lu']; lv=rest*frame['lv']; v1=rest*frame['v1']
    e1=frame['e1']; e2=frame['e2']
    a1=A@e1; a2=A@e2
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
        nu0=-cross[:,:,0]/np.where(edge_sine>0,edge_sine,1.)
        determinant=delta[ids]*lu[ids]*lv[ids]
        m1=-np.sum(nu0*theta,axis=1)/2
        sum_abs_m1=np.sum(np.abs(nu0*theta),axis=1)/2
        sine=edge_sine; cosine=edge_cosine
        if normalize_edges:
            circle_norm=np.sqrt(sine*sine+cosine*cosine)
            safe_circle=np.where(circle_norm>0,circle_norm,1.)
            sine=sine/safe_circle; cosine=cosine/safe_circle
        valid_edges=np.all(sine>0,axis=1)
        safe_sine=np.where(sine>0,sine,1.)
        one_plus=1+cosine
        aa2=(rays[:,:,0]+nxt[:,:,0])**2/(2*one_plus)
        bb2=(nxt[:,:,0]-rays[:,:,0])**2*one_plus/(2*safe_sine**2)
        difference=theta-sine
        small=theta<.1
        tt=theta[small]
        difference[small]=tt**3*(1/6-tt**2/120+tt**4/5040-tt**6/362880+tt**8/39916800)
        integral2=aa2*(theta+sine)/2+bb2*difference/2
        boundary=np.sum(nu0*integral2,axis=1)
        moment=(2*m1-boundary)/4
        abs_terms=(2*sum_abs_m1+np.sum(np.abs(nu0*integral2),axis=1))/4
        sine_remainder=np.zeros_like(theta)
        sine_remainder[small]=tt**13/6227020800.
        series_bound=np.sum(np.abs(nu0)*bb2*sine_remainder,axis=1)/(8*determinant*delta[ids]**p)
        cancellations=32*np.finfo(float).eps*abs_terms/np.maximum(np.abs(moment),np.finfo(float).tiny)
        estimate=moment/(determinant*delta[ids]**p)
        qhi=np.max(endpoint_norm[ids]**2,axis=1)
        # Shared signed-L1 plane provides q>=1/d throughout the triangle.
        lower=.5*qhi**(-(p+3)/2)
        upper=np.full(len(ids),.5*d**((p+3)/2))
        acceptable=(valid_edges&np.isfinite(estimate)&(estimate>=lower*(1-1e-10))&
                    (estimate<=upper*(1+1e-10))&(cancellations<=roundoff_screen))
        geometric_rejections=int(np.count_nonzero(~acceptable))
        values[ids[acceptable]]=estimate[acceptable]
        good[ids[~acceptable]]=False
    fallback_stats=[]
    for i in np.flatnonzero(~good):
        values[i],meta=positive(vv[i],p+3,absolute_tolerance,128)
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
                 tangent_basis_constructions=0, fixed_frame_reuses=n,qr_factorizations=0,svd_condition_estimates=0,
                 linear_solves=0,model_forward_passes=0,known_plane_residual=residual,
                 boundary_triangles_attempted=len(ids),
                 great_circle_edges=3*len(ids),great_circle_arctan2_calls=3*len(ids),
                 solid_angle_arctan2_calls=0,
                 elementary_tan_calls=3*len(ids) if p==2 else 0,
                 elementary_sin_calls=0,
                 elementary_cos_calls=0, edge_pair_normalizations=3*len(ids) if normalize_edges else 0,
                 numerical_screen_rejections=geometric_rejections,
                 maximum_heuristic_relative_roundoff=float(np.max(cancellations)) if len(ids) else 0.,
                 method='degree_three_half_angle_squares_with_guarded_positive_quadrature',
                 normalized_edge_pairs=bool(normalize_edges),
                 roundoff_certified=False)
    return values.reshape(shape),stats
