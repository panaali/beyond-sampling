"""Exact-formula two-input tail, with explicitly numerical polygon geometry."""
import itertools
import numpy as np
from moments import unit, area, tail_mass_with_fallback

GEOMETRY_TOL = 1e-13


def clip(polygon, normal, stats):
    stats["clips"] += 1
    if len(polygon)<3:
        return np.empty((0,3))
    if not np.any(normal):
        return polygon.copy()
    n=unit(normal)
    values=polygon@n
    if values.min() >= -GEOMETRY_TOL:
        return polygon.copy()
    if values.max() <= GEOMETRY_TOL:
        return np.empty((0,3))
    output=[]
    for u,v,a,b in zip(polygon,np.roll(polygon,-1,axis=0),values,np.roll(values,-1)):
        if a>=-GEOMETRY_TOL:
            output.append(u)
        if (a>GEOMETRY_TOL and b < -GEOMETRY_TOL) or (a < -GEOMETRY_TOL and b>GEOMETRY_TOL):
            output.append(unit(u+(v-u)*a/(a-b)))
    clean=[]
    for v in output:
        if not clean or np.linalg.norm(v-clean[-1])>1e-14:
            clean.append(v)
    if len(clean)>1 and np.linalg.norm(clean[0]-clean[-1])<1e-14:
        clean.pop()
    p=np.array(clean).reshape(-1,3)
    return p if len(p)>=3 and area(p)>0 else np.empty((0,3))


def conditional_probability(model, normalized_hidden, prefix_top, prefix_dimension):
    """Return P(correct | normalized prefix summary), analytically integrating
    its independent chi_k radius and two new Gaussian inputs.

    prefix_top is a list of (normalized_value,index), length min(2,k), highest first.
    Model is (Wi flat, Wh matrix, Wo matrix). All arithmetic is floating point;
    moment guards can fail explicitly. No random/model sample evaluations occur.
    """
    wi,wh,wo=(np.asarray(w,dtype=float) for w in model)
    H=len(wi);T=len(wo);k=prefix_dimension
    if k!=T-2 or k<1 or len(prefix_top)!=min(k,2):
        raise ValueError("need actual prefix dimension T-2 and its largest two records")
    if len(prefix_top)==2 and prefix_top[0][0]<=prefix_top[1][0]:
        raise ValueError("prefix records must be strictly ordered")
    stats=dict(clips=0,affine_products=0,moment_calls=0,gate_cells=0,
               correct_pieces=0,high_precision_fallbacks=0,positive_fallbacks=0,
               positive_quadrature_fallbacks=0,fallback_rule_generations=0,
               fallback_rule_generation_nodes=0,
               fallback_quadrature_nodes=0,fallback_quadrature_error_bound=0.,
               method_class="analytic_regions_with_numerical_quadrature_fallback")
    B=np.zeros((H,3));B[:,0]=normalized_hidden
    regions=[(np.diag([1.,a,b]),B.copy()) for a,b in itertools.product((-1,1),repeat=2)]
    for input_coordinate in (1,2):
        next_regions=[]
        for polygon,B in regions:
            Z=wh@B;Z[:,input_coordinate]+=wi
            stats["affine_products"]+=1
            partial=[(polygon,np.zeros_like(B))]
            for neuron in range(H):
                if not np.any(Z[neuron]):
                    continue
                split=[]
                for p,A in partial:
                    plus=clip(p,Z[neuron],stats)
                    minus=clip(p,-Z[neuron],stats)
                    if len(plus):
                        C=A.copy();C[neuron]=Z[neuron]
                        split.append((plus,C))
                    if len(minus):
                        split.append((minus,A.copy()))
                partial=split
            next_regions.extend(partial)
        regions=next_regions
    stats["gate_cells"]=len(regions)
    rank_values=[(np.array([v,0.,0.]),idx) for v,idx in prefix_top]
    rank_values.extend([(np.array([0.,1.,0.]),k),(np.array([0.,0.,1.]),k+1)])
    probability=0.;partition=0.
    for polygon,B in regions:
        partition+=tail_mass_with_fallback(polygon,k,stats);stats["moment_calls"]+=1
        for rank_position,(value,target_index) in enumerate(rank_values):
            q=polygon
            # Reject positive-volume ties that first-index argmax awards elsewhere.
            for competitor in range(T):
                if competitor==target_index:
                    continue
                margin=(wo[target_index]-wo[competitor])@B
                if not np.any(margin):
                    if competitor<target_index:
                        q=np.empty((0,3));break
                else:
                    q=clip(q,margin,stats)
                if not len(q):
                    break
            if not len(q):
                continue
            for max_position,(max_value,_) in enumerate(rank_values):
                if max_position==rank_position:
                    continue
                p=clip(q,max_value-value,stats)
                for other_position,(other,_) in enumerate(rank_values):
                    if other_position not in (rank_position,max_position):
                        p=clip(p,value-other,stats)
                if len(p):
                    probability+=tail_mass_with_fallback(p,k,stats)
                    stats["moment_calls"]+=1;stats["correct_pieces"]+=1
    stats["partition_mass"]=partition
    if not (-1e-12<=probability<=1+1e-12):
        raise ArithmeticError("probability outside [0,1]")
    if abs(partition-1)>1e-10:
        raise ArithmeticError(f"gate partition mass {partition}")
    return float(probability),stats


def prefix_summary(model,prefix):
    wi,wh,_=model
    radius=np.linalg.norm(prefix)
    if not radius:
        raise ValueError("zero prefix has probability zero and needs separate handling")
    values=np.asarray(prefix,dtype=float)/radius
    hidden=np.zeros(len(wi))
    for value in values:
        hidden=np.maximum(wh@hidden+wi*value,0)
    indices=np.argsort(values)[::-1][:2]
    return hidden,[(float(values[i]),int(i)) for i in indices]
