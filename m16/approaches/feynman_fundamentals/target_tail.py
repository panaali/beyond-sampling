"""Two-fresh-input terminal for normalized target/count backward transport."""
import itertools
import numpy as np
from moments import tail_mass_with_fallback
from tail import clip


def target_probability(model,normalized_hidden,target_normalized_value,count,target_index):
    """Return (P(output=j and target has rank2), work statistics).

    State is a=h/R, b=u/R, K=#processed non-target inputs>u (capped at2).
    Exactly two non-target Gaussian inputs remain. The target is already inserted
    for j<=T-3; otherwise it is inserted between/after the two fresh inputs.
    Prefix radius has dimension k=T-2, including the preselected target value.

    For arbitrary grid states this defines an operator extension; some such
    states are unreachable under the original network/input law.
    """
    wi,wh,wo=(np.asarray(w,dtype=float) for w in model)
    H=len(wi);T=len(wo);k=T-2
    a=np.asarray(normalized_hidden,dtype=float)
    b=float(target_normalized_value);j=int(target_index)
    if T<3 or a.shape!=(H,) or np.any(a<0) or not np.all(np.isfinite(a)):
        raise ValueError("need T>=3 and a finite nonnegative hidden state")
    if abs(b)>1+1e-12 or not np.isfinite(b) or j!=target_index or not 0<=j<T:
        raise ValueError("invalid target coordinate/index")
    if count not in (0,1,2):
        raise ValueError("count must be 0,1,2")
    stats=dict(clips=0,affine_products=0,readout_margin_products=0,
               moment_calls=0,gate_cells=0,correct_pieces=0,
               positive_fallbacks=0,positive_quadrature_fallbacks=0,
               high_precision_fallbacks=0,fallback_rule_generations=0,
               fallback_rule_generation_nodes=0,fallback_quadrature_nodes=0,
               fallback_quadrature_error_bound=0.,prefix_dimension=k,
               target_index=j,
               method_class="analytic_regions_with_numerical_quadrature_fallback")
    if count==2:
        stats["partition_mass"]=1.;stats["absorbed_count_two"]=True
        return 0.,stats
    e0,e1,e2=np.eye(3)
    if j<=T-3:
        inputs=[e1,e2];stats["input_order"]=["fresh","fresh"]
    elif j==T-2:
        inputs=[e1,b*e0,e2];stats["input_order"]=["fresh","target","fresh"]
    else:
        inputs=[e1,e2,b*e0];stats["input_order"]=["fresh","fresh","target"]
    B=np.zeros((H,3));B[:,0]=a
    regions=[(np.diag([1.,s1,s2]),B.copy())
             for s1,s2 in itertools.product((-1,1),repeat=2)]
    for input_vector in inputs:
        next_regions=[]
        for polygon,B in regions:
            Z=wh@B+wi[:,None]*input_vector[None,:]
            stats["affine_products"]+=1
            partial=[(polygon,np.zeros_like(B))]
            for neuron in range(H):
                if not np.any(Z[neuron]):
                    continue
                split=[]
                for p,A in partial:
                    plus=clip(p,Z[neuron],stats);minus=clip(p,-Z[neuron],stats)
                    if len(plus):
                        C=A.copy();C[neuron]=Z[neuron];split.append((plus,C))
                    if len(minus):
                        split.append((minus,A.copy()))
                partial=split
            next_regions.extend(partial)
        regions=next_regions
        if len(regions)>20000:
            raise ArithmeticError("tail region cap exceeded")
    stats["gate_cells"]=len(regions)
    # Below-threshold normals are bR-x and bR-y; equality has zero measure.
    below_x=b*e0-e1;below_y=b*e0-e2
    rank_conditions=[(below_x,below_y)] if count==1 else [(-below_x,below_y),(below_x,-below_y)]
    result=0.;partition=0.
    for polygon,B in regions:
        partition+=tail_mass_with_fallback(polygon,k,stats)
        stats["moment_calls"]+=1
        q=polygon
        for other in range(T):
            if other==j:
                continue
            margin=(wo[j]-wo[other])@B
            stats["readout_margin_products"]+=1
            if not np.any(margin):
                if other<j:
                    q=np.empty((0,3));break
            else:
                q=clip(q,margin,stats)
            if not len(q):
                break
        if not len(q):
            continue
        for first,second in rank_conditions:
            p=clip(clip(q,first,stats),second,stats)
            if len(p):
                result+=tail_mass_with_fallback(p,k,stats)
                stats["moment_calls"]+=1;stats["correct_pieces"]+=1
    stats["partition_mass"]=partition
    if abs(partition-1)>1e-10 or not -1e-12<=result<=1+1e-12:
        raise ArithmeticError(f"invalid terminal/partition probability {result}, {partition}")
    return float(result),stats
