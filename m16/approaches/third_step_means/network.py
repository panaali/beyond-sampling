"""Exact stored-weight two-step sectors and third-step linear maps."""
from fractions import Fraction as F
from collections import Counter
from primitive import Exact

def build(wi,wh):
    wi=list(map(F,wi));wh=[list(map(F,row)) for row in wh];h=len(wi)
    assert h>0 and len(wh)==h and all(len(row)==h for row in wh)
    A=Exact();extra=Counter();sectors=[];branches=[]
    for sign in (1,-1):
        first=[max(sign*w,F(0)) for w in wi]
        extra['input_sign_multiplications']+=h;extra['input_positive_part_comparisons']+=h
        a=[A.dot(row,first) for row in wh]
        roots={}
        for j,(aa,b) in enumerate(zip(a,wi)):
            if b:
                # u=|x1|>0, root a*u+b*x2=0. Exact quotient orders rays.
                slope=-aa/b;extra['exact_slope_divisions']+=1;extra['exact_slope_negations']+=1
                roots.setdefault(slope,[]).append(j)
        ordered=sorted(roots);cuts=[(F(0),F(-1))]+[(F(1),q) for q in ordered]+[(F(0),F(1))]
        branches.append({'sign':sign,'a':a,'root_slopes':ordered,'coincident_root_groups':[roots[q] for q in ordered]})
        for index,(lo,hi) in enumerate(zip(cuts,cuts[1:])):
            rep=(A.op('add',lo[0],hi[0]),A.op('add',lo[1],hi[1]))
            whole_halfplane=rep==(F(0),F(0))
            if whole_halfplane:rep=(F(1),F(0))
            assert rep[0]>0
            active=[A.dot((aa,b),rep)>0 for aa,b in zip(a,wi)];extra['activation_comparisons']+=h
            B=[(aa,b) if flag else (F(0),F(0)) for aa,b,flag in zip(a,wi,active)]
            c0=[row[0] for row in B];c1=[row[1] for row in B]
            v=[(A.dot(row,c0),A.dot(row,c1),w) for row,w in zip(wh,wi)]
            normals=[(-lo[1],lo[0],F(0)),(hi[1],-hi[0],F(0))];extra['boundary_normal_negations']+=2
            sectors.append({'sign':sign,'index':index,'lo':lo,'hi':hi,'representative':rep,'whole_halfplane':whole_halfplane,
                            'active':active,'h2_matrix':B,'h3_rows':v,'normals':normals})
    return {'H':h,'wi':wi,'wh':wh,'branches':branches,'sectors':sectors,
            'exact_named_counts':dict(A.count),'extra_named_counts':dict(extra),'retained_result_bits':[A.peak_num,A.peak_den],
            'limits':'Exact comparisons/sorting/dictionary/gcd/control/conversion/transient intermediates and runtime incompletely priced. Quotient operand bits are not included in this reduced-dot-result bit census.'}

def check_maps(certificate):
    """Independent direct recurrence at fixed exact boundary/interior points."""
    A=Exact();wi=certificate['wi'];wh=certificate['wh'];rows=[];forwards=0
    for sector in certificate['sectors']:
        lo,hi=sector['lo'],sector['hi']
        if sector['whole_halfplane']:inside=[(F(1),F(0)),(F(1),F(-1)),(F(1),F(1))]
        else:inside=[tuple(a+b for a,b in zip(lo,hi)),tuple(3*a+b for a,b in zip(lo,hi)),tuple(a+3*b for a,b in zip(lo,hi))]
        for kind,points in [('boundary',[lo,hi]),('interior',inside)]:
            for point in points:
                x1=sector['sign']*point[0];x2=point[1]
                h1=[max(A.op('mul',w,x1),F(0)) for w in wi]
                h2=[max(A.op('add',A.dot(row,h1),A.op('mul',w,x2)),F(0)) for row,w in zip(wh,wi)]
                map2=[A.dot(row,point) for row in sector['h2_matrix']]
                assert h2==map2
                for z in (F(-1),F(0),F(1)):
                    h3=[max(A.op('add',A.dot(row,h2),A.op('mul',w,z)),F(0)) for row,w in zip(wh,wi)]
                    map3=[max(A.dot(row,(*point,z)),F(0)) for row in sector['h3_rows']]
                    forwards+=1
                    rows.append({'sign':sector['sign'],'sector':sector['index'],'kind':kind,'point':point,'z':z,'pass':h3==map3})
                    assert h3==map3
    return {'rows':rows,'direct_analytic_forwards':forwards,'exact_named_counts':dict(A.count),
            'limits':'Validation-only work; point construction/comparisons/control and complete runtime not fully priced.'}
