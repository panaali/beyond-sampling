from pathlib import Path
from fractions import Fraction as F
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'global_reset_words'))
from geometry import Geometry

def parameter_certificate(model):
    wi,wh,wo=model;alpha,beta=-F(wi[0]),F(wi[1]);a,b,c,d=F(wh[0][0]),-F(wh[0][1]),F(wh[1][0]),-F(wh[1][1])
    checks={'positive_parameters':min(alpha,beta,a,b,c,d)>0,'bc_gt_ad':b*c>a*d,'a_squared_lt_bc':a*a<b*c,'d_at_most_one':d<=1}
    if not all(checks.values()):raise ValueError(checks)
    axis=[F(row[1]) for row in wo];winner=max(range(len(axis)),key=lambda i:axis[i])
    return {'checks':checks,'winner':winner,'d':d,'beta':beta,'axis_scores':axis}

def uniform_output(model,word):
    wi,wh,wo=model;g=Geometry(wi,wh,seconds=15);cells,trace,word=g.propagate(word)
    first=cells[0];center=tuple(sum((v[i] for v in first[0]),F(0)) for i in range(3))
    state=[g.dot(row,center) for row in first[1]];scores=[g.dot(tuple(map(F,row)),state) for row in wo]
    winner=max(range(len(scores)),key=lambda i:scores[i]);failure=None
    for p,B,_ in cells:
        for competitor,row in enumerate(wo):
            if competitor==winner:continue
            margin=tuple(g.dot((F(wo[winner][0])-F(row[0]),F(wo[winner][1])-F(row[1])),(B[0][j],B[1][j])) for j in range(3))
            for v in p:
                value=g.dot(margin,v)
                if value<0 or (v[0]>0 and value==0 and competitor<winner):
                    failure={'competitor':competitor,'vertex':[str(x) for x in v],'margin':str(value)};break
            if failure:break
        if failure:break
    return {'uniform':failure is None,'winner':winner if failure is None else None,'candidate':winner,'failure':failure,'trace':trace,'stats':g.stats}
