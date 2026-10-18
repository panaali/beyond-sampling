from fractions import Fraction as F
from copy import deepcopy


def ident(n): return [[int(i==j) for j in range(n)] for i in range(n)]

def cases():
    base={'H':2,'T':3,'Wi':[0,0],'Wh':ident(2),'Wo':[[1,0],[0,1],[0,0]],'h':[1,0],'rank':[2,-1,0,1]}
    def item(label,m,c,terms,reach='extension-only',prefix=None):
        return {'label':label,'model':m,'expected':{'constant':F(c),'terms':[[F(x),n] for x,n in terms]},'reachability':reach,'prefix':prefix}
    out=[]
    z={'H':1,'T':3,'Wi':[1],'Wh':[[0]],'Wo':[[0],[0],[0]],'h':[0],'rank':[2,1,0,1]}
    out.append(item('zero_readout',z,1,[(2,-1)]))
    m=deepcopy(base);m['h']=[0,1];out.append(item('constant_second',m,1,[(1,-1)]))
    m=deepcopy(base);m['Wo']=[[0,0],[0,0],[1,0]];out.append(item('constant_last',m,-1,[(1,1),(2,1)]))
    m=deepcopy(z);m['Wh']=[[1]];m['Wo']=[[1],[1],[1]];m['h']=[1];out.append(item('identical_logits',m,1,[(2,-1)]))
    m=deepcopy(base);m['Wi']=[1,-1];m['Wh']=[[0,0],[0,0]];m['h']=[0,0];m['rank']=[1,-1,0,1]
    out.append(item('memoryless_signed',m,2,[(1,-2)]))
    m={'H':4,'T':4,'Wi':[1,-1,0,0],'Wh':ident(4),'Wo':[[1,0,0,0],[0,1,0,0],[F(3,4),F(3,4),0,0],[-1,-1,0,0]],'h':[1,1,0,0],'rank':[2,0,0,2]}
    out.append(item('reachable_interior_roots',m,F(1,2),[(F(1,2),1),(2,-1)],'certified synthetic prefix',[2,-1,0]))
    for lab,h in [('weak_equal_first',[1,0]),('weak_equal_second',[0,1])]:
        m=deepcopy(base);m['h']=h;m['rank']=[2,-2,0,1];out.append(item(lab,m,1,[(2,-1)]))
    for lab,h,root in [('jump_left',[1,2],1),('jump_right',[2,1],2)]:
        m=deepcopy(base);m['h']=h;out.append(item(lab,m,1,[(root,-1)]))
    return out


def guards():
    base=cases()[0]['model'];out=[]
    for label in ['bad_shape','negative_hidden','equal_ranks','equal_indices','last_index','boolean_weight']:
        m=deepcopy(base)
        if label=='bad_shape': m['Wh']=[[]]
        if label=='negative_hidden': m['h']=[-1]
        if label=='equal_ranks': m['rank']=[1,1,0,1]
        if label=='equal_indices': m['rank']=[2,1,0,0]
        if label=='last_index': m['rank']=[2,1,0,2]
        if label=='boolean_weight': m['Wi']=[True]
        out.append({'label':label,'model':m})
    return out
