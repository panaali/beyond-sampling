"""Exact source extraction of the two inherited fixture constructors; no driver."""
from fractions import Fraction as F

def net(B,V,L=None):
    n=dict(B=B,V=V)
    if L is not None:n['L']=L
    return n

def fixtures():
    e=[1,0];f=[0,1];pos=[e,f];neg=[[-1,0],[0,-1]];full=[e,f,[-1,0],[0,-1]];zero=net([], [[]],[[0,0]]);relu=net([e],[[1]]);linear=net([], [[]],[e]);rows=[]
    def add(name,a,b,ra,rb,expected):rows.append(dict(id=name,a=a,b=b,rays_a=ra,rays_b=rb,expected=expected))
    add('identity_crossing',relu,relu,full,full,True)
    add('positive_linear',relu,linear,pos,[e,[1,1]],True)
    add('negative_zero',relu,zero,neg,[[-1,0],[-1,1]],True)
    add('negative_orientation_active',net([[-1,0]],[[1]]),net([], [[]],[[-1,0]]),neg,[[-1,0],[-1,1]],True)
    add('absolute_positive',net([e,[-1,0]],[[1,1]]),linear,pos,[e,[1,1]],True)
    add('absolute_negative',net([e,[-1,0]],[[1,1]]),net([], [[]],[[-1,0]]),neg,[[-1,0],[-1,1]],True)
    add('second_domain_blocks',relu,linear,pos,neg,False)
    add('zero_boundary_hinge',relu,zero,[f],[[0,-1]],True)
    add('lower_dimensional_linear_unknown',linear,zero,[f],[[0,-1]],False)
    add('negative_orientation_identity',net([[-1,0]],[[1]]),net([e],[[1]],[[-1,0]]),full,full,True)
    add('positive_hinge_rescale',net([[2,0]],[[F(1,2)]]),relu,full,full,True)
    add('first_index_full_ties',net([e],[[0],[0]]),net([f],[[0],[0]]),full,pos,True)
    add('shared_linear',net([], [[],[]],[[1,2],[-1,3]]),net([], [[],[]],[[1,2],[-1,3]]),full,neg,True)
    add('different_linear',linear,net([], [[]],[[2,0]]),pos,pos,False)
    add('vector_positive_linear',net([e],[[1],[2]]),net([], [[],[]],[[1,0],[2,0]]),pos,[e,[1,1]],True)
    add('different_hyperplane',relu,net([[1,1]],[[1]]),pos,pos,False)
    assert len(rows)==16;return rows
