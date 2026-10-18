"""Literal deterministic synthetic fixtures, no random generator."""
import numpy as np


def inputs():
    rows=[list(range(-5,5)),list(range(4,-6,-1)),[0.]*10,[1.]*10,[-1.]*10,
          [1.,-1.]*5,[2.,2.,1.,1.,0.,0.,-1.,-1.,-2.,-2.],
          [j/8-.5 for j in range(10)],[1.]+[0.]*9,[0.]*9+[1.],
          [2.**20,-2.**20,3.,-3.,2.,-2.,1.,-1.,.5,-.5],[-0.,0.]*5]
    x=np.array(rows,dtype=np.float64)
    h=np.empty((12,11,4),dtype=np.float32)
    for r in range(12):
        for s in range(11):
            for j in range(4):h[r,s,j]=(r+1)*(s+1)*(j+1)/16
    return x,h


def selections():
    zero=np.zeros((12,16),dtype=np.float64)
    increasing=np.array([[float(f)+j/32 for j in range(16)] for f in range(12)])
    ties=np.full((12,16),-2.,dtype=np.float64)
    for j in range(16):ties[j%12,j]=1;ties[(j+5)%12,j]=1
    negative=-increasing
    return [('all_tie',zero),('increasing',increasing),('column_ties',ties),('negative',negative)]


def endpoints():
    out=[];n=3
    for mode in ['zero','signed','readout_tie','float32_boundary']:
        a=np.zeros((12,n,16),dtype=np.float64);w=np.zeros((10,16),dtype=np.float64);c=np.array([j%12 for j in range(16)],dtype=np.int64)
        if mode in ('signed','readout_tie'):
            for f in range(12):
                for r in range(n):
                    for j in range(16):a[f,r,j]=(f+1)*(r+1)*(j-7)/16
            w[0,:8]=1;w[1,8:]=1;w[2,8:]=1 if mode=='readout_tie' else -1
        if mode=='float32_boundary':
            a[:,:,0]=16777217.;a[:,:,1]=16777219.;a[:,1,0]=2.**-150;a[:,1,1]=2.**-149;a[:,2,:]=-0.
            w[0,0]=1;w[0,1]=-1;w[1,0]=-1;w[1,1]=1
        out.append((mode,a,c,w))
    return out


def guards():
    # Each case selects a fixed invalid argument, produced without invoking core.
    return ['x_dtype','x_shape','x_infinite','t_zero','t_boolean','family_range','hinge_shape','hinge_dtype','score_nonfinite','choice_range']
